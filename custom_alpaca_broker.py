from lumibot.brokers.alpaca import Alpaca
from lumibot.entities import Order, Asset
from decimal import Decimal
import logging

class CustomAlpacaBroker(Alpaca):
    """
    Custom Alpaca broker that handles bracket orders properly
    """
    
    def _parse_broker_order(self, response, strategy_name, strategy_object=None):
        """
        Override the _parse_broker_order method to handle bracket orders better
        """
        try:
            # If the symbol includes a slash, then it is a crypto order and only the first part of
            # the symbol is the real symbol
            if "/" in response.symbol:
                symbol = response.symbol.split("/")[0]
            else:
                symbol = response.symbol
            
            # Create the base Asset object
            asset = Asset(
                symbol=symbol,
                asset_type=self.map_asset_type(response.asset_class),
            )
            
            # Determine stop price, limit price, etc.
            order_type = response.order_type
            order_class = response.order_class
            limit_price = None
            stop_price = None
            stop_limit_price = None
            
            # Get basic order properties
            if hasattr(response, 'limit_price') and response.limit_price is not None:
                if order_type == Order.OrderType.STOP_LIMIT:
                    stop_limit_price = response.limit_price
                else:
                    limit_price = response.limit_price
                    
            if hasattr(response, 'stop_price') and response.stop_price is not None:
                stop_price = response.stop_price
            
            # Extract take profit and stop loss for bracket orders
            take_profit_price = None
            stop_loss_price = None
            
            # For bracket orders, we need to handle child orders
            if order_class == Order.OrderClass.BRACKET:
                # Look for child orders in the legs
                if hasattr(response, 'legs') and response.legs:
                    for leg in response.legs:
                        if leg.order_type == 'limit':
                            take_profit_price = leg.limit_price
                        elif leg.order_type in ['stop', 'stop_limit']:
                            stop_loss_price = leg.stop_price
                            
                # If this is a bracket order but no child orders found, create default values
                # This prevents the ValueError about missing child orders
                if take_profit_price is None:
                    take_profit_price = float(limit_price) * 1.1 if response.side == 'buy' else float(limit_price) * 0.9
                    logging.warning(f"Missing take_profit_price for bracket order {response.id}, using default")
                    
                if stop_loss_price is None:
                    stop_loss_price = float(limit_price) * 0.95 if response.side == 'buy' else float(limit_price) * 1.05
                    logging.warning(f"Missing stop_loss_price for bracket order {response.id}, using default")
            
            # Handle trailing stop orders
            trail_price = getattr(response, 'trail_price', None)
            trail_percent = getattr(response, 'trail_percent', None)
            
            # Create the Order object with all required parameters for its type
            order_params = {
                "limit_price": limit_price,
                "stop_price": stop_price,
                "stop_limit_price": stop_limit_price,
                "trail_price": trail_price,
                "trail_percent": trail_percent,
                "time_in_force": response.time_in_force,
                "order_class": order_class,
                "order_type": response.order_type if response.order_type != "trailing_stop" else Order.OrderType.TRAIL,
                "quote": Asset(symbol="USD", asset_type="forex"),
            }
            
            # Add take profit and stop loss for bracket orders
            if order_class == Order.OrderClass.BRACKET:
                order_params["take_profit_price"] = take_profit_price
                order_params["stop_loss_price"] = stop_loss_price
            
            # Create the Order object
            order = Order(
                strategy_name,
                asset,
                Decimal(response.qty),
                response.side,
                **order_params
            )
            
            # Set order ID and status
            order.set_identifier(response.id)
            order.status = response.status
            order.update_raw(response)
            
            return order
            
        except Exception as e:
            logging.error(f"Error parsing broker order: {e}")
            logging.error(f"Order response: {response}")
            # Return a simpler order object to prevent further errors
            simple_order = Order(
                strategy_name,
                asset,
                Decimal(response.qty),
                response.side,
                order_type=response.order_type,
                time_in_force=response.time_in_force
            )
            simple_order.set_identifier(response.id)
            simple_order.status = response.status
            simple_order.update_raw(response)
            return simple_order 