"""
Automated Trading Module

Handles automated trading based on analysis recommendations with risk management rules.
Supports both long and short positions using entry prices, stops, and targets from analysis.
Places trades once per day using bracket orders.
"""

import logging
import os
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopLossRequest, TakeProfitRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, OrderStatus, QueryOrderStatus, OrderClass

from logging_utils import get_component_logger, setup_logging
from strategy_analyzer import TradingStrategy
from position_analyzer import PositionAnalysis, PositionAction, PositionType
from screener import screen_for_swing_trades
from analysis_generator import generate_llm_strategy_analysis
from csv_writer import generate_recommendations_csv
from trading_journal import TradingJournal, TradeRecord, TradeStatus

@dataclass
class RiskManagementConfig:
    """Risk management configuration"""
    max_position_size_percent: float = 0.10  # 10% of account
    min_position_size: int = 1000            # $1,000 minimum position
    max_concurrent_positions: int = 5
    min_account_balance: int = 25000
    min_available_cash: int = 10000

@dataclass
class TradeExpiration:
    """Track trade details"""
    ticker: str
    entry_order_id: str
    expiration_date: datetime
    strategy: TradingStrategy
    position_type: PositionType
    trade_id: Optional[str] = None

class AlpacaTrader:
    def __init__(self, api_key: str, api_secret: str, is_paper: bool = True, is_backtest: bool = False, backtest_date=None):
        self.logger = get_component_logger("Trader")
        self.trading_client = TradingClient(api_key, api_secret, paper=is_paper)
        self.risk_config = RiskManagementConfig()
        self.trade_expirations: Dict[str, TradeExpiration] = {}
        self.is_backtest = is_backtest
        self.backtest_date = backtest_date
        
        # Initialize journal with backtest info if needed
        self.journal = TradingJournal(
            backtest_mode=is_backtest,
            backtest_date=backtest_date
        )

    def get_account_state(self) -> Dict:
        """Get current account state including positions"""
        account = self.trading_client.get_account()
        positions = self.trading_client.get_all_positions()
        orders = self.trading_client.get_orders()
        
        return {
            "buying_power": float(account.buying_power),
            "equity": float(account.equity),
            "positions": positions,
            "orders": orders
        }
    
    def can_place_new_trade(self, strategy: TradingStrategy, ticker: str) -> bool:
        """Determine if a new trade can be placed based on strategy and risk rules"""
        account_state = self.get_account_state()
        
        # Check account minimums
        if float(account_state["equity"]) < self.risk_config.min_account_balance:
            self.logger.info("Account balance below minimum requirement")
            return False
            
        if float(account_state["buying_power"]) < self.risk_config.min_available_cash:
            self.logger.info("Available cash below minimum requirement")
            return False
            
        # Check position limits
        if len(account_state["positions"]) >= self.risk_config.max_concurrent_positions:
            self.logger.info("Maximum number of positions reached")
            return False
            
        # Check risk/reward ratio from strategy
        if strategy.position_type == PositionType.LONG:
            risk = strategy.entry_price - strategy.stop_loss_price
            reward = strategy.profit_target - strategy.entry_price
        else:  # SHORT
            risk = strategy.stop_loss_price - strategy.entry_price
            reward = strategy.entry_price - strategy.profit_target
            
            # Additional checks for short positions
            if not strategy.shares_available or strategy.shares_available <= 0:
                self.logger.info("No shares available to short")
                return False
                
            if strategy.borrow_cost and strategy.borrow_cost > 5.0:  # 5% annual borrow cost threshold
                self.logger.info(f"Borrow cost too high: {strategy.borrow_cost}%")
                return False
                
        if (reward / risk) < 2:
            self.logger.info("Risk/reward ratio below 2:1")
            return False
            
        return True
    
    def calculate_position_size(self, strategy: TradingStrategy) -> int:
        """Calculate number of shares based on risk management rules"""
        account_state = self.get_account_state()
        equity = float(account_state["equity"])
        
        # Calculate position size based on percentage of account
        max_position_value = equity * self.risk_config.max_position_size_percent
        
        # Calculate shares
        shares = int(max_position_value / strategy.entry_price)
        position_value = shares * strategy.entry_price
        
        # Check minimum position size
        if position_value < self.risk_config.min_position_size:
            return 0
            
        return shares
    
    def cancel_existing_orders(self, symbol: str) -> None:
        """Cancel all existing orders for a given symbol"""
        try:
            # Create filter request
            filter_request = GetOrdersRequest(
                symbol=symbol,
                status=QueryOrderStatus.OPEN
            )
            
            open_orders = self.trading_client.get_orders(filter=filter_request)
            for order in open_orders:
                self.logger.info(f"Canceling existing order {order.id} for {symbol}")
                self.trading_client.cancel_order_by_id(order.id)
                
                # Find the trade ID if available
                trade_id = None
                for exp in self.trade_expirations.values():
                    if exp.ticker == symbol and exp.entry_order_id == order.id:
                        trade_id = exp.trade_id
                        break
                
                # Record cancellation in journal if we have the trade ID
                if trade_id:
                    self.journal.cancel_trade(
                        trade_id=trade_id,
                        reason="Order cancelled before execution"
                    )
                
        except Exception as e:
            self.logger.error(f"Error canceling orders for {symbol}: {e}")
            raise

    def place_new_trade(self, strategy: TradingStrategy, ticker: str) -> bool:
        """Place a new trade based on strategy analysis using bracket orders"""
        try:
            if not self.can_place_new_trade(strategy, ticker):
                return False
                
            # Check for existing position
            try:
                position = self.trading_client.get_position(ticker)
                self.logger.info(f"Already have position in {ticker}, skipping new trade")
                return False
            except Exception:
                # No position exists, continue with trade
                pass
                
            # Log the type of trade being made
            trade_type = "LONG" if strategy.position_type == PositionType.LONG else "SHORT"
            self.logger.info(f"Placing a {trade_type} trade for {ticker}")
                
            # Cancel any existing orders
            self.cancel_existing_orders(ticker)
                
            shares = self.calculate_position_size(strategy)
            if shares == 0:
                self.logger.info("Position size too small")
                return False
                
            # Check for sufficient funds for short
            if strategy.position_type == PositionType.SHORT and not self.has_sufficient_funds_for_short(ticker, shares):
                self.logger.info(f"Insufficient funds to short {ticker}")
                return False
                
            # Create bracket order that expires at end of day
            order_data = LimitOrderRequest(
                symbol=ticker,
                qty=shares,
                side=OrderSide.BUY if strategy.position_type == PositionType.LONG else OrderSide.SELL,
                type=OrderType.LIMIT,
                limit_price=strategy.entry_price,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=dict(
                    limit_price=strategy.profit_target
                ),
                stop_loss=dict(
                    stop_price=strategy.stop_loss_price,
                    limit_price=strategy.stop_loss_price * 0.99 if strategy.position_type == PositionType.LONG else strategy.stop_loss_price * 1.01  # Add a 1% buffer for the limit price
                )
            )
            
            # Place the order and capture the response
            order_response = self.trading_client.submit_order(order_data)
            order_id = order_response.id

            # Record the trade in the journal
            trade_id = self.journal.record_new_trade(
                ticker=ticker,
                strategy=strategy,
                order_id=order_id,
                shares=shares
            )

            # Store the trade_id for future reference
            self.trade_expirations[ticker] = TradeExpiration(
                ticker=ticker,
                entry_order_id=order_id, 
                expiration_date=datetime.now().replace(hour=16, minute=0, second=0),
                strategy=strategy,
                position_type=strategy.position_type,
                trade_id=trade_id
            )
            
            position_type_str = "LONG" if strategy.position_type == PositionType.LONG else "SHORT"
            
            # Enhanced logging for bracket orders
            self.logger.info("=" * 50)
            self.logger.info(f"BRACKET ORDER PLACED: {ticker} ({position_type_str}) - ORDER ID: {order_id}")
            self.logger.info("-" * 50)
            self.logger.info(f"Entry:      {shares} shares at ${strategy.entry_price:.2f}")
            
            # Calculate risk and reward in dollars
            if strategy.position_type == PositionType.LONG:
                risk_per_share = strategy.entry_price - strategy.stop_loss_price
                reward_per_share = strategy.profit_target - strategy.entry_price
            else:  # SHORT
                risk_per_share = strategy.stop_loss_price - strategy.entry_price
                reward_per_share = strategy.entry_price - strategy.profit_target
                
            total_risk = risk_per_share * shares
            total_reward = reward_per_share * shares
            
            self.logger.info(f"Stop Loss:  ${strategy.stop_loss_price:.2f} (Risk: ${risk_per_share:.2f}/share, Total: ${total_risk:.2f})")
            self.logger.info(f"Take Profit: ${strategy.profit_target:.2f} (Reward: ${reward_per_share:.2f}/share, Total: ${total_reward:.2f})")
            self.logger.info(f"Risk/Reward: {strategy.risk_reward}")
            self.logger.info(f"Expiration: {datetime.now().replace(hour=16, minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')}")
            
            if strategy.position_type == PositionType.SHORT:
                self.logger.info(f"Borrow Cost: {strategy.borrow_cost}%")
                self.logger.info(f"Shares Available: {strategy.shares_available}")
                self.logger.info(f"Short Squeeze Risk: {strategy.short_squeeze_risk}")
            
            self.logger.info("=" * 50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error placing trade for {ticker}: {e}")
            return False

    def cleanup_expired_orders(self) -> None:
        """Cancel any orders that haven't filled by end of day"""
        try:
            filter_request = GetOrdersRequest(
                status=QueryOrderStatus.OPEN
            )
            
            open_orders = self.trading_client.get_orders(filter=filter_request)
            for order in open_orders:
                self.logger.info(f"Canceling unfilled order {order.id} for {order.symbol} at end of day")
                try:
                    self.trading_client.cancel_order_by_id(order.id)
                except Exception as e:
                    self.logger.error(f"Error canceling order {order.id}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up expired orders: {e}")

    def cleanup(self) -> None:
        """Cleanup method to be called when shutting down"""
        self.cleanup_expired_orders()

    def manage_existing_positions(self, position_analyses: Dict[str, PositionAnalysis]) -> None:
        """Review and manage existing positions based on position analysis"""
        try:
            positions = self.trading_client.get_all_positions()
            
            for position in positions:
                ticker = position.symbol
                if ticker in position_analyses:
                    analysis = position_analyses[ticker]
                    
                    # For now, we're only handling EXIT recommendations
                    if analysis.recommendation == PositionAction.EXIT:
                        self.logger.info(f"Closing position in {ticker} based on analysis")
                        
                        # Get the position details before closing
                        position = self.trading_client.get_position(ticker)
                        close_price = float(position.current_price)
                        
                        # Close the position
                        close_response = self.trading_client.close_position(ticker)
                        
                        # Find the trade ID if available
                        trade_id = None
                        for exp in self.trade_expirations.values():
                            if exp.ticker == ticker:
                                trade_id = exp.trade_id
                                break
                        
                        # Record in the journal if we have the trade ID
                        if trade_id:
                            self.journal.close_trade(
                                trade_id=trade_id,
                                exit_price=close_price,
                                exit_order_id=getattr(close_response, 'id', 'market_close'),
                                exit_reason=f"Position exited based on analysis: {analysis.recommendation}"
                            )
                        
        except Exception as e:
            self.logger.error(f"Error managing positions: {e}")

    def run_analysis(self) -> Dict[str, TradingStrategy]:
        """Run the stock analysis process"""
        logger = get_component_logger("Analysis")
        
        # Dictionary to store all analysis results
        all_analyses = {}
        all_market_data = {}
        all_sentiment_analysis = {}
        all_macro_analysis = {}
        
        # Step 1: Screen for top swing trading candidates
        logger.info("\n======== Screening for Top Swing Trading Candidates ========")
        max_stocks = 1 if self.is_backtest else 5
        screener_results = screen_for_swing_trades(max_stocks=max_stocks, is_backtest=self.is_backtest)
        stock_candidates = screener_results.candidates
        
        # Extract tickers for analysis (both long and short)
        top_tickers = [candidate.ticker for candidate in stock_candidates]
        
        logger.info(f"\nTop candidates: {', '.join(top_tickers)}")
            
        # Step 2: Run complete analyses for each candidate
        logger.info("\n======== Running Complete Analyses ========")
        for ticker in top_tickers:
            try:
                logger.info(f"\n----- Starting Complete Analysis for {ticker} -----\n")
                start_time = time.time()
                analysis, market_data, sentiment_analysis, macro_analysis = generate_llm_strategy_analysis(ticker)
                all_analyses[ticker] = analysis
                all_market_data[ticker] = market_data
                all_sentiment_analysis[ticker] = sentiment_analysis
                all_macro_analysis[ticker] = macro_analysis
                elapsed_time = time.time() - start_time
                logger.info(f"Analysis for {ticker} completed in {elapsed_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        # Step 3: Generate a CSV file with all recommendations
        logger.info("\n======== Generating Recommendations CSV ========")
        csv_file = generate_recommendations_csv(all_analyses, all_market_data)
        logger.info(f"Recommendations saved to {csv_file}")
        
        logger.info("\n======== Analysis Complete ========")
        return all_analyses

    def has_sufficient_funds_for_short(self, ticker: str, shares: int) -> bool:
        """Check if there are sufficient funds to execute a short position"""
        # Implement logic to check for sufficient funds
        # This is a placeholder implementation
        account_state = self.get_account_state()
        required_margin = self.calculate_required_margin_for_short(ticker, shares)
        return account_state['buying_power'] >= required_margin

    def calculate_required_margin_for_short(self, ticker: str, shares: int) -> float:
        """Calculate the required margin for a short position"""
        # Implement logic to calculate required margin
        # This is a placeholder implementation
        return shares * 10.0  # Example calculation

    def show_performance(self) -> None:
        """Display trading performance metrics"""
        performance = self.journal.calculate_performance()
        
        self.logger.info("\n=== Trading Performance ===")
        self.logger.info(f"Total Trades: {performance['total_trades']}")
        self.logger.info(f"Winning Trades: {performance['winning_trades']}")
        self.logger.info(f"Win Rate: {performance['win_rate']:.1f}%")
        self.logger.info(f"Total P/L: ${performance['profit_loss']:.2f}")
        self.logger.info(f"Avg P/L per Trade: ${performance['avg_profit_per_trade']:.2f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stock analysis and trading tool")
    parser.add_argument("--demo", action="store_true", default=True, help="Run a demo of trading capabilities")
    parser.add_argument("--ticker", type=str, help="Specific ticker to analyze or trade")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Set the logging level")
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    setup_logging(
        log_file=f"trader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        console_level=log_level
    )
    
    logger = get_component_logger("TradingDemo")
    
    if args.demo:
        logger.info("====== Running Trader Demo ======")
        # Create trader instance
        try:
            trader = AlpacaTrader(
                api_key=os.environ["ALPACA_API_KEY_UNITTEST"],
                api_secret=os.environ["ALPACA_API_SECRET_UNITTEST"],
                is_paper=True,
                is_backtest=False
            )
            
            # 1. Demo: Get account state
            logger.info("\n=== Account State ===")
            account_state = trader.get_account_state()
            logger.info(f"Account Equity: ${float(account_state['equity']):.2f}")
            logger.info(f"Buying Power: ${float(account_state['buying_power']):.2f}")
            logger.info(f"Current Positions: {len(account_state['positions'])}")
            logger.info(f"Current Orders: {len(account_state['orders'])}")
            
            # 2. Demo: Place a bracket order (LONG)
            logger.info("\n=== Placing a Demo LONG Bracket Order ===")
            # Create a demo strategy for a LONG position
            ticker = args.ticker or "AAPL"  # Default to AAPL if no ticker provided
            
            # Get current market price
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            
            # Get current price from Alpaca
            data_client = StockHistoricalDataClient(
                api_key=os.environ["ALPACA_API_KEY_UNITTEST"],
                secret_key=os.environ["ALPACA_API_SECRET_UNITTEST"]
            )
            latest_quote_request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            try:
                latest_quote = data_client.get_stock_latest_quote(latest_quote_request)
                current_price = float(latest_quote[ticker].ask_price)
                logger.info(f"Current price for {ticker}: ${current_price:.2f}")
            except Exception as e:
                logger.warning(f"Could not get latest quote: {e}")
                current_price = 150.0  # Fallback for demo
                logger.info(f"Using demo price for {ticker}: ${current_price:.2f}")
                
            # Create demo strategy
            from strategy_analyzer import TradingStrategy, PositionType
            demo_strategy = TradingStrategy(
                position_type=PositionType.LONG,
                expiration_date=datetime.now().strftime("%Y-%m-%d 16:00:00"),
                entry_price=current_price * 0.99,  # Slightly below current price
                stop_loss_price=current_price * 0.95,  # 5% below current price
                profit_target=current_price * 1.05,  # 5% above current price
                risk_reward="1:1",
                summary="Demo LONG trade for bracket order testing",
                explanation="This is a test trade to demonstrate bracket orders"
            )
            
            # Place the trade
            trade_result = trader.place_new_trade(demo_strategy, ticker)
            logger.info(f"Trade placement result: {'Success' if trade_result else 'Failed'}")
            
            # 3. Demo: Show open orders
            logger.info("\n=== Current Open Orders ===")
            filter_request = GetOrdersRequest(
                status=QueryOrderStatus.OPEN
            )
            open_orders = trader.trading_client.get_orders(filter=filter_request)
            for order in open_orders:
                logger.info(f"Order ID: {order.id}")
                logger.info(f"Symbol: {order.symbol}")
                logger.info(f"Side: {order.side}")
                logger.info(f"Type: {order.type}")
                logger.info(f"Qty: {order.qty}")
                logger.info(f"Status: {order.status}")
                logger.info(f"Class: {order.order_class}")
                logger.info(f"Time in Force: {order.time_in_force}")
                logger.info("---")
            
            if not open_orders:
                logger.info("No open orders found.")
            
            # 3b. Demo: Check existing positions and close one if it exists
            logger.info("\n=== Current Positions ===")
            positions = trader.trading_client.get_all_positions()
            for position in positions:
                logger.info(f"Position: {position.symbol}")
                logger.info(f"Quantity: {position.qty} shares")
                logger.info(f"Side: {'LONG' if float(position.qty) > 0 else 'SHORT'}")
                logger.info(f"Average Entry: ${float(position.avg_entry_price):.2f}")
                logger.info(f"Current Price: ${float(position.current_price):.2f}")
                logger.info(f"P/L: ${float(position.unrealized_pl):.2f} ({float(position.unrealized_plpc) * 100:.2f}%)")
                logger.info("---")
            
            if not positions:
                logger.info("No positions found.")
            else:
                # Demo closing a position with market order
                logger.info("\n=== Closing a Position (Market Order) ===")
                position_to_close = positions[0]  # Take the first position
                ticker_to_close = position_to_close.symbol
                
                # Create a simple position analysis for exit
                position_analysis = PositionAnalysis(
                    ticker=ticker_to_close,
                    current_price=float(position_to_close.current_price),
                    purchase_price=float(position_to_close.avg_entry_price),
                    purchase_date=datetime.now(),
                    days_held=0,
                    recommendation=PositionAction.EXIT,
                    stop_loss=float(position_to_close.avg_entry_price) * 0.9,  # Demo values
                    target_price=float(position_to_close.avg_entry_price) * 1.1,  # Demo values
                    summary="Demo position being closed",
                    explanation="This is a test of the market order for closing positions",
                    position_type=PositionType.LONG if float(position_to_close.qty) > 0 else PositionType.SHORT
                )
                
                logger.info(f"Closing position in {ticker_to_close} with market order")
                try:
                    # Close the position
                    trader.trading_client.close_position(ticker_to_close)
                    logger.info(f"Successfully closed position in {ticker_to_close}")
                except Exception as e:
                    logger.error(f"Error closing position: {e}")
            
            # 4. Demo: Cancel orders (if any)
            if open_orders:
                logger.info("\n=== Cancelling Orders ===")
                for order in open_orders:
                    logger.info(f"Cancelling order {order.id} for {order.symbol}")
                    try:
                        trader.trading_client.cancel_order_by_id(order.id)
                        logger.info("Order cancelled successfully")
                    except Exception as e:
                        logger.error(f"Error cancelling order: {e}")
            
            # 5. Demo: Place a SHORT bracket order (if requested)
            if args.ticker:  # Only do this if user specified a ticker
                logger.info("\n=== Placing a Demo SHORT Bracket Order ===")
                # Create a demo strategy for a SHORT position
                short_strategy = TradingStrategy(
                    position_type=PositionType.SHORT,
                    expiration_date=datetime.now().strftime("%Y-%m-%d 16:00:00"),
                    entry_price=current_price * 1.01,  # Slightly above current price
                    stop_loss_price=current_price * 1.05,  # 5% above current price
                    profit_target=current_price * 0.95,  # 5% below current price
                    risk_reward="1:1",
                    summary="Demo SHORT trade for bracket order testing",
                    explanation="This is a test trade to demonstrate SHORT bracket orders",
                    borrow_cost=1.5,  # Demo borrow cost
                    shares_available=1000  # Demo shares available
                )
                
                # Place the trade
                short_result = trader.place_new_trade(short_strategy, ticker)
                logger.info(f"SHORT trade placement result: {'Success' if short_result else 'Failed'}")
            
            # 6. Demo cleanup at the end
            logger.info("\n=== Cleaning Up Demo ===")
            trader.cleanup_expired_orders()
            logger.info("Demo complete!")
            
            # Include in demo or after running trades
            logger.info("\n=== Trading Journal Performance ===")
            trader.show_performance()
            
        except Exception as e:
            logger.error(f"Error in trader demo: {e}")
            logger.error(traceback.format_exc())
    else:
        # Just show help if no specific parameters
        print("\nTrader Module - Trading Capabilities Demo")
        print("=========================================")
        print("Run the demo with: python trader.py --demo")
        print("  Add a specific ticker: python trader.py --demo --ticker MSFT")
        print("  For more options: python trader.py --help")
        print("\nThe demo will show:")
        print("1. Current account state")
        print("2. Placing a LONG bracket order")
        print("3. Viewing open orders")
        print("4. Closing positions with market orders")
        print("5. Cancelling open orders")
        print("6. Placing a SHORT bracket order (if ticker provided)") 