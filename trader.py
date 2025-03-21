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
from strategy_analyzer import TradingStrategy, get_strategy_with_agent
from position_analyzer import PositionAnalysis, PositionAction, PositionType
from screener import screen_for_swing_trades
from sentiment_analyzer import SentimentAnalysisResult, get_sentiment_analysis_with_agent
from macro_analyzer import get_macro_analysis_with_agent
from market_data_fetcher import MarketData, get_market_data_with_agent
from csv_writer import generate_recommendations_csv

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

class AlpacaTrader:
    def __init__(self, api_key: str, api_secret: str, is_paper: bool = True, is_backtest: bool = False):
        self.logger = get_component_logger("Trader")
        self.trading_client = TradingClient(api_key, api_secret, paper=is_paper)
        self.risk_config = RiskManagementConfig()
        self.trade_expirations: Dict[str, TradeExpiration] = {}
        self.is_backtest = is_backtest

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
                    stop_price=strategy.stop_loss_price
                )
            )
            
            # Place the order and capture the response
            order_response = self.trading_client.submit_order(order_data)
            order_id = order_response.id  # Extract the order ID from the response
            self.logger.info(f"Order placed for {ticker} with ID {order_id}")
            
            # Track the order using the order ID
            self.trade_expirations[ticker] = TradeExpiration(
                ticker=ticker,
                entry_order_id=order_id,
                expiration_date=datetime.now().replace(hour=16, minute=0, second=0),
                strategy=strategy,
                position_type=strategy.position_type
            )
            
            position_type_str = "LONG" if strategy.position_type == PositionType.LONG else "SHORT"
            self.logger.info(f"Placed day bracket order for {ticker} ({position_type_str}):")
            self.logger.info(f"Entry: {shares} shares at ${strategy.entry_price}")
            self.logger.info(f"Stop Loss: ${strategy.stop_loss_price}")
            self.logger.info(f"Take Profit: ${strategy.profit_target}")
            if strategy.position_type == PositionType.SHORT:
                self.logger.info(f"Borrow Cost: {strategy.borrow_cost}%")
                self.logger.info(f"Shares Available: {strategy.shares_available}")
                self.logger.info(f"Short Squeeze Risk: {strategy.short_squeeze_risk}")
            self.logger.info(f"Order expires at market close today")
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
                        self.trading_client.close_position(ticker)
                        
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
                analysis, market_data, sentiment_analysis, macro_analysis = generate_complete_analysis(ticker)
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

def generate_complete_analysis(ticker: str) -> Tuple[TradingStrategy, MarketData, SentimentAnalysisResult, Any]:
    """Generate a complete analysis for a stock"""
    logger = get_component_logger("CompleteAnalysis")
    logger.info(f"Generating complete analysis for {ticker}")
    
    start_time = time.time()

    try:
        # 1. Get market data
        market_data = get_market_data_with_agent(ticker)
        logger.info(f"Current price for {ticker}: ${market_data.latest_price:.2f}" if market_data.latest_price else "Price data not available")
            
        # 2. Get sentiment analysis
        sentiment_analysis = get_sentiment_analysis_with_agent(ticker)
        logger.info(f"Completed sentiment analysis for {ticker}")
        
        # 3. Get macro analysis
        macro_analysis = get_macro_analysis_with_agent(ticker)
        logger.info(f"Completed macro analysis for {ticker}")

        # 4. Get trading strategy
        strategy = get_strategy_with_agent(
            ticker=ticker,
            market_data=market_data,
            sentiment_analysis=sentiment_analysis,
            macro_analysis=macro_analysis
        )
        logger.info(f"Completed trading strategy for {ticker}")

        elapsed_time = time.time() - start_time
        logger.info(f"Complete analysis for {ticker} completed in {elapsed_time:.2f} seconds")

        return strategy, market_data, sentiment_analysis, macro_analysis
        
    except Exception as e:
        logger.error(f"Error generating complete analysis for {ticker}: {e}")
        logger.error(traceback.format_exc())
        raise

def run_trader(ticker: str = None):
    """Main trading function"""
    logger = get_component_logger("Trader")
    
    trader = None
    try:
        trader = AlpacaTrader(
            api_key=os.environ["ALPACA_API_KEY_UNITTEST"],
            api_secret=os.environ["ALPACA_API_SECRET_UNITTEST"],
            is_paper=True,
            is_backtest=True
        )

        # Get account state
        account_state = trader.get_account_state()

        # First, analyze and manage any existing positions
        existing_positions = account_state["positions"]
        if existing_positions:
            logger.info("\n======== Analyzing Existing Positions ========")
            position_analyses = {}
            
            for position in existing_positions:
                try:
                    # Get complete analysis for the position
                    strategy, market_data, _, _ = generate_complete_analysis(position.symbol)
                    
                    # Create position analysis
                    position_analysis = PositionAnalysis(
                        ticker=position.symbol,
                        current_price=float(position.current_price),
                        purchase_price=float(position.avg_entry_price),
                        purchase_date=datetime.now(),  # Use current date since we don't have purchase date
                        days_held=0,  # We don't have the actual purchase date, so start with 0
                        recommendation=PositionAction.HOLD,  # Default to HOLD, will be updated based on analysis
                        stop_loss=strategy.stop_loss_price,
                        target_price=strategy.profit_target,
                        summary=strategy.summary,
                        explanation=strategy.explanation
                    )
                    
                    # Determine recommendation based on analysis
                    if float(position.current_price) <= strategy.stop_loss_price:
                        position_analysis.recommendation = PositionAction.EXIT
                    elif float(position.current_price) >= strategy.profit_target:
                        position_analysis.recommendation = PositionAction.EXIT
                    
                    position_analyses[position.symbol] = position_analysis
                    
                    logger.info(f"\nPosition Analysis for {position.symbol}:")
                    logger.info(f"Days Held: {position_analysis.days_held}")
                    logger.info(f"Entry Price: ${float(position.avg_entry_price):.2f}")
                    logger.info(f"Current Price: ${float(position.current_price):.2f}")
                    logger.info(f"P/L: ${float(position.unrealized_pl):.2f} ({float(position.unrealized_plpc) * 100:.1f}%)")
                    logger.info(f"Recommendation: {position_analysis.recommendation}")
                    logger.info(f"New Stop Loss: ${position_analysis.stop_loss:.2f}")
                    logger.info(f"New Target: ${position_analysis.target_price:.2f}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing position for {position.symbol}: {e}")
                    continue
            
            # Manage positions based on analysis
            if position_analyses:
                trader.manage_existing_positions(position_analyses)

            # Get fresh account state after managing positions
            account_state = trader.get_account_state()

        # Now check account criteria for new trades
        if float(account_state["equity"]) < trader.risk_config.min_account_balance:
            logger.info(f"Account balance (${float(account_state['equity']):.2f}) below minimum requirement (${trader.risk_config.min_account_balance})")
            return
            
        if float(account_state["buying_power"]) < trader.risk_config.min_available_cash:
            logger.info(f"Available cash (${float(account_state['buying_power']):.2f}) below minimum requirement (${trader.risk_config.min_available_cash})")
            return
            
        # Check position limits
        if len(account_state["positions"]) >= trader.risk_config.max_concurrent_positions:
            logger.info(f"Maximum number of positions ({trader.risk_config.max_concurrent_positions}) reached")
            return
        
        # Then proceed with new trade analysis
        if ticker:
            # Analyze specific ticker
            logger.info(f"Analyzing specific ticker: {ticker}")
            strategy, market_data, _, _ = generate_complete_analysis(ticker)
            analyses = {ticker: strategy}
        else:
            # Run full analysis
            analyses = trader.run_analysis()
        
        # Place new trades for qualifying strategies
        for ticker, strategy in analyses.items():
            if trader.can_place_new_trade(strategy, ticker):
                trader.place_new_trade(strategy, ticker)
                
    except Exception as e:
        logger.error(f"Error in trader execution: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stock analysis and trading tool")
    parser.add_argument("--ticker", type=str, help="Specific ticker to analyze")
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
    
    # Run trader
    run_trader(args.ticker) 