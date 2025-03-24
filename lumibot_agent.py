from datetime import datetime, timedelta
import traceback
# from lumibot.brokers import Alpaca  # Not needed anymore, using CustomAlpacaBroker
from lumibot.strategies import Strategy
from lumibot.traders import Trader
import pandas as pd
import os
from logging_utils import get_component_logger, setup_logging
from position_analyzer import PositionAction, PositionAnalysis, PositionType, get_position_analysis
from trader import AlpacaTrader, RiskManagementConfig
from analysis_generator import generate_llm_strategy_analysis
from lumibot.backtesting import PolygonDataBacktesting
import pytz
from custom_alpaca_broker import CustomAlpacaBroker
from typing import Dict
import shutil

# Helper function to check if we're in backtest mode
def get_is_backtest():
    return os.environ.get("BACKTEST", "false").lower() == "true"

# Clean up previous backtest files if in backtest mode
if get_is_backtest():
    dirs_to_clean = ['./journal', './logs', './output']
    for dir_path in dirs_to_clean:
        try:
            shutil.rmtree(dir_path, ignore_errors=True)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Cleaned and recreated directory: {dir_path}")
        except Exception as e:
            print(f"Error cleaning directory {dir_path}: {e}")

# Set up logging
setup_logging()

class ThreePhaseStrategy(Strategy):

    def initialize(self):
        # Trading schedule
        self.sleeptime = "15M"
        self.risk_config = RiskManagementConfig()
        self.logger = get_component_logger("Lumibot")
        self.is_backtest = get_is_backtest()
        
        # In backtest mode, pass the current datetime to the trader
        backtest_date = self.get_datetime() if self.is_backtest else None
        
        self.trader = AlpacaTrader(
            api_key=os.environ["ALPACA_API_KEY_LLM_AGENT"],
            api_secret=os.environ["ALPACA_API_SECRET_LLM_AGENT"],
            is_paper=os.environ.get("ALPACA_PAPER_TRADING_LLM_AGENT", "true").lower() == "true",
            is_backtest=self.is_backtest,
            backtest_date=backtest_date
        )
        

    def before_starting_trading(self):
        self.logger.info(f"Executing before starting trading at {self.get_current_datetime().strftime('%Y-%m-%d %H:%M:%S %Z')} EST...")

        self.morning_run_complete = False
        self.midday_run_complete = False
        self.end_day_run_complete = False


    def on_trading_iteration(self):
        current_time = self.get_datetime()
        
        # Update backtest date if in backtest mode
        if self.is_backtest:
            self.trader.backtest_date = current_time
            self.trader.journal.backtest_date = current_time
        
        if current_time.hour == 9 and not self.morning_run_complete:
            self.logger.info(f"Executing morning strategy at {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')} EST...")
            self.morning_run_complete = True
            self.execute_morning_strategy()

        if current_time.hour == 12 and not self.midday_run_complete:
            self.logger.info(f"Executing mid-day position management at {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')} EST...")
            self.midday_run_complete = True
            self.execute_midday_strategy()            

        if current_time.hour == 4 and not self.end_day_run_complete:
            self.logger.info(f"Executing end of day cleanup at {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')} EST...")
            self.end_day_run_complete = True
            self.execute_end_day_strategy()


    def on_canceled_order(self, order):
        self.logger.info(f"Order for {order.asset} canceled")


    def execute_existing_positions(self):
        """Manage existing positions"""

        # Get account state
        account_state = self.trader.get_account_state()

        # First, analyze and manage any existing positions
        existing_positions = account_state["positions"]
        if not existing_positions:
            return
        
        self.logger.info("\n======== Analyzing Existing Positions ========")
        position_analyses = {}
        
        for position in existing_positions:
            try:
                # Get complete analysis for the position
                strategy, market_data, _, _ = generate_llm_strategy_analysis(position.symbol)
                
                # Create position analysis
                position_analysis = PositionAnalysis(
                    ticker=position.symbol,
                    current_price=float(position.current_price),
                    purchase_price=float(position.avg_entry_price),
                    purchase_date=self.get_current_datetime(),  # Use current date since we don't have purchase date
                    days_held=0,  # We don't have the actual purchase date, so start with 0
                    recommendation=PositionAction.HOLD,  # Default to HOLD, will be updated based on analysis
                    stop_loss=strategy.stop_loss_price,
                    target_price=strategy.profit_target,
                    summary=strategy.summary,
                    explanation=strategy.explanation,
                    position_type=strategy.position_type,
                    borrow_cost=strategy.borrow_cost if strategy.position_type == PositionType.SHORT else None,
                    short_squeeze_risk=strategy.short_squeeze_risk if strategy.position_type == PositionType.SHORT else None
                )
            
                # Determine recommendation based on analysis
                if strategy.position_type == PositionType.LONG:
                    if float(position.current_price) <= strategy.stop_loss_price:
                        position_analysis.recommendation = PositionAction.EXIT
                    elif float(position.current_price) >= strategy.profit_target:
                        position_analysis.recommendation = PositionAction.EXIT
                else:  # SHORT
                    if float(position.current_price) >= strategy.stop_loss_price:
                        position_analysis.recommendation = PositionAction.EXIT
                    elif float(position.current_price) <= strategy.profit_target:
                        position_analysis.recommendation = PositionAction.EXIT
                
                position_analyses[position.symbol] = position_analysis
            
                self.logger.info(f"\nPosition Analysis for {position.symbol} ({strategy.position_type}):")
                self.logger.info(f"Days Held: {position_analysis.days_held}")
                self.logger.info(f"Entry Price: ${float(position.avg_entry_price):.2f}")
                self.logger.info(f"Current Price: ${float(position.current_price):.2f}")
                self.logger.info(f"P/L: ${float(position.unrealized_pl):.2f} ({float(position.unrealized_plpc) * 100:.1f}%)")
                self.logger.info(f"Recommendation: {position_analysis.recommendation}")
                self.logger.info(f"New Stop Loss: ${position_analysis.stop_loss:.2f}")
                self.logger.info(f"New Target: ${position_analysis.target_price:.2f}")
                if strategy.position_type == PositionType.SHORT:
                    self.logger.info(f"Borrow Cost: {position_analysis.borrow_cost}%")
                    self.logger.info(f"Short Squeeze Risk: {position_analysis.short_squeeze_risk}")
            
            except Exception as e:
                self.logger.error(f"Error analyzing position for {position.symbol}: {e}")
                continue
        
        # Manage positions based on analysis
        if position_analyses:
            self.trader.manage_existing_positions(position_analyses)


    def execute_morning_strategy(self):
        """Pre-market analysis and trade setup"""

        try:
            # Check account state
            account_state = self.trader.get_account_state()
            
            # Check account minimums
            if account_state["equity"] < self.risk_config.min_account_balance:
                self.logger("Account balance below minimum requirement")
                return
                
            if account_state["buying_power"] < self.risk_config.min_available_cash:
                self.logger("Available cash below minimum requirement")
                return
                
            # Check position limits
            if len(account_state["positions"]) >= self.risk_config.max_concurrent_positions:
                self.logger("Maximum number of positions reached")
                return
            
            self.execute_existing_positions()

            # Get fresh account state after managing positions
            account_state = self.trader.get_account_state()

            # Now check account criteria for new trades
            if float(account_state["equity"]) < self.trader.risk_config.min_account_balance:
                self.logger.info(f"Account balance (${float(account_state['equity']):.2f}) below minimum requirement (${trader.risk_config.min_account_balance})")
                return
            
            if float(account_state["buying_power"]) < self.trader.risk_config.min_available_cash:
                self.logger.info(f"Available cash (${float(account_state['buying_power']):.2f}) below minimum requirement (${self.trader.risk_config.min_available_cash})")
                return
            
            # Check position limits
            if len(account_state["positions"]) >= self.trader.risk_config.max_concurrent_positions:
                self.logger.info(f"Maximum number of positions ({self.trader.risk_config.max_concurrent_positions}) reached")
                return
        
            analyses = self.trader.run_analysis()

            # Place new trades for qualifying strategies
            for ticker, strategy in analyses.items():
                # First check if we already have a position in this stock
                existing_positions = account_state["positions"]
                if any(pos.symbol == ticker for pos in existing_positions):
                    self.logger.info(f"Already have position in {ticker}, skipping new trade")
                    continue
                    
                # Then check if we can place the trade
                if self.trader.can_place_new_trade(strategy, ticker):
                    self.trader.place_new_trade(strategy, ticker)

        except Exception as e:
            self.logger.error(f"Error in trader execution: {e}")
            self.logger.error(traceback.format_exc())


                    
    def execute_midday_strategy(self):
        """Mid-day position management"""
        try:
            self.execute_existing_positions()
        except Exception as e:
            self.logger.error(f"Error in midday strategy execution: {e}")
                    

    def execute_end_day_strategy(self):
        """End of day cleanup and position management"""
        try:
            self.execute_existing_positions()
        except Exception as e:
            self.logger.error(f"Error in midday strategy execution: {e}")

    def get_current_datetime(self):
        """Get the current datetime, respecting backtest mode"""
        if self.is_backtest and self.trader.backtest_date:
            return self.trader.backtest_date
        return datetime.now()


if __name__ == "__main__":
    # Configure broker
    ALPACA_CONFIG = {
        "API_KEY": os.getenv("ALPACA_API_KEY_LLM_AGENT"),
        "API_SECRET": os.getenv("ALPACA_API_SECRET_LLM_AGENT"),
        "PAPER": True if os.getenv("ALPACA_PAPER_TRADING_LLM_AGENT", "true").lower() == "true" else False,
        "URL": os.getenv("ALPACA_BASE_URL")
    }

    # Create the broker instance - use our custom broker that handles bracket orders better
    broker = CustomAlpacaBroker(ALPACA_CONFIG)
    
    backtest = get_is_backtest()
    strategy = ThreePhaseStrategy(broker=broker)

    if backtest:
        start = datetime(2023, 3, 1)  # Using 2023 instead of 2025
        end = datetime(2023, 3, 31)   # Using a full month of historical data
        strategy.run_backtest(
            PolygonDataBacktesting,
            start,
            end,
        )
    else:
        # Start trading
        trader = Trader()
        trader.add_strategy(strategy)
        # Start trading
        trader.run_all()