"""
Trading Journal Module

Tracks trading activity including entries, exits, and performance metrics.
Logs trade data to structured CSV files for later analysis.
"""

import os
import csv
import json
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from logging_utils import get_component_logger
from strategy_analyzer import TradingStrategy
from position_analyzer import PositionType

class TradeStatus(str, Enum):
    """Status of a trade"""
    PENDING = "PENDING"       # Order placed but not filled
    ACTIVE = "ACTIVE"         # Position is open
    CLOSED_PROFIT = "CLOSED_PROFIT"  # Closed with profit
    CLOSED_LOSS = "CLOSED_LOSS"      # Closed with loss
    CANCELLED = "CANCELLED"   # Order was cancelled before fill
    EXPIRED = "EXPIRED"       # Order expired without filling

@dataclass
class TradeRecord:
    """Complete record of a trade from entry to exit"""
    # Trade identification
    trade_id: str
    ticker: str
    position_type: PositionType
    status: TradeStatus
    
    # Entry details
    entry_date: Optional[datetime] = None
    entry_price: Optional[float] = None
    entry_order_id: Optional[str] = None
    shares: Optional[int] = None
    
    # Exit details
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_order_id: Optional[str] = None
    
    # Performance
    profit_loss_dollars: Optional[float] = None
    profit_loss_percent: Optional[float] = None
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_planned: Optional[str] = None
    
    # Strategy details
    strategy_summary: Optional[str] = None
    entry_reason: Optional[str] = None
    exit_reason: Optional[str] = None
    
    # Additional notes and data
    notes: Optional[str] = None
    extra_data: Optional[Dict[str, Any]] = None
    
    # Add new fields for stop-loss details
    stop_loss_limit_price: Optional[float] = None  # New field for stop-limit orders
    stop_loss_trigger_price: Optional[float] = None  # Rename existing stop_loss to be more explicit
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary format for serialization"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, dict):
                result[key] = json.dumps(value)
            else:
                result[key] = value
        return result

class TradingJournal:
    """
    Trading Journal to track all trades and calculate performance metrics
    """
    
    def __init__(self, journal_dir="./journal", backtest_mode=False, backtest_date=None):
        """
        Initialize the trading journal
        
        Args:
            journal_dir: Directory to store journal files
            backtest_mode: Whether operating in backtest mode
            backtest_date: The date to use for backtesting (datetime object)
        """
        self.logger = get_component_logger("TradingJournal")
        self.journal_dir = journal_dir
        self.backtest_mode = backtest_mode
        self.backtest_date = backtest_date
        
        # Set up the file name with appropriate date
        date_str = self._get_current_datetime().strftime('%Y%m%d')
        mode_prefix = "backtest_" if backtest_mode else ""
        self.trades_file = os.path.join(journal_dir, f"{mode_prefix}trades_{date_str}.csv")
        self.trades: Dict[str, TradeRecord] = {}
        
        # Ensure journal directory exists
        os.makedirs(journal_dir, exist_ok=True)
        
        # Create the CSV file with headers if it doesn't exist
        if not os.path.exists(self.trades_file):
            self._initialize_csv()
        
        mode_msg = "BACKTEST MODE" if backtest_mode else "LIVE MODE"
        self.logger.info(f"Trading journal initialized in {mode_msg}. Recording to {self.trades_file}")
    
    def _get_current_datetime(self) -> datetime:
        """
        Get the current datetime, accounting for backtest mode
        
        Returns:
            datetime object to use (either current or backtest date)
        """
        if self.backtest_mode and self.backtest_date:
            return self.backtest_date
        return datetime.now()
    
    def _initialize_csv(self):
        """Create the CSV file with headers"""
        # Get field names from TradeRecord
        sample_record = TradeRecord(
            trade_id="sample",
            ticker="SAMPLE",
            position_type=PositionType.LONG,
            status=TradeStatus.PENDING
        )
        fieldnames = list(sample_record.to_dict().keys())
        
        with open(self.trades_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    def record_new_trade(self, ticker: str, strategy: TradingStrategy, 
                         order_id: str, shares: int) -> str:
        """
        Record a new trade when an order is placed
        
        Args:
            ticker: The stock symbol
            strategy: The trading strategy used
            order_id: The broker's order ID
            shares: Number of shares in the order
            
        Returns:
            trade_id: Unique identifier for this trade
        """
        # Get current time (real or simulated)
        current_time = self._get_current_datetime()
        
        # Generate a unique trade ID
        trade_id = f"{ticker}_{current_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Create a new trade record with additional stop loss details
        trade = TradeRecord(
            trade_id=trade_id,
            ticker=ticker,
            position_type=strategy.position_type,
            status=TradeStatus.PENDING,
            entry_order_id=order_id,
            shares=shares,
            entry_date=current_time,
            entry_price=strategy.entry_price,
            stop_loss_trigger_price=strategy.stop_loss_price,  # Original stop price
            stop_loss_limit_price=strategy.stop_loss_price * (0.99 if strategy.position_type == PositionType.LONG else 1.01),  # New limit price
            take_profit=strategy.profit_target,
            risk_reward_planned=strategy.risk_reward,
            strategy_summary=strategy.summary,
            entry_reason=strategy.explanation
        )
        
        # Store in memory
        self.trades[trade_id] = trade
        
        # Write to CSV
        self._append_to_csv(trade)
        
        mode_prefix = "[BACKTEST] " if self.backtest_mode else ""
        self.logger.info(f"{mode_prefix}Recorded new {strategy.position_type} trade for {ticker}: {trade_id}")
        return trade_id
    
    def update_trade_filled(self, trade_id: str, filled_price: float) -> None:
        """
        Update a trade when the order is filled
        
        Args:
            trade_id: The unique trade identifier
            filled_price: The actual fill price
        """
        if trade_id not in self.trades:
            self.logger.error(f"Trade {trade_id} not found in journal")
            return
        
        trade = self.trades[trade_id]
        trade.status = TradeStatus.ACTIVE
        trade.entry_price = filled_price
        
        # Update CSV
        self._update_csv()
        
        mode_prefix = "[BACKTEST] " if self.backtest_mode else ""
        self.logger.info(f"{mode_prefix}Updated trade {trade_id} as filled at ${filled_price:.2f}")
    
    def close_trade(self, trade_id: str, exit_price: float, exit_order_id: str, 
                   exit_reason: str = "Target or stop hit") -> None:
        """
        Record the closing of a trade
        
        Args:
            trade_id: The unique trade identifier
            exit_price: The exit price
            exit_order_id: The order ID of the exit order
            exit_reason: The reason for exiting the trade
        """
        if trade_id not in self.trades:
            self.logger.error(f"Trade {trade_id} not found in journal")
            return
        
        trade = self.trades[trade_id]
        trade.exit_date = self._get_current_datetime()
        trade.exit_price = exit_price
        trade.exit_order_id = exit_order_id
        trade.exit_reason = exit_reason
        
        # Calculate P&L
        if trade.entry_price and trade.shares:
            if trade.position_type == PositionType.LONG:
                trade.profit_loss_dollars = (exit_price - trade.entry_price) * trade.shares
                trade.profit_loss_percent = (exit_price / trade.entry_price - 1) * 100
            else:  # SHORT
                trade.profit_loss_dollars = (trade.entry_price - exit_price) * trade.shares
                trade.profit_loss_percent = (trade.entry_price / exit_price - 1) * 100
                
            # Set status based on profit/loss
            if trade.profit_loss_dollars > 0:
                trade.status = TradeStatus.CLOSED_PROFIT
            else:
                trade.status = TradeStatus.CLOSED_LOSS
        
        # Update CSV
        self._update_csv()
        
        mode_prefix = "[BACKTEST] " if self.backtest_mode else ""
        self.logger.info(f"{mode_prefix}Closed trade {trade_id} at ${exit_price:.2f} with P&L: ${trade.profit_loss_dollars:.2f} ({trade.profit_loss_percent:.2f}%)")
    
    def cancel_trade(self, trade_id: str, reason: str = "Order cancelled") -> None:
        """
        Record a cancelled trade
        
        Args:
            trade_id: The unique trade identifier
            reason: The reason for cancellation
        """
        if trade_id not in self.trades:
            self.logger.error(f"Trade {trade_id} not found in journal")
            return
        
        trade = self.trades[trade_id]
        trade.status = TradeStatus.CANCELLED
        trade.exit_date = self._get_current_datetime()
        trade.exit_reason = reason
        
        # Update CSV
        self._update_csv()
        
        mode_prefix = "[BACKTEST] " if self.backtest_mode else ""
        self.logger.info(f"{mode_prefix}Cancelled trade {trade_id}: {reason}")
    
    def add_note(self, trade_id: str, note: str) -> None:
        """
        Add a note to a trade
        
        Args:
            trade_id: The unique trade identifier
            note: The note to add
        """
        if trade_id not in self.trades:
            self.logger.error(f"Trade {trade_id} not found in journal")
            return
        
        trade = self.trades[trade_id]
        current_time = self._get_current_datetime()
        
        # Append to existing notes or create new
        if trade.notes:
            trade.notes += f"\n[{current_time.strftime('%Y-%m-%d %H:%M')}] {note}"
        else:
            trade.notes = f"[{current_time.strftime('%Y-%m-%d %H:%M')}] {note}"
        
        # Update CSV
        self._update_csv()
        
        mode_prefix = "[BACKTEST] " if self.backtest_mode else ""
        self.logger.info(f"{mode_prefix}Added note to trade {trade_id}")
    
    def get_trade(self, trade_id: str) -> Optional[TradeRecord]:
        """
        Get a specific trade record
        
        Args:
            trade_id: The unique trade identifier
            
        Returns:
            The trade record or None if not found
        """
        return self.trades.get(trade_id)
    
    def get_open_trades(self) -> List[TradeRecord]:
        """
        Get all currently open trades
        
        Returns:
            List of open trade records
        """
        return [t for t in self.trades.values() 
                if t.status in (TradeStatus.PENDING, TradeStatus.ACTIVE)]
    
    def get_closed_trades(self) -> List[TradeRecord]:
        """
        Get all closed trades
        
        Returns:
            List of closed trade records
        """
        return [t for t in self.trades.values() 
                if t.status in (TradeStatus.CLOSED_PROFIT, TradeStatus.CLOSED_LOSS, 
                               TradeStatus.CANCELLED, TradeStatus.EXPIRED)]
    
    def calculate_performance(self) -> Dict[str, Any]:
        """
        Calculate performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        closed_trades = self.get_closed_trades()
        
        if not closed_trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_loss": 0,
                "avg_profit_per_trade": 0
            }
        
        # Filter to actual filled and closed trades
        actual_trades = [t for t in closed_trades 
                         if t.status in (TradeStatus.CLOSED_PROFIT, TradeStatus.CLOSED_LOSS)]
        
        if not actual_trades:
            return {
                "total_trades": len(closed_trades),
                "win_rate": 0,
                "profit_loss": 0,
                "avg_profit_per_trade": 0
            }
        
        # Calculate metrics
        winning_trades = [t for t in actual_trades if t.profit_loss_dollars and t.profit_loss_dollars > 0]
        
        total_profit_loss = sum(t.profit_loss_dollars for t in actual_trades if t.profit_loss_dollars)
        
        return {
            "total_trades": len(actual_trades),
            "winning_trades": len(winning_trades),
            "win_rate": len(winning_trades) / len(actual_trades) * 100,
            "profit_loss": total_profit_loss,
            "avg_profit_per_trade": total_profit_loss / len(actual_trades) if actual_trades else 0,
        }
    
    def _append_to_csv(self, trade: TradeRecord) -> None:
        """
        Append a trade record to the CSV file
        
        Args:
            trade: The trade record to append
        """
        try:
            with open(self.trades_file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(trade.to_dict().keys()))
                writer.writerow(trade.to_dict())
        except Exception as e:
            self.logger.error(f"Error writing to trade journal CSV: {e}")
    
    def _update_csv(self) -> None:
        """Update the entire CSV file with current trade records"""
        try:
            # Get field names from the first trade or use a default set
            fieldnames = list(next(iter(self.trades.values())).to_dict().keys()) if self.trades else []
            
            if not fieldnames:
                return  # No trades to write
            
            with open(self.trades_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for trade in self.trades.values():
                    writer.writerow(trade.to_dict())
        except Exception as e:
            self.logger.error(f"Error updating trade journal CSV: {e}")
