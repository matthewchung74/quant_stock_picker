"""
CSV Writer Module

This module handles the generation of CSV files for trading recommendations
and position analyses.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict

from logging_utils import get_component_logger
from market_data_fetcher import MarketData
from strategy_analyzer import TradingStrategy
from position_analyzer import PositionAnalysis

def generate_recommendations_csv(strategies: Dict[str, TradingStrategy], market_data: Dict[str, MarketData]) -> str:
    """
    Generate a CSV file with trading recommendations.
    
    Args:
        strategies: Dictionary of trading strategies, keyed by ticker
        market_data: Dictionary of market data, keyed by ticker
        
    Returns:
        Path to the generated CSV file
    """
    logger = get_component_logger("CSVGenerator")
    logger.info("Generating recommendations CSV")
    
    # Create output directory if it doesn't exist
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f"recommendations_{timestamp}.csv"
    
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'Ticker',
                'Current Price',
                'Expiration Date',
                'Entry Price',
                'Stop Loss Price',
                'Profit Target',
                'Risk/Reward',
                'Summary',
                'Explanation'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for ticker, strategy in strategies.items():
                current_price = f"${market_data[ticker].latest_price:.2f}"
                
                row = {
                    'Ticker': ticker,
                    'Current Price': current_price,
                    'Expiration Date': strategy.expiration_date or "N/A",
                    'Entry Price': f"${strategy.entry_price:.2f}" if strategy.entry_price else "N/A",
                    'Stop Loss Price': f"${strategy.stop_loss_price:.2f}" if strategy.stop_loss_price else "N/A",
                    'Profit Target': f"${strategy.profit_target:.2f}" if strategy.profit_target else "N/A",
                    'Risk/Reward': strategy.risk_reward or "N/A",
                    'Summary': strategy.summary or "N/A",
                    'Explanation': strategy.explanation or "N/A"
                }
                writer.writerow(row)
        
        logger.info(f"Recommendations CSV generated successfully: {csv_path}")
        return str(csv_path)
        
    except Exception as e:
        logger.error(f"Error generating recommendations CSV: {e}")
        raise

def generate_position_analysis_csv(positions: Dict[str, PositionAnalysis]) -> str:
    """
    Generate a CSV file with position analyses.
    
    Args:
        positions: Dictionary of position analyses, keyed by ticker
        
    Returns:
        Path to the generated CSV file
    """
    logger = get_component_logger("CSVGenerator")
    logger.info("Generating position analysis CSV")
    
    # Create output directory if it doesn't exist
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f"position_analysis_{timestamp}.csv"
    
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'Ticker',
                'Current Price',
                'Purchase Price',
                'Purchase Date',
                'Days Held',
                'Unrealized P/L',
                'P/L Percentage',
                'Recommendation',
                'Stop Loss',
                'Target Price',
                'Summary',
                'Explanation'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for ticker, analysis in positions.items():
                row = {
                    'Ticker': ticker,
                    'Current Price': f"${analysis.current_price:.2f}",
                    'Purchase Price': f"${analysis.purchase_price:.2f}",
                    'Purchase Date': analysis.purchase_date.strftime('%Y-%m-%d'),
                    'Days Held': analysis.days_held,
                    'Unrealized P/L': f"${(analysis.current_price - analysis.purchase_price) * 100:.2f}",
                    'P/L Percentage': f"{((analysis.current_price - analysis.purchase_price) / analysis.purchase_price * 100):.1f}%",
                    'Recommendation': analysis.recommendation,
                    'Stop Loss': f"${analysis.stop_loss:.2f}",
                    'Target Price': f"${analysis.target_price:.2f}",
                    'Summary': analysis.summary,
                    'Explanation': analysis.explanation
                }
                writer.writerow(row)
        
        logger.info(f"Position analysis CSV generated successfully: {csv_path}")
        return str(csv_path)
        
    except Exception as e:
        logger.error(f"Error generating position analysis CSV: {e}")
        raise 