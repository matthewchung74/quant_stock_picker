#!/usr/bin/env python3
"""
Quantitative Stock Analysis Agent

This agent provides comprehensive analysis for stocks using market data,
sentiment analysis, macroeconomic factors, and technical indicators to
develop trading strategies and risk assessments.
"""

import time
import csv
import traceback
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import argparse
# import asyncio
import shutil

# Import the logging utilities from our new module
from logging_utils import setup_logging, get_component_logger

from screener import screen_for_swing_trades
from sentiment_analyzer import SentimentAnalysisResult, get_sentiment_analysis_with_agent
from macro_analyzer import get_macro_analysis_with_agent
from market_data_fetcher import MarketData, get_market_data_with_agent
from strategy_analyzer import get_strategy_with_agent, TradingStrategy

# Import the new CSV functions
from csv_writer import generate_recommendations_csv, generate_position_analysis_csv

# Initialize logging at the start of the application
logger = setup_logging(log_file=f"quant_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", redirect_stdout=False)

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def generate_complete_analysis(ticker: str) -> TradingStrategy:
    """
    Generate a complete analysis for a stock.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        StockAnalysis object containing all analysis results
    """
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


def run_analysis():
    """Run the stock analysis process"""
    logger = get_component_logger("Analysis")
    
    # Dictionary to store all analysis results
    all_analyses = {}
    all_market_data = {}
    all_sentiment_analysis = {}
    all_macro_analysis = {}

    # Step 1: Screen for top swing trading candidates
    logger.info("\n======== Screening for Top Swing Trading Candidates ========")
    stock_candidates, screen_results = screen_for_swing_trades(max_stocks=3)
    
    # Extract tickers for analysis
    top_tickers = [candidate["ticker"] for candidate in stock_candidates]
    
    logger.info(screen_results)
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
    
    # Step 4: Analyze positions as if purchased 2 days ago
    logger.info("\n======== Analyzing Hypothetical Positions ========")
    from position_analyzer import get_position_analysis, ExistingPosition
    from datetime import datetime, timedelta

    purchase_date = datetime.now() - timedelta(days=2)
    position_analyses = {}
    
    for ticker in top_tickers:
        try:
            market_data = all_market_data[ticker]
            hypothetical_purchase_price = market_data.latest_price * 0.98
            
            position = ExistingPosition(
                ticker=ticker,
                purchase_price=hypothetical_purchase_price,
                purchase_date=purchase_date,
                quantity=100,
                current_price=market_data.latest_price,
                unrealized_pl=(market_data.latest_price - hypothetical_purchase_price) * 100,
                pl_percentage=((market_data.latest_price - hypothetical_purchase_price) / hypothetical_purchase_price) * 100
            )
            
            position_analysis = get_position_analysis(
                position=position,
                market_data=market_data,
                sentiment_analysis=all_sentiment_analysis[ticker],
                macro_analysis=all_macro_analysis[ticker]
            )
            
            position_analyses[ticker] = position_analysis
            
            # Log the results
            logger.info(f"\nPosition Analysis for {ticker}:")
            logger.info(f"Purchase Price: ${position.purchase_price:.2f}")
            logger.info(f"Current Price: ${position.current_price:.2f}")
            logger.info(f"Days Held: {position_analysis.days_held}")
            logger.info(f"P/L: ${position.unrealized_pl:.2f} ({position.pl_percentage:.1f}%)")
            logger.info(f"Recommendation: {position_analysis.recommendation}")
            logger.info(f"Stop Loss: ${position_analysis.stop_loss:.2f}")
            logger.info(f"Target Price: ${position_analysis.target_price:.2f}")
            logger.info(f"Summary: {position_analysis.summary}")
            
        except Exception as e:
            logger.error(f"Error analyzing position for {ticker}: {e}")
            logger.error(traceback.format_exc())
            continue
    
    # Generate position analysis CSV
    logger.info("\n======== Generating Position Analysis CSV ========")
    position_csv_file = generate_position_analysis_csv(position_analyses)
    logger.info(f"Position analyses saved to {position_csv_file}")
    
    logger.info("\n======== Analysis Complete ========")
    return all_analyses

def clean_directories():
    """
    Delete logs and output directories for a fresh start each run
    """
    logger = logging.getLogger("CleanUp")
    
    # List of directories to clean
    dirs_to_clean = ['logs', 'output']
    
    for dir_name in dirs_to_clean:
        dir_path = Path(dir_name)
        if dir_path.exists():
            try:
                # Delete the directory and all contents
                shutil.rmtree(dir_path)
                logger.info(f"Successfully deleted {dir_name} directory")
            except Exception as e:
                logger.warning(f"Error deleting {dir_name} directory: {e}")
        else:
            logger.info(f"{dir_name} directory does not exist, no cleanup needed")
            
    # Recreate empty directories
    for dir_name in dirs_to_clean:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Created empty {dir_name} directory")

# Main execution
if __name__ == "__main__":
    # Clean logs and output directories first
    clean_directories()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Stock analysis and recommendation tool")
    parser.add_argument("--ticker", type=str, help="Specific ticker to analyze")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    parser.add_argument("--disable-output-redirect", action="store_true", 
                        help="Disable redirecting stdout/stderr to the logger")
    args = parser.parse_args()
    
    # Set up logging based on arguments
    log_level = getattr(logging, args.log_level)
    setup_logging(console_level=log_level, redirect_stdout=not args.disable_output_redirect)
    
    # Get a logger for the main module
    logger = get_component_logger("Main")
    logger.info("Starting stock analysis application")
    
    try:
        # If a specific ticker is provided, analyze just that ticker
        if args.ticker:
            logger.info(f"Analyzing specific ticker: {args.ticker}")
            analysis = generate_complete_analysis(args.ticker)
            analyses = {args.ticker: analysis}
            generate_recommendations_csv(analyses)
        else:
            # Otherwise run the full screening and analysis process
            analyses = run_analysis()
        
        logger.info("Analysis completed successfully")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.error(traceback.format_exc())
