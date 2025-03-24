"""
Analysis Generator Module

This module handles the generation of complete analysis for trading strategies.
It coordinates market data, sentiment analysis, macro analysis, and strategy generation.
"""

import logging
import os
import time
import traceback
from datetime import datetime
from typing import Tuple, Any

from logging_utils import get_component_logger
from strategy_analyzer import TradingStrategy, get_strategy_with_agent
from market_data_fetcher import MarketData, get_market_data_with_agent
from sentiment_analyzer import SentimentAnalysisResult, get_sentiment_analysis_with_agent
from macro_analyzer import get_macro_analysis_with_agent

def generate_llm_strategy_analysis(ticker: str) -> Tuple[TradingStrategy, MarketData, SentimentAnalysisResult, Any]:
    """Generate a complete analysis for a stock using LLM-based strategy"""
    logger = get_component_logger("StrategyAnalysis")
    logger.info(f"Generating LLM strategy analysis for {ticker}")
    
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
        logger.info(f"LLM strategy analysis for {ticker} completed in {elapsed_time:.2f} seconds")

        return strategy, market_data, sentiment_analysis, macro_analysis
        
    except Exception as e:
        logger.error(f"Error generating LLM strategy analysis for {ticker}: {e}")
        logger.error(traceback.format_exc())
        raise 