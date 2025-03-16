#!/usr/bin/env python3
"""
Position Analysis Module

This module provides functions for analyzing existing stock positions and generating
recommendations based on comprehensive analysis.
"""

import asyncio
import json
import re
import time
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field

from logging_utils import get_component_logger
from main import RecommendationType, SellRecommendation, StrategyAgentDependencies, strategy_agent
from strategy_analyzer import get_strategy_with_agent
from sentiment_analyzer import get_sentiment_analysis_with_agent
from macro_analyzer import get_macro_analysis_with_agent
from market_data_fetcher import get_market_data_with_agent

def analyze_existing_position(ticker: str, purchase_price: float, purchase_date: str) -> SellRecommendation:
    """
    Analyze an existing stock position and provide a recommendation.
    
    Args:
        ticker: Stock ticker symbol
        purchase_price: Original purchase price
        purchase_date: Date of purchase (YYYY-MM-DD)
        
    Returns:
        SellRecommendation object with analysis results
    """
    logger = get_component_logger("PositionAnalysis")
    logger.info(f"Analyzing existing position for {ticker}, purchased at ${purchase_price} on {purchase_date}")
    
    start_time = time.time()
    
    try:
        # Get market data
        market_data = get_market_data_with_agent(ticker)
        logger.info(f"Current price for {ticker}: ${market_data.latest_price:.2f}" if market_data.latest_price else "Price data not available")
        
        # Get sentiment analysis
        sentiment_analysis = get_sentiment_analysis_with_agent(ticker)
        logger.info(f"Completed sentiment analysis for {ticker}")
        
        # Get macro analysis
        macro_analysis = get_macro_analysis_with_agent(ticker)
        logger.info(f"Completed macro analysis for {ticker}")
        
        # Get trading strategy
        strategy = get_strategy_with_agent(ticker, market_data.latest_price, market_data, sentiment_analysis, macro_analysis)
        logger.info(f"Completed trading strategy for {ticker}")
        
        # Create analysis dictionary to match the structure expected by the rest of the function
        analysis = {
            'market_data': market_data,
            'sentiment_analysis': sentiment_analysis,
            'macro_analysis': macro_analysis,
            'strategy': strategy
        }
        
        # Extract current price from the analysis
        current_price = 0
        if market_data and market_data.latest_price:
            current_price = market_data.latest_price
        
        if current_price <= 0:
            logger.warning(f"Could not get current price for {ticker}, using placeholder value")
            current_price = purchase_price  # Use purchase price as fallback
        
        # Calculate days held
        purchase_datetime = datetime.fromisoformat(purchase_date)
        days_held = (datetime.now() - purchase_datetime).days
        
        # Calculate profit/loss
        p_l_dollar = current_price - purchase_price
        p_l_percent = (p_l_dollar / purchase_price) * 100
        
        # Default to ANALYSIS_ONLY recommendation
        action = RecommendationType.ANALYSIS_ONLY
        limit_price = None
        stop_loss_price = None
        target_price = None
        entry_price = None
        rationale = f"Analysis only recommendation for {ticker}."
        
        # If we have a strategy, extract any price targets
        if strategy:
            # Extract structured parameters from the strategy
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Create dependency context for the strategy agent
                deps = StrategyAgentDependencies(
                    ticker=ticker,
                    current_price=current_price
                )
                
                # Create a prompt to extract parameters from the strategy text
                prompt = f"""
                Below is a trading strategy for {ticker}. Please extract the entry prices, stop-loss prices, profit targets, and risk/reward ratios for each timeframe (Day, Swing, and Long-Term).
                
                For each timeframe, I need:
                1. Entry price
                2. Stop-loss price
                3. Profit target price
                4. Risk/reward ratio
                
                If any information is not available, indicate it as missing.
                
                Return the information as JSON with this structure:
                {{
                    "ticker": "{ticker}",
                    "strategies": [
                        {{
                            "timeframe": "DAY",
                            "entry_price": 123.45,
                            "stop_loss_price": 120.00,
                            "profit_target": 130.00,
                            "risk_reward": "1:2"
                        }},
                        ... (other timeframes)
                    ]
                }}
                
                STRATEGY TEXT:
                {strategy}
                """
                
                # Call the strategy agent directly
                result = loop.run_until_complete(
                    strategy_agent.run(prompt, deps=deps)
                )
                loop.close()
                
                # Parse the result
                try:
                    # Try to extract structured JSON from the result
                    json_match = re.search(r'\{[\s\S]*\}', result)
                    if json_match:
                        json_str = json_match.group(0)
                        params_dict = json.loads(json_str)
                        
                        # Extract price targets from the first strategy (day trading)
                        if "strategies" in params_dict and params_dict["strategies"]:
                            for strat in params_dict["strategies"]:
                                if strat.get("timeframe") == "DAY":
                                    entry_price = strat.get("entry_price")
                                    stop_loss_price = strat.get("stop_loss_price")
                                    target_price = strat.get("profit_target")
                                    break
                            
                            # If we found price targets, update the recommendation
                            if target_price and current_price < target_price:
                                action = RecommendationType.HOLD_WITH_STOP_LOSS
                                rationale = f"Hold {ticker} with a stop loss at ${stop_loss_price:.2f} and target price of ${target_price:.2f}."
                            elif stop_loss_price and current_price < stop_loss_price:
                                action = RecommendationType.SELL_NOW
                                rationale = f"Sell {ticker} now as price is below stop loss of ${stop_loss_price:.2f}."
                            
                            logger.info(f"Successfully extracted structured strategy parameters for {ticker}")
                except Exception as e:
                    logger.error(f"Error extracting strategy parameters: {e}")
                    # Continue without parameters
            except Exception as e:
                logger.error(f"Error extracting parameters: {e}")
                # Continue without parameters
        
        # Create the recommendation
        recommendation = SellRecommendation(
            ticker=ticker,
            current_price=current_price,
            purchase_price=purchase_price,
            purchase_date=purchase_date,
            days_held=days_held,
            p_l_percent=p_l_percent,
            p_l_dollar=p_l_dollar,
            action=action,
            limit_price=limit_price,
            stop_loss_price=stop_loss_price,
            target_price=target_price,
            entry_price=entry_price,
            rationale=rationale,
            recommendation_time_seconds=time.time() - start_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Generated recommendation for {ticker}: {action.value}")
        return recommendation
        
    except Exception as e:
        logger.error(f"Error analyzing position for {ticker}: {e}")
        logger.error(traceback.format_exc())
        
        # Return error recommendation
        return SellRecommendation(
            ticker=ticker,
            current_price=0,
            purchase_price=purchase_price,
            purchase_date=purchase_date,
            action=RecommendationType.ERROR,
            rationale=f"Error generating recommendation: {str(e)}",
            recommendation_time_seconds=time.time() - start_time
        ) 