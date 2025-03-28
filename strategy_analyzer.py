#!/usr/bin/env python3
"""
Strategy Analysis Module

This module provides functions for developing trading strategies for stocks using AI.
"""

import os
import time
import traceback
from typing import Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

# Import the logging utilities from the main application
from logging_utils import get_component_logger
from macro_analyzer import MacroAnalysisResult, MacroOutlook
from market_data_fetcher import MarketData
from sentiment_analyzer import SentimentAnalysisResult

# Load environment variables from .env file
load_dotenv()

class PositionType(str, Enum):
    """Types of positions"""
    LONG = "LONG"
    SHORT = "SHORT"

class TradingStrategy(BaseModel):
    """Model for strategy parameters for a specific timeframe."""
    position_type: PositionType = Field(..., description="Type of position (LONG/SHORT)")
    expiration_date: str = Field(None, description="Expiration date for this timeframe")
    entry_price: float = Field(None, description="Entry price for this timeframe")
    stop_loss_price: float = Field(None, description="Stop loss price for this timeframe")
    profit_target: float = Field(None, description="Profit target for this timeframe")
    risk_reward: str = Field(None, description="Risk/reward ratio (e.g. 1:2)")
    summary: str = Field(None, description="Summary of the strategy")
    explanation: str = Field(None, description="Explanation of the strategy")
    borrow_cost: Optional[float] = Field(None, description="Borrow cost for short positions")
    shares_available: Optional[int] = Field(None, description="Number of shares available to short")
    short_squeeze_risk: Optional[str] = Field(None, description="Assessment of short squeeze risk")

# Define dependencies for the Strategy Agent
@dataclass
class StrategyAgentDependencies:
    """Dependencies for the Strategy Agent."""
    ticker: str
    market_data: Optional[MarketData] = None
    sentiment_analysis: Optional[SentimentAnalysisResult] = None
    macro_analysis: Optional[MacroAnalysisResult] = None

ai_model = OpenAIModel(
    model_name=os.environ.get("PREFERRED_AI_MODEL", "deepseek-chat"),
    base_url=os.environ.get("PREFERRED_AI_URL", "https://api.deepseek.com/v1"),
    api_key=os.environ.get("DEEPSEEK_API_KEY")
)

# Create the Strategy Agent with the TradingStrategy result type
strategy_agent = Agent(
    model=ai_model,
    deps_type=StrategyAgentDependencies,
    result_type=TradingStrategy,
    system_prompt=(
        "As a trading strategy expert, your role is to develop actionable trading plans "
        "through comprehensive analysis. Your objective is to formulate detailed trading strategies "
        "for both long and short positions over various timeframes.\n\n"
        "For long positions, focus on upside potential with controlled downside risk.\n"
        "For short positions, consider:\n"
        "- Borrowing costs and availability\n"
        "- Short squeeze risk factors\n"
        "- Technical breakdown patterns\n"
        "- Fundamental deterioration\n\n"
        "For each strategy, specify:\n"
        "- Position type (LONG/SHORT)\n"
        "- Entry points\n"
        "- Exit targets\n"
        "- Stop-loss levels\n"
        "- Position sizing recommendations\n"
        "- For shorts: borrow costs, availability, and squeeze risks\n\n"
        "Incorporate both technical and fundamental analysis in your rationale.\n"
    )
)

def get_strategy_with_agent(
    ticker: str,
    market_data: MarketData,
    sentiment_analysis: SentimentAnalysisResult,
    macro_analysis: MacroAnalysisResult
) -> TradingStrategy:
    """
    Get a comprehensive trading strategy for a stock using a specialized agent.
    
    Args:
        ticker: Stock ticker symbol
        current_price: Current price of the stock
        market_data: Market data for the stock
        sentiment_analysis: Sentiment analysis result
        macro_analysis: Macro analysis result
        
    Returns:
        TradingStrategy object containing the structured trading strategy
    """
    logger = get_component_logger("StrategyAnalysis")
    logger.info(f"Getting trading strategy for {ticker} at ${market_data.latest_price}")
    
    start_time = time.time()
    
    try:
        # Set up dependencies
        deps = StrategyAgentDependencies(
            ticker=ticker,
            market_data=market_data,
            sentiment_analysis=sentiment_analysis,
            macro_analysis=macro_analysis
        )
        
        # Format sentiment analysis for strategy generation
        sentiment_text = (
            f"Overall Sentiment: {sentiment_analysis.sentiment_rating}, "
            f"Confidence: {sentiment_analysis.confidence_level}\n\n"
            f"{sentiment_analysis.explanation}"
        )
        if sentiment_analysis.news_summary:
            sentiment_text += f"\n\nNews Summary: {sentiment_analysis.news_summary}"
        if sentiment_analysis.social_media_summary:
            sentiment_text += f"\n\nSocial Media Summary: {sentiment_analysis.social_media_summary}"
        if sentiment_analysis.analyst_ratings_summary:
            sentiment_text += f"\n\nAnalyst Ratings Summary: {sentiment_analysis.analyst_ratings_summary}"
        
        # Format macro analysis for strategy generation
        macro_text = f"MACROECONOMIC OUTLOOK: {macro_analysis.outlook}\n\n"
        macro_text += f"ECONOMIC INDICATORS IMPACT:\n{macro_analysis.economic_indicators_impact}\n\n"
        macro_text += f"GEOPOLITICAL FACTORS IMPACT:\n{macro_analysis.geopolitical_factors_impact}\n\n"
        macro_text += f"INDUSTRY TRENDS IMPACT:\n{macro_analysis.industry_trends_impact}\n\n"
        
        # Prompt for strategy development
        prompt = f"""
        Develop a comprehensive trading strategy for {ticker} stock, currently trading at ${market_data.latest_price:.2f}.
        
        MARKET DATA:
        - Price Changes: {market_data.price_changes}
        - Technical Indicators: {market_data.latest_indicators}
        
        SENTIMENT ANALYSIS:
        {sentiment_text}
        
        MACRO ANALYSIS:
        {macro_text}
        
        IMPORTANT: Your strategy MUST include precise numerical values for:
        1. Entry price
        2. Stop loss price (must be lower than entry for LONG, higher for SHORT)
        3. Profit target (must be higher than entry for LONG, lower for SHORT)
        4. Risk/reward ratio
        5. Expiration date for the order (format: YYYY-MM-DD HH:MM:SS)
        
        Also provide:
        6. Brief summary (1-2 sentences)
        7. Detailed explanation of the strategy
        """
        
        # Run the agent to get the strategy
        result = strategy_agent.run_sync(
            prompt,
            deps=deps,
            model_settings={"temperature": 0.1}
        )
        
        strategy = result.data
        
        # Ensure that stop_loss_price, profit_target, and expiration_date are defined
        if not strategy.stop_loss_price or not strategy.profit_target or not strategy.expiration_date:
            raise ValueError(f"Strategy for {ticker} is missing required fields: stop_loss_price, profit_target, or expiration_date.")
        
        # Log completion
        elapsed_time = time.time() - start_time
        logger.info(f"Strategy generation for {ticker} completed in {elapsed_time:.2f} seconds")
        
        return strategy
    
    except Exception as e:
        logger.error(f"Error getting trading strategy: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    """Main function to demonstrate the strategy analysis functionality"""
    ticker = "AAPL"  # Default ticker    
    
    print(f"\nDeveloping trading strategy for {ticker}\n")
    
    try:        
        # Mock data for additional parameters
        market_data = MarketData(
            latest_price=175.0,
            ticker=ticker,
            price_changes={"1d": 0.5, "1w": 2.0, "1m": 5.0},
            latest_indicators={
                "daily": {
                    "rsi": 55.0, 
                    "sma_50": 170.0, 
                    "sma_200": 160.0,
                    "vwap": 172.5
                }
            },
            support_resistance={
                "support": [165.0, 160.0],
                "resistance": [180.0, 185.0]
            }
        )
        
        sentiment_analysis = SentimentAnalysisResult(
            ticker=ticker,
            sentiment_rating="BULLISH",
            confidence_level="HIGH",
            explanation="Strong earnings report and positive market sentiment",
            timestamp=datetime.now().isoformat()
        )
        
        macro_analysis = MacroAnalysisResult(
            ticker=ticker,
            outlook=MacroOutlook.FAVORABLE,
            economic_indicators_impact="Positive impact from low interest rates",
            geopolitical_factors_impact="Neutral",
            industry_trends_impact="Positive trend in tech sector",
            key_risks=["Potential regulatory changes", "Market volatility"],
            opportunities=["Expansion in new markets", "Innovative product launches"],
            summary="Overall favorable macroeconomic conditions for the stock",
            timestamp=datetime.now().isoformat()
        )
        
        start_time = time.time()
        strategy = get_strategy_with_agent(ticker, market_data, sentiment_analysis, macro_analysis)
        elapsed_time = time.time() - start_time
        
        # Print the results
        print("\n" + "=" * 50)
        print(f"TRADING STRATEGY FOR {ticker}")
        print("=" * 50)
        print(f"Position Type: {strategy.position_type}")
        print(f"Expiration Date: {strategy.expiration_date}")
        print(f"Entry Price: {strategy.entry_price}")
        print(f"Stop Loss Price: {strategy.stop_loss_price}")
        print(f"Profit Target: {strategy.profit_target}")
        print(f"Risk/Reward Ratio: {strategy.risk_reward}")
        print(f"Summary: {strategy.summary}")
        print("\nExplanation:")
        print(strategy.explanation)
        
        print("\n" + "-" * 50)
        print(f"Strategy developed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc() 