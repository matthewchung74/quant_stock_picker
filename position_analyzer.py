#!/usr/bin/env python3
"""
Position Analysis Module

This module provides specialized analysis for existing stock positions using AI.
"""

import os
import time
import traceback
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from logging_utils import get_component_logger
from market_data_fetcher import MarketData
from sentiment_analyzer import SentimentAnalysisResult
from macro_analyzer import MacroAnalysisResult

# Load environment variables
load_dotenv()

class PositionAction(str, Enum):
    """Types of position actions"""
    HOLD = "HOLD"
    EXIT = "EXIT"
    ADD = "ADD"
    ERROR = "ERROR"

class ExistingPosition(BaseModel):
    """Details of an existing stock position"""
    ticker: str
    purchase_price: float
    purchase_date: datetime
    quantity: int
    current_price: float
    unrealized_pl: float
    pl_percentage: float

class PositionAnalysis(BaseModel):
    """Analysis results for an existing position"""
    ticker: str
    current_price: float
    purchase_price: float
    purchase_date: datetime
    days_held: int
    recommendation: PositionAction
    stop_loss: float
    target_price: float
    summary: str
    explanation: str
    analysis_time_seconds: float = 0

@dataclass
class PositionAnalyzerDependencies:
    """Dependencies for the Position Analysis Agent"""
    position: ExistingPosition
    market_data: MarketData
    sentiment_analysis: SentimentAnalysisResult
    macro_analysis: MacroAnalysisResult

# Initialize AI model
ai_model = OpenAIModel(
    model_name=os.environ.get("PREFERRED_AI_MODEL", "deepseek-chat"),
    base_url=os.environ.get("PREFERRED_AI_URL", "https://api.deepseek.com/v1"),
    api_key=os.environ.get("DEEPSEEK_API_KEY")
)

# Create the Position Analysis Agent
position_agent = Agent(
    model=ai_model,
    deps_type=PositionAnalyzerDependencies,
    result_type=PositionAnalysis,
    system_prompt=(
        "You are a position management specialist focusing on analyzing existing stock positions. "
        "Your task is to evaluate current positions and provide specific recommendations for managing them. "
        "Provide clear, actionable recommendations with specific price targets and rationale.\n\n"
        "Your analysis must include:\n"
        "1. Clear HOLD, EXIT, or ADD recommendation\n"
        "2. Stop loss price\n"
        "3. Target price\n"
        "4. Brief summary\n"
        "5. Detailed explanation of the recommendation"
    )
)

def get_position_analysis(
    position: ExistingPosition,
    market_data: MarketData,
    sentiment_analysis: SentimentAnalysisResult,
    macro_analysis: MacroAnalysisResult
) -> PositionAnalysis:
    """
    Analyze an existing position and provide management recommendations.
    
    Args:
        position: Details of the existing position
        market_data: Current market data
        sentiment_analysis: Current sentiment analysis
        macro_analysis: Current macro analysis
        
    Returns:
        PositionAnalysis with recommendations
    """
    logger = get_component_logger("PositionAnalysis")
    logger.info(f"Analyzing position for {position.ticker}")
    
    start_time = time.time()
    days_held = (datetime.now() - position.purchase_date).days
    
    try:
        # Set up dependencies
        deps = PositionAnalyzerDependencies(
            position=position,
            market_data=market_data,
            sentiment_analysis=sentiment_analysis,
            macro_analysis=macro_analysis
        )
        
        # Build the analysis prompt
        current_date = datetime.now()
        prompt = f"""
        Analyze the existing position in {position.ticker} stock:
        
        CURRENT DATE: {current_date.strftime('%Y-%m-%d')}
        
        POSITION DETAILS:
        - Purchase Price: ${position.purchase_price:.2f}
        - Purchase Date: {position.purchase_date.strftime('%Y-%m-%d')}
        - Days Held: {days_held}
        - Current Price: ${position.current_price:.2f}
        - Unrealized P/L: ${position.unrealized_pl:.2f} ({position.pl_percentage:.1f}%)
        - Position Size: {position.quantity:,} shares
        
        MARKET DATA:
        - Price Changes: {market_data.price_changes}
        - Technical Indicators: {market_data.latest_indicators}
        
        SENTIMENT: {sentiment_analysis.sentiment_rating} ({sentiment_analysis.confidence_level})
        {sentiment_analysis.explanation}
        
        MACRO ANALYSIS:
        {macro_analysis.summary}
        
        Provide:
        1. Clear recommendation (HOLD/EXIT/ADD)
        2. Stop loss price
        3. Target price
        4. Brief summary (1-2 sentences)
        5. Detailed explanation
        """
        
        # Get analysis from the agent
        result = position_agent.run_sync(
            prompt,
            deps=deps,
            model_settings={"temperature": 0.2}
        )
        
        # Add analysis time
        analysis = result.data
        analysis.analysis_time_seconds = time.time() - start_time
        
        logger.info(f"Position analysis completed in {analysis.analysis_time_seconds:.2f} seconds")
        logger.info(f"Recommendation: {analysis.recommendation}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing position: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    """Example usage of position analysis"""
    # Example position
    position = ExistingPosition(
        ticker="AAPL",
        purchase_price=150.00,
        purchase_date=datetime(2023, 1, 1),
        quantity=100,
        current_price=175.00,
        unrealized_pl=2500.00,
        pl_percentage=16.67
    )
    
    try:
        # Mock data for example
        market_data = MarketData(
            ticker="AAPL",
            latest_price=175.00,
            price_changes={"1d": 1.2, "1w": 2.5, "1m": 5.0},
            latest_indicators={"daily": {"rsi": 55, "sma_50": 170, "sma_200": 160}}
        )
        
        sentiment_analysis = SentimentAnalysisResult(
            ticker="AAPL",
            sentiment_rating="BULLISH",
            confidence_level="HIGH",
            explanation="Strong earnings and positive market sentiment"
        )
        
        macro_analysis = MacroAnalysisResult(
            ticker="AAPL",
            outlook="FAVORABLE",
            economic_indicators_impact="Low interest rates and stable inflation supporting tech valuations",
            geopolitical_factors_impact="Limited exposure to current geopolitical tensions",
            industry_trends_impact="Strong tech sector growth and AI adoption driving demand",
            summary="Favorable macro conditions with strong tech sector performance",
            timestamp=datetime.now().isoformat()
        )
        
        # Get analysis
        analysis = get_position_analysis(position, market_data, sentiment_analysis, macro_analysis)
        
        # Print results
        print("\n" + "=" * 50)
        print(f"POSITION ANALYSIS FOR {position.ticker}")
        print("=" * 50)
        print(f"Recommendation: {analysis.recommendation}")
        print(f"Stop Loss: ${analysis.stop_loss:.2f}")
        print(f"Target Price: ${analysis.target_price:.2f}")
        print(f"\nSummary: {analysis.summary}")
        print("\nExplanation:")
        print(analysis.explanation)
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc() 