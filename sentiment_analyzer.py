#!/usr/bin/env python3
"""
Sentiment Analysis Module

This module provides functions for analyzing stock market sentiment using AI.
"""

import os
import time
from typing import Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import traceback

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.common_tools.tavily import tavily_search_tool

# Import the logging utilities from the main application
from logging_utils import get_component_logger

# Load environment variables from .env file
load_dotenv()

# Define dependencies for the Sentiment Analysis Agent
@dataclass
class SentimentAnalysisDependencies:
    """Dependencies for the Sentiment Analysis Agent."""
    ticker: str
    include_news: bool = True
    include_social_media: bool = True
    include_analyst_ratings: bool = True

class SentimentRating(str, Enum):
    """Enumeration of possible sentiment ratings."""
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"

class ConfidenceLevel(str, Enum):
    """Enumeration of possible confidence levels."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class SentimentAnalysisResult(BaseModel):
    """Model for sentiment analysis results."""
    ticker: str
    sentiment_rating: SentimentRating
    confidence_level: ConfidenceLevel
    explanation: str
    news_summary: Optional[str] = None
    social_media_summary: Optional[str] = None
    analyst_ratings_summary: Optional[str] = None
    short_interest_summary: Optional[str] = None  # Summary of short interest data
    institutional_holdings_changes: Optional[str] = None  # Recent institutional position changes
    options_sentiment: Optional[str] = None  # Put/Call ratio and options flow analysis
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# Initialize the DeepSeek R1 model
ai_model = OpenAIModel(
    model_name=os.environ["PREFERRED_AI_MODEL"],  # DeepSeek chat model
    base_url=os.environ["PREFERRED_AI_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
)

# Create the Sentiment Analysis Agent
sentiment_agent = Agent(
    model=ai_model,
    deps_type=SentimentAnalysisDependencies,
    result_type=SentimentAnalysisResult,
    tools=[duckduckgo_search_tool(), tavily_search_tool(os.getenv('TAVILY_API_KEY'))],
    system_prompt=(
        "You are a sentiment analysis specialist focusing on stock market sentiment. "
        "You will use Tavily to search the web for the most relevant and recent news articles about the stock. "
        "Your task is to analyze news, social media, analyst opinions, short interest, "
        "institutional holdings, and options flow to gauge the overall market sentiment "
        "for specific stocks. Pay special attention to bearish indicators that might "
        "suggest short opportunities, including:\n"
        "- High or increasing short interest\n"
        "- Institutional selling pressure\n"
        "- Bearish options flow (high put/call ratio)\n"
        "- Negative news catalysts\n"
        "- Technical breakdown patterns\n\n"
        "You must return a structured result with the following fields:\n"
        "- ticker: The stock symbol being analyzed\n"
        "- sentiment_rating: One of 'BULLISH', 'NEUTRAL', or 'BEARISH'\n"
        "- confidence_level: One of 'HIGH', 'MEDIUM', or 'LOW'\n"
        "- explanation: A comprehensive explanation of your rating and confidence\n"
        "- news_summary: Optional summary of news analysis\n"
        "- social_media_summary: Optional summary of social media sentiment\n"
        "- analyst_ratings_summary: Optional summary of analyst ratings\n"
        "- short_interest_summary: Optional summary of short interest data\n"
        "- institutional_holdings_changes: Optional summary of institutional position changes\n"
        "- options_sentiment: Optional summary of options market sentiment\n"
        "\nEnsure your sentiment_rating and confidence_level exactly match one of the allowed values."
    )
)

# Main function to get sentiment analysis with the specialized agent
def get_sentiment_analysis_with_agent(ticker: str) -> SentimentAnalysisResult:
    """
    Get comprehensive sentiment analysis for a stock using the sentiment agent.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        SentimentAnalysisResult object containing sentiment rating, confidence, and explanations
    """
    logger = get_component_logger("SentimentAnalysis")
    logger.info(f"Getting sentiment analysis for {ticker} using specialized agent")
    
    try:
        # Set up dependencies
        deps = SentimentAnalysisDependencies(ticker=ticker)
        
        # Run the agent with a prompt to analyze sentiment - use the sync method
        result = sentiment_agent.run_sync(
            f"Provide a comprehensive sentiment analysis for {ticker} stock, covering news, social media, and analyst opinions.",
            deps=deps,
            model_settings={"temperature": 0.1}
        )
        
        # Log the sentiment rating
        logger.info(f"Sentiment for {ticker}: {result.data.sentiment_rating} (Confidence: {result.data.confidence_level})")
        
        return result.data
    except Exception as e:
        logger.error(f"Error in get_sentiment_analysis_with_agent: {e}")
        logger.error(traceback.format_exc())
        
        # Re-raise the exception to be handled by the caller
        raise

if __name__ == "__main__":
    """Main function to demonstrate the sentiment analysis functionality"""
    ticker = "TSLA"        
    print(f"\nAnalyzing sentiment for {ticker}...\n")
    
    try:        
        # Get the sentiment analysis
        start_time = time.time()
        result = get_sentiment_analysis_with_agent(ticker)
        elapsed_time = time.time() - start_time
        
        # Print the results
        print("\n" + "=" * 50)
        print(f"SENTIMENT ANALYSIS FOR {ticker}")
        print("=" * 50)
        print(f"Sentiment Rating: {result.sentiment_rating}")
        print(f"Confidence Level: {result.confidence_level}")
        print(f"Timestamp: {result.timestamp}")
        print("\nEXPLANATION:")
        print(result.explanation)
        
        if result.news_summary:
            print("\nNEWS SUMMARY:")
            print(result.news_summary)
            
        if result.social_media_summary:
            print("\nSOCIAL MEDIA SUMMARY:")
            print(result.social_media_summary)
            
        if result.analyst_ratings_summary:
            print("\nANALYST RATINGS SUMMARY:")
            print(result.analyst_ratings_summary)
            
        if result.short_interest_summary:
            print("\nSHORT INTEREST SUMMARY:")
            print(result.short_interest_summary)
            
        if result.institutional_holdings_changes:
            print("\nINSTITUTIONAL HOLDINGS CHANGES:")
            print(result.institutional_holdings_changes)
            
        if result.options_sentiment:
            print("\nOPTIONS SENTIMENT:")
            print(result.options_sentiment)
            
        print("\n" + "-" * 50)
        print(f"Analysis completed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc() 

