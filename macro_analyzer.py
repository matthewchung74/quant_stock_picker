#!/usr/bin/env python3
"""
Macro Analysis Module

This module provides functions for analyzing macroeconomic factors affecting stocks using AI.
"""

import os
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import traceback
from enum import Enum

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.common_tools.tavily import tavily_search_tool

# Import the logging utilities from the main application
from logging_utils import get_component_logger

# Load environment variables from .env file
load_dotenv()

# Define dependencies for the Macro Analysis Agent
@dataclass
class MacroAnalysisDependencies:
    """Dependencies for the Macro Analysis Agent."""
    ticker: str
    industry: Optional[str] = None
    sector: Optional[str] = None

# Define Pydantic models for structured macro analysis
class MacroOutlook(str, Enum):
    """Enumeration of possible macroeconomic outlook ratings."""
    FAVORABLE = "FAVORABLE"
    NEUTRAL = "NEUTRAL"
    UNFAVORABLE = "UNFAVORABLE"

class MacroAnalysisResult(BaseModel):
    """Model for structured macroeconomic analysis results."""
    ticker: str
    outlook: MacroOutlook
    economic_indicators_impact: str
    geopolitical_factors_impact: str
    industry_trends_impact: str
    key_risks: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)
    short_specific_factors: Dict[str, str] = Field(
        default_factory=dict,
        description="Factors specifically relevant for short positions"
    )
    sector_rotation_impact: str = Field(
        "",
        description="Analysis of sector rotation trends and their impact"
    )
    liquidity_analysis: str = Field(
        "",
        description="Analysis of market liquidity conditions"
    )
    summary: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# Initialize the AI model based on environment configuration
ai_model = OpenAIModel(
    model_name=os.environ["PREFERRED_AI_MODEL"],  # DeepSeek chat model
    base_url=os.environ["PREFERRED_AI_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
)

# Create the Macro Analysis Agent
macro_agent = Agent(
    model=ai_model,
    deps_type=MacroAnalysisDependencies,
    result_type=MacroAnalysisResult,
    tools=[duckduckgo_search_tool(), tavily_search_tool(os.getenv('TAVILY_API_KEY'))],
    system_prompt=(
        "You are a macroeconomic analysis expert specializing in how broader economic factors "
        "impact specific stocks and sectors. "
        "You will use Tavily search to find the most relevant and recent macroeconomic information. "
        "If the Tavily search fails or has an error, you will use DuckDuckGo search to find more information. "
        "Your task is to provide detailed analysis on how "
        "macroeconomic indicators, interest rates, inflation, geopolitical events, and industry "
        "trends might affect a specific stock. Pay special attention to factors that could "
        "create opportunities for both long and short positions, such as:\n"
        "- Sector rotation trends\n"
        "- Market liquidity conditions\n"
        "- Interest rate impacts\n"
        "- Currency fluctuations\n"
        "- Supply chain disruptions\n"
        "- Regulatory changes\n"
        "- Competitive pressures\n\n"
        "You must return a structured result with the following fields:\n"
        "- ticker: The stock symbol being analyzed\n"
        "- outlook: One of 'FAVORABLE', 'NEUTRAL', or 'UNFAVORABLE'\n"
        "- economic_indicators_impact: Analysis of how economic indicators affect the stock\n"
        "- geopolitical_factors_impact: Analysis of how geopolitical factors affect the stock\n"
        "- industry_trends_impact: Analysis of how industry trends affect the stock\n"
        "- key_risks: List of key macroeconomic risks for this stock\n"
        "- opportunities: List of key macroeconomic opportunities for this stock\n"
        "- short_specific_factors: Dictionary of factors specifically relevant for short positions\n"
        "- sector_rotation_impact: Analysis of sector rotation trends\n"
        "- liquidity_analysis: Analysis of market liquidity conditions\n"
        "- summary: A comprehensive summary of the entire macroeconomic analysis\n"
        "\nEnsure your outlook value exactly matches one of the allowed values."
    )
)

# Main function to get macroeconomic analysis for a stock
def get_macro_analysis_with_agent(ticker: str, industry: Optional[str] = None, sector: Optional[str] = None) -> MacroAnalysisResult:
    """
    Get macroeconomic analysis for a stock using a specialized agent.
    
    Args:
        ticker: The stock ticker symbol
        industry: The industry the company is in (optional)
        sector: The sector the company is in (optional)
        
    Returns:
        A MacroAnalysisResult object containing structured macroeconomic analysis
    """
    logger = get_component_logger("MacroAnalysis")
    logger.info(f"Getting macro analysis for {ticker}")
    
    try:        
        deps = MacroAnalysisDependencies(ticker=ticker)

        # Run the agent with a prompt to analyze sentiment - use the sync method
        result = macro_agent.run_sync(
            f"Provide a comprehensive macroeconomic analysis for {ticker} stock, by examining economic indicators",
            deps=deps,
            model_settings={"temperature": 0.1}
        )        
        
        logger.info(f"Successfully obtained macro analysis for {ticker} with outlook: {result.data.outlook}")
        return result.data
    except Exception as e:
        logger.error(f"Error getting macro analysis: {e}")
        logger.error(traceback.format_exc())
        # Re-raise the exception to be handled by the caller
        raise

if __name__ == "__main__":
    # Simple demonstration if the file is run directly
    ticker = "TSLA"
    print(f"\nAnalyzing macroeconomic factors for {ticker}...\n")
    
    try:        
        # Get the macro analysis
        start_time = time.time()
        result = get_macro_analysis_with_agent(ticker)
        elapsed_time = time.time() - start_time
        
        # Print the results
        print("\n" + "=" * 50)
        print(f"MACROECONOMIC ANALYSIS FOR {ticker}")
        print("=" * 50)
        print(f"Outlook: {result.outlook}")
        print("\nECONOMIC INDICATORS IMPACT:")
        print(result.economic_indicators_impact)
        print("\nGEOPOLITICAL FACTORS IMPACT:")
        print(result.geopolitical_factors_impact)
        print("\nINDUSTRY TRENDS IMPACT:")
        print(result.industry_trends_impact)
        print("\nKEY RISKS:")
        for i, risk in enumerate(result.key_risks, 1):
            print(f"{i}. {risk}")
        print("\nOPPORTUNITIES:")
        for i, opportunity in enumerate(result.opportunities, 1):
            print(f"{i}. {opportunity}")
        print("\nSHORT SPECIFIC FACTORS:")
        for factor, description in result.short_specific_factors.items():
            print(f"{factor}: {description}")
        print("\nSECTOR ROTATION IMPACT:")
        print(result.sector_rotation_impact)
        print("\nLIQUIDITY ANALYSIS:")
        print(result.liquidity_analysis)
        print("\nSUMMARY:")
        print(result.summary)
        print("\n" + "-" * 50)
        print(f"Analysis completed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc() 