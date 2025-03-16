#!/usr/bin/env python3
"""
Market Data Fetcher

This module provides functions to fetch market data using Alpaca API and populate
the MarketData model. It can be used as a standalone script or imported into other modules.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
import traceback

# Import Alpaca specific libraries
from alpaca.data import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import Adjustment
from pydantic import BaseModel, Field

# Import the logging utilities
try:
    from logging_utils import get_component_logger
    logger = get_component_logger("MarketDataFetcher")
except ImportError:
    # Fallback logging if logging_utils is not available
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("MarketDataFetcher")

# Load environment variables
load_dotenv()

# Check for Alpaca API credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")

if not ALPACA_API_KEY or not ALPACA_API_SECRET:
    logger.warning("Alpaca API credentials not found. Market data retrieval may fail.")

# Initialize Alpaca client
try:
    alpaca_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    logger.info("Alpaca client initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Alpaca client: {e}")
    alpaca_client = None

class PriceIndicator(BaseModel):
    """Model for a technical price indicator."""
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    rsi: Optional[float] = None
    vwap: Optional[float] = None

class SupportResistanceLevels(BaseModel):
    """Model for support and resistance price levels."""
    support: List[float] = []
    resistance: List[float] = []

class MarketData(BaseModel):
    """Model for market data of a stock."""
    latest_price: Optional[float] = None
    volume: Optional[float] = None
    ticker: Optional[str] = None
    price_changes: Dict[str, float] = {}
    latest_indicators: Dict[str, PriceIndicator] = {}
    support_resistance: SupportResistanceLevels = SupportResistanceLevels()


def get_stock_data(ticker: str, lookback_days: int = 200) -> Optional[pd.DataFrame]:
    """
    Fetch historical stock data for the given ticker using Alpaca API.
    
    Args:
        ticker: Stock ticker symbol
        lookback_days: Number of days of historical data to fetch
        
    Returns:
        DataFrame containing OHLCV data or None if data retrieval fails
    """
    logger.info(f"Fetching {lookback_days} days of historical data for {ticker}")
    
    if not alpaca_client:
        logger.error("Alpaca client not initialized. Cannot fetch data.")
        return None
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    try:
        # Request daily bars for the ticker
        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start_date.date(),
            end=end_date.date(),
            adjustment=Adjustment.ALL
        )
        
        bars_data = alpaca_client.get_stock_bars(request_params)
        
        # Process the data
        if ticker in bars_data.data:
            # Convert to DataFrame
            df = pd.DataFrame([bar.model_dump() for bar in bars_data.data[ticker]])
            
            # Set timestamp as index and ensure columns exist
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Only include stocks with sufficient data
            if len(df) >= 50:  # At least 50 days of data
                logger.info(f"Successfully retrieved {len(df)} days of data for {ticker}")
                return df
            else:
                logger.warning(f"Insufficient data for {ticker}: only {len(df)} days available")
                return df if len(df) > 0 else None
        else:
            logger.warning(f"No data available for {ticker}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_technical_indicators(df: pd.DataFrame) -> Dict:
    """
    Calculate technical indicators for the stock data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Dictionary with technical indicators
    """
    logger.info("Calculating technical indicators")
    
    try:
        # Calculate Moving Averages
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Calculate Exponential Moving Averages for MACD
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        # Improved handling of division by zero and NaN values
        rs = pd.Series(np.where(avg_loss == 0, 100, avg_gain / avg_loss.replace(0, np.nan)), index=avg_loss.index)
        rs = rs.fillna(0)  # Replace NaN with 0
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate Volatility (20-day standard deviation of returns)
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)  # Annualized
        
        # Calculate Average Volume (20-day)
        df['avg_volume'] = df['volume'].rolling(window=20).mean()
        
        # Calculate Recent Price Changes
        df['change_1d'] = df['close'].pct_change(periods=1) * 100  # as percentage
        df['change_5d'] = df['close'].pct_change(periods=5) * 100
        df['change_1m'] = df['close'].pct_change(periods=20) * 100  # ~20 trading days
        df['change_3m'] = df['close'].pct_change(periods=60) * 100  # ~60 trading days
        
        # Calculate VWAP (Volume Weighted Average Price) - simplified daily version
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Get the latest values for analysis
        latest = df.iloc[-1].copy()
        
        # Fill any NaN values in the latest data
        for col in latest.index:
            if pd.isna(latest[col]):
                if col in ['rsi', 'volatility', 'change_1d', 'change_5d', 'change_1m', 'change_3m']:
                    latest[col] = 0  # Default value for percentages and indicators
                elif col in ['sma_50', 'sma_200', 'vwap']:
                    latest[col] = latest['close']  # Use current price
                elif col == 'avg_volume':
                    latest[col] = latest['volume']  # Use current volume
        
        # Identify support and resistance levels
        # Simple method: use recent lows as support and recent highs as resistance
        support_levels = []
        resistance_levels = []
        
        # Look for local minima and maxima in the last 60 days
        if len(df) >= 60:
            recent_df = df.iloc[-60:]
            
            # Find local minima (support)
            for i in range(2, len(recent_df) - 2):
                if (recent_df['low'].iloc[i] < recent_df['low'].iloc[i-1] and 
                    recent_df['low'].iloc[i] < recent_df['low'].iloc[i-2] and
                    recent_df['low'].iloc[i] < recent_df['low'].iloc[i+1] and
                    recent_df['low'].iloc[i] < recent_df['low'].iloc[i+2]):
                    support_levels.append(round(recent_df['low'].iloc[i], 2))
            
            # Find local maxima (resistance)
            for i in range(2, len(recent_df) - 2):
                if (recent_df['high'].iloc[i] > recent_df['high'].iloc[i-1] and 
                    recent_df['high'].iloc[i] > recent_df['high'].iloc[i-2] and
                    recent_df['high'].iloc[i] > recent_df['high'].iloc[i+1] and
                    recent_df['high'].iloc[i] > recent_df['high'].iloc[i+2]):
                    resistance_levels.append(round(recent_df['high'].iloc[i], 2))
        
        # Limit to the 3 most relevant levels (closest to current price)
        current_price = latest['close']
        
        # Sort support levels (ascending) and resistance levels (descending)
        support_levels = sorted(set(support_levels))
        resistance_levels = sorted(set(resistance_levels), reverse=True)
        
        # Filter to levels that are reasonably close to current price
        support_levels = [level for level in support_levels if level < current_price and level > current_price * 0.8]
        resistance_levels = [level for level in resistance_levels if level > current_price and level < current_price * 1.2]
        
        # Take the 3 closest levels
        support_levels = support_levels[-3:] if support_levels else []
        resistance_levels = resistance_levels[:3] if resistance_levels else []
        
        # If we don't have enough levels, add some based on percentages
        if len(support_levels) < 2:
            support_levels.append(round(current_price * 0.95, 2))  # 5% below current price
            support_levels.append(round(current_price * 0.9, 2))   # 10% below current price
            support_levels = sorted(set(support_levels))
        
        if len(resistance_levels) < 2:
            resistance_levels.append(round(current_price * 1.05, 2))  # 5% above current price
            resistance_levels.append(round(current_price * 1.1, 2))   # 10% above current price
            resistance_levels = sorted(set(resistance_levels), reverse=True)
        
        # Create price changes dictionary
        price_changes = {
            '1d': latest['change_1d'],
            '5d': latest['change_5d'],
            '1m': latest['change_1m'],
            '3m': latest['change_3m']
        }
        
        # Create indicators dictionary for daily timeframe
        indicators = {
            'daily': {
                'sma_50': latest['sma_50'],
                'sma_200': latest['sma_200'],
                'rsi': latest['rsi'],
                'vwap': latest['vwap']
            }
        }
        
        # Return all analysis in a dictionary
        return {
            'latest_price': latest['close'],
            'volume': latest['volume'],
            'price_changes': price_changes,
            'latest_indicators': indicators,
            'support_levels': support_levels,
            'resistance_levels': resistance_levels
        }
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        logger.error(traceback.format_exc())
        return {}

def get_market_data(ticker: str) -> Dict:
    """
    Get market data for a stock that can be used to populate the MarketData model.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with market data that matches the MarketData model structure
    """
    logger.info(f"Getting market data for {ticker}")
    
    try:
        # Get historical stock data
        df = get_stock_data(ticker, lookback_days=252)  # 1 year of data
        
        if df is None or len(df) == 0:
            logger.warning(f"No data available for {ticker}")
            return {
                'ticker': ticker,
                'latest_price': None,
                'volume': None,
                'price_changes': {},
                'latest_indicators': {},
                'support_resistance': {'support': [], 'resistance': []}
            }
        
        # Calculate technical indicators
        analysis = calculate_technical_indicators(df)
        
        # Format the data to match the MarketData model
        market_data = {
            'ticker': ticker,
            'latest_price': analysis.get('latest_price'),
            'volume': analysis.get('volume'),
            'price_changes': analysis.get('price_changes', {}),
            'latest_indicators': analysis.get('latest_indicators', {}),
            'support_resistance': {
                'support': analysis.get('support_levels', []),
                'resistance': analysis.get('resistance_levels', [])
            }
        }
        
        logger.info(f"Successfully retrieved market data for {ticker}")
        return market_data
        
    except Exception as e:
        logger.error(f"Error getting market data for {ticker}: {e}")
        logger.error(traceback.format_exc())
        
        # Return a minimal structure with the ticker
        return {
            'ticker': ticker,
            'latest_price': None,
            'volume': None,
            'price_changes': {},
            'latest_indicators': {},
            'support_resistance': {'support': [], 'resistance': []}
        }

def get_market_data_with_agent(ticker: str) -> MarketData:
    """
    Get market data for a stock and return it as a MarketData object.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        MarketData object with the fetched data
    """
    # Get the market data
    data = get_market_data(ticker)
    
    # Create and return a MarketData object
    return MarketData(**data)

def main():
    """
    Main function for standalone execution.
    """
    ticker = "TSLA"

    # Get the market data
    market_data = get_market_data(ticker)
    
    # Print the market data
    print(f"\nMarket Data for {ticker}:")
    print(f"Latest Price: ${market_data['latest_price']:.2f}" if market_data['latest_price'] else "Latest Price: N/A")
    print(f"Volume: {market_data['volume']:,.0f}" if market_data['volume'] else "Volume: N/A")
    
    print("\nPrice Changes:")
    for period, change in market_data['price_changes'].items():
        print(f"  {period}: {change:.2f}%")
    
    print("\nTechnical Indicators:")
    for timeframe, indicators in market_data['latest_indicators'].items():
        print(f"  {timeframe.capitalize()} Timeframe:")
        for indicator, value in indicators.items():
            if indicator.startswith('sma') or indicator == 'vwap':
                print(f"    {indicator.upper()}: ${value:.2f}")
            else:
                print(f"    {indicator.upper()}: {value:.2f}")
    
    print("\nSupport Levels:")
    for level in market_data['support_resistance']['support']:
        print(f"  ${level:.2f}")
    
    print("\nResistance Levels:")
    for level in market_data['support_resistance']['resistance']:
        print(f"  ${level:.2f}")
    
if __name__ == "__main__":
    main() 