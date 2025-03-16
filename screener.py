#!/usr/bin/env python3
"""
Stock Screener - Identifies top swing trading candidates based on technical indicators.

The screener:
1. Gets a list of S&P 500 stocks from Wikipedia
2. Downloads 200 days of price/volume data for each stock using Alpaca API
3. Calculates technical indicators:
   - Moving Averages (50/200 day)
   - Trading Volume
   - Volatility
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Recent price changes
   - Distance from 52-week high
4. Scores stocks based on:
   - Liquidity & Volatility (0-30 points)
   - Momentum & Trend (0-40 points)
   - Recent Price Action (0-30 points)
5. Returns the top N candidates
"""

import os
import logging
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional
from bs4 import BeautifulSoup
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StockScreener')

# Load environment variables
load_dotenv()

# Alpaca API credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")

# Initialize Alpaca client
stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

def get_sp500_tickers() -> List[str]:
    """
    Scrape S&P 500 stock tickers from Wikipedia.
    Returns a list of tickers or raises an exception if scraping fails.
    """
    logger.info("Fetching S&P 500 tickers from Wikipedia...")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        
        if not table:
            raise ValueError("Could not find S&P 500 table on Wikipedia page")
        
        tickers = []
        for row in table.find_all('tr')[1:]:
            ticker = row.find_all('td')[0].text.strip()
            tickers.append(ticker.replace('.', ''))  # Remove dots from ticker names
        
        if not tickers:
            raise ValueError("No tickers found in S&P 500 table")
            
        logger.info(f"Successfully fetched {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        logger.error(f"Failed to scrape S&P 500 tickers: {e}")
        raise RuntimeError(f"Failed to retrieve S&P 500 tickers: {e}")

def get_stock_data(tickers: List[str], lookback_days: int = 200) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical stock data for the given tickers using Alpaca API.
    
    Args:
        tickers: List of stock tickers
        lookback_days: Number of days of historical data to fetch
        
    Returns:
        Dictionary mapping tickers to DataFrames containing OHLCV data
    """
    logger.info(f"Fetching {lookback_days} days of historical data for {len(tickers)} stocks...")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    # Process stocks in batches to avoid API limits
    batch_size = 50
    all_stock_data = {}
    
    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1} ({len(batch_tickers)} stocks)")
        
        try:
            # Request daily bars for the batch
            request_params = StockBarsRequest(
                symbol_or_symbols=batch_tickers,
                timeframe=TimeFrame.Day,
                start=start_date.date(),
                end=end_date.date(),
                adjustment=Adjustment.ALL
            )
            
            bars_data = stock_client.get_stock_bars(request_params)
            
            # Process each ticker's data
            for ticker in batch_tickers:
                if ticker in bars_data.data:
                    # Convert to DataFrame
                    df = pd.DataFrame([bar.model_dump() for bar in bars_data.data[ticker]])
                    
                    # Set timestamp as index and ensure columns exist
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    
                    # Only include stocks with sufficient data
                    if len(df) >= 50:  # At least 50 days of data
                        all_stock_data[ticker] = df
        except Exception as e:
            logger.error(f"Error fetching data for batch: {e}")
    
    logger.info(f"Successfully retrieved data for {len(all_stock_data)} stocks")
    return all_stock_data

def calculate_technical_indicators(stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """
    Calculate technical indicators for each stock.
    
    Args:
        stock_data: Dictionary mapping tickers to DataFrames with OHLCV data
        
    Returns:
        Dictionary with technical analysis for each stock
    """
    logger.info(f"Calculating technical indicators for {len(stock_data)} stocks...")
    
    # Add debugging for the first few stocks
    sample_tickers = list(stock_data.keys())[:5]
    
    analysis_results = {}
    skipped_due_to_length = 0
    skipped_due_to_columns = 0
    skipped_due_to_error = 0
    
    for ticker, df in stock_data.items():
        try:
            # Skip if not enough data - reduced from 200 to 100 days
            if len(df) < 100:
                skipped_due_to_length += 1
                continue
                
            # Make sure we have the necessary columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                skipped_due_to_columns += 1
                continue
            
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
            df['change_5d'] = df['close'].pct_change(periods=5)
            df['change_10d'] = df['close'].pct_change(periods=10)
            df['change_20d'] = df['close'].pct_change(periods=20)
            
            # Calculate 52-week high and proximity to it
            year_high = df['close'].rolling(window=min(252, len(df))).max()  # Use available data if less than 252 days
            df['pct_from_high'] = (df['close'] - year_high) / year_high * 100
            
            # Get the latest values for analysis
            latest = df.iloc[-1].copy()
            
            # Fill any NaN values in the latest data
            for col in latest.index:
                if pd.isna(latest[col]):
                    if col in ['rsi', 'volatility', 'change_5d', 'change_10d', 'change_20d', 'pct_from_high']:
                        latest[col] = 0  # Default value for percentages and indicators
                    elif col in ['sma_50', 'sma_200', 'avg_volume']:
                        latest[col] = latest['close'] if col.startswith('sma') else latest['volume']  # Use current values
            
            # Store all analysis in a dictionary
            analysis_results[ticker] = {
                'current_price': latest['close'],
                'sma_50': latest['sma_50'],
                'sma_200': latest['sma_200'],
                'volume': latest['volume'],
                'avg_volume': latest['avg_volume'],
                'volatility': latest['volatility'],
                'rsi': latest['rsi'],
                'macd': latest['macd'],
                'macd_signal': latest['macd_signal'],
                'macd_hist': latest['macd_hist'],
                'change_5d': latest['change_5d'] * 100,  # Convert to percentage
                'change_10d': latest['change_10d'] * 100,
                'change_20d': latest['change_20d'] * 100,
                'pct_from_high': latest['pct_from_high'],
                'above_sma_50': latest['close'] > latest['sma_50'],
                'above_sma_200': latest['close'] > latest['sma_200'],
                'golden_cross': latest['sma_50'] > latest['sma_200'],
                'volume_price_ratio': (latest['volume'] * latest['close']) / 1_000_000  # In millions
            }
        except Exception as e:
            skipped_due_to_error += 1
            logger.warning(f"Error calculating indicators for {ticker}: {e}")
    
    logger.info(f"Stocks skipped due to: insufficient data length: {skipped_due_to_length}, missing columns: {skipped_due_to_columns}, errors: {skipped_due_to_error}")
    logger.info(f"Completed technical analysis for {len(analysis_results)} stocks")
    return analysis_results

def score_stocks(analysis_results: Dict[str, Dict]) -> List[Dict]:
    """
    Score each stock based on technical indicators and return sorted results.
    
    Args:
        analysis_results: Dictionary with technical analysis for each stock
        
    Returns:
        List of dictionaries with stock scores, sorted by total score
    """
    logger.info(f"Scoring {len(analysis_results)} stocks...")
    
    scored_stocks = []
    
    for ticker, analysis in analysis_results.items():
        try:
            score = {
                'ticker': ticker,
                'current_price': analysis['current_price'],
                'liquidity_volatility_score': 0,
                'momentum_trend_score': 0,
                'price_action_score': 0,
                'total_score': 0,
                'technical_indicators': {},
                'rationale': []
            }
            
            # 1. Liquidity & Volatility Score (0-30 points)
            
            # Volume criteria (0-15 points)
            volume_score = 0
            if analysis['volume_price_ratio'] > 50:  # Over $50M daily volume
                volume_score = 15
                score['rationale'].append("High trading volume")
            elif analysis['volume_price_ratio'] > 20:  # Over $20M daily volume
                volume_score = 10
                score['rationale'].append("Good trading volume")
            elif analysis['volume_price_ratio'] > 5:  # Over $5M daily volume
                volume_score = 5
                score['rationale'].append("Adequate trading volume")
            score['liquidity_volatility_score'] += volume_score
            
            # Volatility criteria (0-15 points)
            volatility_score = 0
            if 0.25 < analysis['volatility'] < 0.6:  # Ideal volatility range
                volatility_score = 15
                score['rationale'].append("Ideal volatility for swing trading")
            elif 0.15 < analysis['volatility'] < 0.8:  # Acceptable volatility
                volatility_score = 10
                score['rationale'].append("Good volatility range")
            elif 0.1 < analysis['volatility'] < 1.0:  # Borderline volatility
                volatility_score = 5
                score['rationale'].append("Acceptable volatility")
            score['liquidity_volatility_score'] += volatility_score
            
            # 2. Momentum & Trend Score (0-40 points)
            
            # Moving Average Position (0-15 points)
            ma_score = 0
            if analysis['above_sma_50'] and analysis['above_sma_200'] and analysis['golden_cross']:
                ma_score = 15
                score['rationale'].append("Strong uptrend (above 50 & 200 SMAs with golden cross)")
            elif analysis['above_sma_50'] and analysis['above_sma_200']:
                ma_score = 10
                score['rationale'].append("Solid uptrend (above 50 & 200 SMAs)")
            elif analysis['above_sma_50'] or analysis['above_sma_200']:
                ma_score = 5
                score['rationale'].append("Mixed trend signals")
            score['momentum_trend_score'] += ma_score
            
            # RSI criteria (0-10 points)
            rsi_score = 0
            if 40 <= analysis['rsi'] <= 70:  # Ideal RSI range
                rsi_score = 10
                score['rationale'].append("RSI in ideal range (40-70)")
            elif 30 <= analysis['rsi'] < 40:  # Potentially oversold
                rsi_score = 5
                score['rationale'].append("Potentially oversold (RSI 30-40)")
            elif 70 < analysis['rsi'] <= 75:  # Strong momentum but approaching overbought
                rsi_score = 5
                score['rationale'].append("Strong momentum but approaching overbought")
            score['momentum_trend_score'] += rsi_score
            
            # MACD criteria (0-15 points)
            macd_score = 0
            if analysis['macd'] > 0 and analysis['macd_hist'] > 0:
                macd_score = 15
                score['rationale'].append("Strong positive MACD with bullish histogram")
            elif analysis['macd'] > 0:
                macd_score = 10
                score['rationale'].append("Positive MACD")
            elif analysis['macd_hist'] > 0:
                macd_score = 5
                score['rationale'].append("MACD histogram turning positive")
            score['momentum_trend_score'] += macd_score
            
            # 3. Recent Price Action Score (0-30 points)
            
            # Recent price changes (0-20 points)
            price_change_score = 0
            if analysis['change_5d'] > 0 and analysis['change_10d'] > 0 and analysis['change_20d'] > 0:
                price_change_score = 20
                score['rationale'].append("Bullish across 5, 10, and 20-day periods")
            elif analysis['change_5d'] > 0 and analysis['change_10d'] > 0:
                price_change_score = 15
                score['rationale'].append("Bullish across 5 and 10-day periods")
            elif analysis['change_5d'] > 0:
                price_change_score = 10
                score['rationale'].append("Bullish 5-day price change")
            elif analysis['change_20d'] > 0:
                price_change_score = 5
                score['rationale'].append("Bullish 20-day trend despite recent weakness")
            score['price_action_score'] += price_change_score
            
            # Distance from 52-week high (0-10 points)
            high_score = 0
            if analysis['pct_from_high'] > -5:  # Within 5% of 52-week high
                high_score = 10
                score['rationale'].append("Near 52-week high (strong momentum)")
            elif analysis['pct_from_high'] > -10:  # Within 10% of 52-week high
                high_score = 8
                score['rationale'].append("Approaching 52-week high")
            elif analysis['pct_from_high'] > -20:  # Within 20% of 52-week high
                high_score = 5
                score['rationale'].append("Within striking distance of 52-week high")
            score['price_action_score'] += high_score
            
            # Calculate total score
            score['total_score'] = (
                score['liquidity_volatility_score'] + 
                score['momentum_trend_score'] + 
                score['price_action_score']
            )
            
            # Add key technical indicators as strings
            score['technical_indicators'] = {
                "Price": f"${analysis['current_price']:.2f}",
                "50-day MA": f"${analysis['sma_50']:.2f}",
                "200-day MA": f"${analysis['sma_200']:.2f}",
                "RSI": f"{analysis['rsi']:.1f}",
                "Daily Volume": f"${(analysis['volume_price_ratio']):.1f}M",
                "Volatility": f"{analysis['volatility']*100:.1f}%",
                "5-day Change": f"{analysis['change_5d']:.1f}%",
                "MACD": f"{analysis['macd']:.2f}",
                "From 52w High": f"{analysis['pct_from_high']:.1f}%",
                "Total Score": f"{score['total_score']}"
            }
            
            # Only include stocks with minimum criteria
            if score['total_score'] >= 30 and analysis['volume_price_ratio'] > 1.0:
                scored_stocks.append(score)
                
        except Exception as e:
            logger.warning(f"Error scoring {ticker}: {e}")
    
    # Sort by total score (descending)
    scored_stocks.sort(key=lambda x: x['total_score'], reverse=True)
    
    logger.info(f"Scored {len(scored_stocks)} stocks")
    return scored_stocks

def format_results(scored_stocks: List[Dict], top_n: int = 3) -> Tuple[List[Dict], str]:
    """
    Format the top N stocks for display and return both structured data and a summary.
    
    Args:
        scored_stocks: List of scored stock dictionaries
        top_n: Number of top stocks to include
        
    Returns:
        Tuple containing:
        - List of top stock candidates with formatted data
        - Summary text describing the screening results
    """
    top_stocks = scored_stocks[:top_n]
    
    formatted_stocks = []
    summary = f"Top {len(top_stocks)} Swing Trading Candidates\n\n"
    
    for i, stock in enumerate(top_stocks, 1):
        # Get current price and technical indicators
        current_price = stock['current_price']
        
        # Calculate entry, stop loss and target prices based on technical indicators
        # Use volatility to determine appropriate stop loss distance
        volatility = float(stock['technical_indicators']['Volatility'].strip('%')) / 100
        
        # Calculate an ATR-like measure for proper stop loss placement (more volatile stocks get wider stops)
        stop_distance_pct = min(max(volatility * 2, 0.03), 0.08)  # Between 3% and 8% based on volatility
        
        # Calculate prices based on volatility and technical strength
        entry_price = round(current_price * (1 - stop_distance_pct * 0.4), 2)  # Entry closer to current price
        stop_loss = round(current_price * (1 - stop_distance_pct), 2)
        
        # Target based on risk/reward of at least 2:1
        risk = current_price - stop_loss
        target_price = round(current_price + (risk * 2), 2)
        
        # Calculate risk/reward ratio
        risk_reward = f"1:{(target_price - entry_price) / (entry_price - stop_loss):.1f}"
        
        # Create a formatted stock candidate
        candidate = {
            "ticker": stock['ticker'],
            "current_price": current_price,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target_price": target_price,
            "risk_reward": risk_reward,
            "conviction": "High" if stock['total_score'] >= 70 else "Medium" if stock['total_score'] >= 50 else "Low",
            "holding_period": "5-15 days",  # Based on typical swing trade duration
            "technical_indicators": stock['technical_indicators'],
            "catalysts": [f"Technical strength (Score: {stock['total_score']}/100)"],
            "rationale": ", ".join(stock['rationale'][:3])  # First 3 rationales
        }
        
        formatted_stocks.append(candidate)
        
        # Build text summary
        summary += f"#{i}: {stock['ticker']} (Score: {stock['total_score']}/100)\n"
        summary += f"Price: ${current_price:.2f} | "
        summary += f"Entry: ${candidate['entry_price']:.2f} | "
        summary += f"Stop: ${candidate['stop_loss']:.2f} | "
        summary += f"Target: ${candidate['target_price']:.2f}\n"
        summary += f"Key indicators: RSI: {stock['technical_indicators']['RSI']}, "
        summary += f"Volume: {stock['technical_indicators']['Daily Volume']}, "
        summary += f"5d Change: {stock['technical_indicators']['5-day Change']}\n"
        summary += f"Rationale: {candidate['rationale']}\n\n"
    
    logger.info(f"Formatted results for top {len(formatted_stocks)} stocks")
    return formatted_stocks, summary

def screen_for_swing_trades(max_stocks: int = 3) -> Tuple[List[Dict], str]:
    """
    Main function to screen for swing trading candidates.
    
    Args:
        max_stocks: Maximum number of top stocks to return
        
    Returns:
        Tuple containing:
        - List of top stock candidates with all data
        - Summary text describing the screening results
        
    Raises:
        RuntimeError: If unable to retrieve S&P 500 tickers or process stock data
    """
    logger.info(f"Starting swing trade screening process for top {max_stocks} candidates")
    
    # Get S&P 500 tickers
    tickers = get_sp500_tickers()  # May raise RuntimeError
    logger.info(f"Retrieved {len(tickers)} tickers to analyze")
    
    if not tickers:
        raise RuntimeError("No tickers available for analysis")
    
    # Get historical stock data
    stock_data = get_stock_data(tickers, lookback_days=252)  # 1 year of data
    
    if not stock_data:
        raise RuntimeError("No stock data retrieved from Alpaca API")
    
    # Calculate technical indicators
    analysis_results = calculate_technical_indicators(stock_data)
    
    if not analysis_results:
        raise RuntimeError("No technical analysis results generated")
    
    # Score stocks based on technical criteria
    scored_stocks = score_stocks(analysis_results)
    
    if not scored_stocks:
        logger.warning("No stocks met the minimum scoring criteria")
        return [], "No stocks met the screening criteria. Try adjusting the parameters."
    
    # Format results for the top N stocks
    candidates, summary = format_results(scored_stocks, top_n=max_stocks)
    
    logger.info("Swing trade screening completed successfully")
    return candidates, summary

if __name__ == "__main__":
    try:
        # Run the screener
        candidates, summary = screen_for_swing_trades(max_stocks=5)
        
        # Display results
        if candidates:
            print(summary)
        else:
            print("No suitable swing trading candidates found.")
    except Exception as e:
        logger.error(f"Stock screening failed: {e}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
