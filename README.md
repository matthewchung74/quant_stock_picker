# Quantitative Stock Analysis Agent

A financial analysis tool that uses specialized AI agents and real-time market data to perform comprehensive analysis of stocks for trading strategy development, supporting both long and short positions.

## Quick Start with Docker Compose

1. Make sure you have Docker and Docker Compose installed on your system
2. Clone this repository
3. Create your `.env` file:
   ```bash
   cp env.example .env
   ```
4. Configure your environment:
   - Edit the `.env` file with your API keys (see "Required API Keys" section below)
   - Set `BACKTEST=true` in `.env` if you want to run in backtest mode
5. Start the application:
   ```bash
   docker-compose up --build
   ```

The application will automatically:
- Build with all required dependencies
- Mount your local directory for easy development
- Create necessary directories (journal, logs, output)
- Start running with your configuration
- Run in backtest mode if BACKTEST=true in your .env file

To stop the application:
```bash
docker-compose down
```

## Required API Keys

The following keys need to be configured in your `.env` file:

- **DEEPSEEK_API_KEY** (required): For the DeepSeek AI model
- **ALPACA_API_KEY_LLM_AGENT** and **ALPACA_API_SECRET_LLM_AGENT**: For market data access and trading execution
- **ALPACA_PAPER_TRADING_LLM_AGENT**: Set to "true" to use paper trading (recommended)
- **POLYGON_API_KEY**: For real-time and historical market data
- **TAVILY_API_KEY** (optional): For enhanced news search when DuckDuckGo has rate limits or fails

Additional configuration options:
- **BACKTEST**: Set to "true" to run in backtest mode
- **ALPACA_BASE_URL**: API endpoint (defaults to paper trading URL)

## Overview

This application uses a multi-agent architecture powered by DeepSeek AI, combined with Alpaca Markets financial data APIs. Each aspect of stock analysis is handled by a dedicated specialized agent:

1. 📈 **Market Data Agent** - Retrieves and analyzes current financial data, including:
   - Real-time price data and recent price changes
   - Technical indicators (RSI, SMA-50, SMA-200, VWAP)
   - Support and resistance levels
   - Volume analysis and patterns
   - Short interest and borrowing metrics

2. 📰 **Sentiment Analysis Agent** - Evaluates market sentiment through:
   - Recent news articles and their impact (using DuckDuckGo with Tavily fallback)
   - Social media sentiment trends
   - Analyst ratings and price targets
   - Institutional investor positioning
   - Short seller activity and bearish sentiment

3. 🌐 **Macro-Economic Analysis Agent** - Assesses the broader economic context:
   - Economic indicators impact (interest rates, inflation, etc.)
   - Geopolitical factors and their market implications
   - Industry trends and sector rotation
   - Short-specific macroeconomic factors
   - Key risks and opportunities in the current environment

4. 📊 **Strategy Development Agent** - Creates comprehensive trading strategies:
   - Long positions (buying with expectation of price increase)
   - Short positions (selling borrowed shares with expectation of price decrease)
   - Specific entry points, stop-loss levels, and profit targets for both strategies
   - Position-specific metrics (borrowing costs for shorts, etc.)

## Output and Results

After analysis completes, you will find:

1. **Journal Directory**: Contains CSV files with detailed trade records
   ```
   journal/
   ├── trades_YYYYMMDD.csv          # Live trading records
   └── backtest_trades_YYYYMMDD.csv # Backtest trading records
   ```

2. **Logs Directory**: Contains detailed execution logs
   ```
   logs/
   ├── trader.log
   └── backtest.log
   ```

3. **Output Directory**: Contains analysis results
   ```
   output/
   └── recommendations_YYYYMMDD.csv
   ```

Example recommendations CSV format:
```
Ticker,Current Price,Position Type,Expiration Date,Entry Price,Stop Loss Price,Profit Target,Risk/Reward,Summary,...
AIG,$83.13,LONG,2025-03-18,$83.13,$82.00,$85.00,1:2,Day trading long strategy for AIG stock ..."
GME,$18.45,SHORT,2025-03-18,$18.45,$19.50,$16.00,1:2.5,Shorting GME based on technical breakdown..."
```

## Technical Stack

- **Docker & Docker Compose**: For containerized deployment
- **Python 3.12+**: Core programming language
- **DeepSeek AI Models**: For intelligent analysis and strategy generation
- **Alpaca Markets API**: For historical market data and trade execution
- **Lumibot**: For automated trading strategy implementation
- **Polygon.io API**: For real-time market data
- **Logging**: Comprehensive logging system with component-specific loggers

## Who This Is For

This tool is designed for:
- Educational purposes for learning about quantitative finance
- Understanding AI agent architectures in financial applications
- Exploring multi-perspective stock analysis methodologies for both long and short strategies

**IMPORTANT DISCLAIMER**: This software is provided for EDUCATIONAL PURPOSES ONLY. It is not financial advice and should not be used to make real investment decisions. The strategies and analyses generated by this tool are experimental and have not been validated for real-world trading. Always consult with a licensed financial advisor before making investment decisions.

## License

MIT 

