# Robinhood Crypto Analyzer

A Python application for analyzing leading cryptocurrencies using the Robinhood Crypto API.

## Features

- **Authentication**: Secure access to Robinhood's Crypto API
- **Market Data Analysis**: Track the best prices and estimated prices for cryptocurrencies
- **Portfolio Analysis**: View and analyze your crypto holdings
- **Visualization**: Generate charts and graphs of crypto price trends
- **Trading Strategies**: Implement and test various trading strategies
  - Moving Average Crossover Strategy
  - RSI (Relative Strength Index) Strategy
  - Spread Trading Strategy
  - **XRP Advanced Trading Strategy**: Multi-indicator strategy optimized for XRP trading

## Requirements

- Python 3.8+
- Robinhood API credentials (API key and secret key)
- PyNaCl library
- Other dependencies listed in requirements.txt

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/bernardogv/robinhood-crypto-analyzer.git
   cd robinhood-crypto-analyzer
   ```

2. Set up a virtual environment (recommended):
   ```
   # Create a virtual environment
   python -m venv venv
   
   # Activate the virtual environment
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your API credentials:
   - Visit the [Robinhood API Credentials Portal](https://robinhood.com/us/en/about/crypto/) to create credentials
   - Generate your key pair using the provided scripts in `utils/generate_keys.py`:
     ```
     python utils/generate_keys.py
     ```
   - Store your API key and private key securely

5. Configure the application:
   - Copy `config.example.py` to `config.py`
   - Add your API key and base64-encoded private key
   - Adjust strategy parameters as needed

## Usage

Run the main analyzer script:
```
python crypto_analyzer.py
```

Alternatively, try one of the example scripts:
```
# Run market analysis
python examples/market_analysis.py

# Run portfolio analysis
python examples/portfolio_analysis.py

# Launch the interactive dashboard
python examples/run_dashboard.py

# Test the XRP trading strategy
python examples/xrp_trading_example.py
```

### XRP Trading Strategy

The XRP Advanced Trading Strategy combines multiple technical indicators to generate trading signals for XRP:

- **Multi-Indicator Approach**: Combines RSI, Bollinger Bands, and MACD indicators
- **Volatility-Based Position Sizing**: Adjusts position size based on market volatility
- **Dynamic Stop-Loss Mechanism**: Implements trailing stops to protect profits
- **Market Sentiment Integration**: Incorporates broader market sentiment in trading decisions

To run a backtest of the XRP trading strategy:
```
python examples/xrp_trading_example.py --mode backtest
```

To run a simulation of live trading (without actual trades):
```
python examples/xrp_trading_example.py --mode simulate
```

## Module Structure

- `crypto_api.py`: Core API client for Robinhood Crypto
- `analyzers/`: Modules for different types of crypto analysis
- `visualizers/`: Data visualization tools
- `strategies/`: Trading strategy implementations
  - `basic_strategies.py`: Implementation of basic trading strategies
  - `xrp_advanced_strategy.py`: Advanced XRP trading strategy
- `utils/`: Utility functions and helper scripts
- `config.py`: Configuration settings

## License

MIT

## Disclaimer

This tool is not affiliated with Robinhood and is for educational purposes only. Trading cryptocurrencies involves risk. Make sure you understand these risks before using this tool for actual trading.
