# Robinhood Crypto Analyzer

A comprehensive Python application for analyzing and trading cryptocurrencies using the Robinhood Crypto API.

## Overview

This tool provides a robust framework for cryptocurrency analysis, combining real-time market data from Robinhood with advanced technical indicators to generate trading signals. It's designed for both educational purposes and actual trading, with features for backtesting strategies before deploying them in a live environment.

## Features

- **Authentication**: Secure access to Robinhood's Crypto API
- **Market Data Analysis**: Track and analyze cryptocurrency prices, spreads, and market trends
- **Portfolio Analysis**: View and analyze your crypto holdings and performance
- **Visualization**: Generate charts and graphs of crypto price trends and strategy performance
- **Trading Strategies**:
  - Moving Average Crossover Strategy
  - RSI (Relative Strength Index) Strategy 
  - Spread Trading Strategy
  - **XRP Advanced Trading Strategy**: Multi-indicator strategy optimized for XRP trading
- **Backtesting Framework**: Test strategies against historical data
- **Live Trading Simulation**: Simulate your strategy in a paper trading environment

## Requirements

- Python 3.8+
- Robinhood API credentials (API key and secret key)
- Dependencies listed in requirements.txt

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/bernardogv/robinhood-crypto-analyzer.git
   cd robinhood-crypto-analyzer
   ```

2. Set up a virtual environment:
   ```bash
   # Create a virtual environment
   python -m venv venv
   
   # Activate the virtual environment
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your API credentials:
   - Visit the [Robinhood API Credentials Portal](https://robinhood.com/us/en/about/crypto/) to create credentials
   - Copy `config.example.py` to `config.py`
   - Add your API key and base64-encoded private key to `config.py`

## Usage

### Basic Usage

Run the main analyzer script:
```bash
python crypto_analyzer.py
```

This will run with default settings. For more options:
```bash
python crypto_analyzer.py --help
```

### Specific Analysis Commands

```bash
# Market analysis of specific symbols
python crypto_analyzer.py --market --symbols BTC-USD ETH-USD XRP-USD

# Analyze price spreads
python crypto_analyzer.py --spreads

# Analyze your holdings
python crypto_analyzer.py --holdings

# Analyze your order history
python crypto_analyzer.py --orders

# Save results to a file
python crypto_analyzer.py --market --output market_data.json
```

### Running the XRP Strategy

```bash
# Run backtest of the XRP strategy
python examples/xrp_trading_example.py --mode backtest

# Run simulation of the XRP strategy (without actual trades)
python examples/xrp_trading_example.py --mode simulate
```

## Project Structure

- `crypto_api.py`: Core API client for Robinhood Crypto
- `crypto_analyzer.py`: Main entry point with command-line interface
- `analyzers/`: Modules for different types of crypto analysis
- `visualizers/`: Data visualization tools
- `strategies/`: Trading strategy implementations
  - `basic_strategies.py`: Implementation of basic trading strategies (MA, RSI, Spread)
  - `xrp_advanced_strategy.py`: Advanced multi-indicator XRP trading strategy
- `examples/`: Example scripts demonstrating specific use cases
- `utils/`: Utility functions and helper scripts
- `config.py`: Configuration settings

## XRP Advanced Trading Strategy

The XRP Advanced Trading Strategy is a sophisticated approach that combines multiple technical indicators to generate trading signals specifically optimized for XRP:

- **Multi-Indicator Approach**: Combines RSI, Bollinger Bands, and MACD indicators
- **Volatility-Based Position Sizing**: Adjusts position size based on market volatility
- **Dynamic Stop-Loss Mechanism**: Implements trailing stops to protect profits
- **Market Sentiment Integration**: Incorporates broader market sentiment in trading decisions

### Strategy Parameters

The strategy's behavior can be customized by modifying parameters in `config.py`:

```python
XRP_STRATEGY_CONFIG = {
    "symbol": "XRP-USD",
    "rsi_window": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "bb_window": 20,
    "bb_std": 2.0,
    # Additional parameters...
}
```

## Contributing

Contributions are welcome! If you'd like to improve the project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a new Pull Request

## License

MIT

## Disclaimer

This tool is not affiliated with Robinhood and is for educational purposes only. Trading cryptocurrencies involves risk. Understand these risks before using this tool for actual trading.
