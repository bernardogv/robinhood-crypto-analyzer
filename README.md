# Robinhood Crypto Analyzer

A Python application for analyzing leading cryptocurrencies using the Robinhood Crypto API.

## Features

- **Authentication**: Secure access to Robinhood's Crypto API
- **Market Data Analysis**: Track the best prices and estimated prices for cryptocurrencies
- **Portfolio Analysis**: View and analyze your crypto holdings
- **Visualization**: Generate charts and graphs of crypto price trends
- **Trading Strategies**: Implement and test various trading strategies

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

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your API credentials:
   - Visit the [Robinhood API Credentials Portal](https://robinhood.com/us/en/about/crypto/) to create credentials
   - Generate your key pair using the provided scripts in `utils/generate_keys.py`
   - Store your API key and private key securely

4. Configure the application:
   - Copy `config.example.py` to `config.py`
   - Add your API key and base64-encoded private key

## Usage

Run the main analyzer script:
```
python crypto_analyzer.py
```

## Module Structure

- `crypto_api.py`: Core API client for Robinhood Crypto
- `analyzers/`: Modules for different types of crypto analysis
- `visualizers/`: Data visualization tools
- `strategies/`: Trading strategy implementations
- `utils/`: Utility functions and helper scripts
- `config.py`: Configuration settings

## License

MIT

## Disclaimer

This tool is not affiliated with Robinhood and is for educational purposes only. Trading cryptocurrencies involves risk. Make sure you understand these risks before using this tool for actual trading.
