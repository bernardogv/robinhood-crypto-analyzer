"""
Configuration for Robinhood Crypto Analyzer

This file contains configuration settings for the Robinhood Crypto Analyzer application.
Copy this file to config.py and add your API credentials and customize settings.

API Credentials:
    - API_KEY: Your Robinhood API key from the Robinhood API Credentials Portal
    - BASE64_PRIVATE_KEY: Your base64-encoded private key (generated with utils/generate_keys.py)

Default Symbols:
    - DEFAULT_SYMBOLS: List of cryptocurrency trading pairs to analyze by default

Analysis Settings:
    - ANALYSIS_SETTINGS: Configuration for various analysis parameters

Trading Strategy Settings:
    - STRATEGY_SETTINGS: Configuration for basic trading strategies
    - XRP_STRATEGY_CONFIG: Configuration for the advanced XRP trading strategy

External API Keys (Optional):
    - EXTERNAL_API_KEYS: API keys for external services used in market analysis
"""

# Your Robinhood API key from the Robinhood API Credentials Portal
API_KEY = "your-api-key-here"

# Your base64-encoded private key (generated with utils/generate_keys.py)
BASE64_PRIVATE_KEY = "your-base64-private-key-here"

# Default symbols to analyze
DEFAULT_SYMBOLS = [
    "BTC-USD",   # Bitcoin
    "ETH-USD",   # Ethereum
    "SOL-USD",   # Solana
    "DOGE-USD",  # Dogecoin
    "ADA-USD",   # Cardano
    "XRP-USD",   # XRP (Ripple)
    # Add more symbols as needed
]

# Analysis settings
ANALYSIS_SETTINGS = {
    # Time intervals for trend analysis (in minutes)
    "time_intervals": [15, 60, 240, 1440],  # 15min, 1h, 4h, 1d
    
    # Maximum number of orders to analyze
    "max_orders": 100,
    
    # Include canceled orders in analysis
    "include_canceled": True,
    
    # Data cache settings
    "cache_enabled": True,
    "cache_duration": 300,  # 5 minutes
    
    # Volatility calculation settings
    "volatility_window": 20,
    "volatility_type": "std",  # Options: "std", "atr", "parkinson"
    
    # Logging settings
    "log_level": "INFO",  # Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    "log_file": "crypto_analyzer.log"
}

# XRP Advanced Trading Strategy Configuration
XRP_STRATEGY_CONFIG = {
    # Trading pair
    "symbol": "XRP-USD",
    
    # Technical indicator parameters
    "rsi_window": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "bb_window": 20,
    "bb_std": 2.0,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    
    # Risk management parameters
    "volatility_window": 20,
    "max_position_size": 0.1,  # 10% of available funds
    "stop_loss_pct": 0.05,     # 5% stop loss
    "take_profit_pct": 0.15,   # 15% take profit
    
    # Sentiment weight in trading decisions
    "sentiment_weight": 0.2,    # 20% weight for sentiment
    
    # Live trading parameters
    "check_interval": 300,      # Check for signals every 5 minutes
    "auto_trade": False,        # Whether to execute trades automatically
    "paper_trading": True,      # Use paper trading mode
    "data_lookback_days": 30    # Days of historical data to use
}

# Alternative XRP Strategy Configuration (more aggressive)
XRP_AGGRESSIVE_STRATEGY_CONFIG = {
    "symbol": "XRP-USD",
    "rsi_window": 14,
    "rsi_oversold": 40,      # Higher threshold for more frequent entries
    "rsi_overbought": 60,    # Lower threshold for more frequent exits
    "bb_window": 15,         # Faster response time
    "bb_std": 1.8,           # Tighter bands
    "macd_fast": 8,          # Faster response
    "macd_slow": 20,         # Faster response
    "macd_signal": 9,
    "volatility_window": 15,
    "max_position_size": 0.15,  # Larger position size
    "stop_loss_pct": 0.04,      # Tighter stop loss
    "take_profit_pct": 0.12,    # Lower profit target for more frequent wins
    "sentiment_weight": 0.3,    # More weight on sentiment
    "check_interval": 180,      # Check more frequently (3 minutes)
    "auto_trade": False,
    "paper_trading": True,
    "data_lookback_days": 20
}

# Basic Trading Strategy Settings
STRATEGY_SETTINGS = {
    # Moving Average Crossover Strategy
    "ma_crossover": {
        "short_window": 20,
        "long_window": 50,
        "signal_threshold": 0.0,   # Additional threshold to reduce false signals
        "position_size": 0.1       # 10% of available funds
    },
    
    # RSI Strategy
    "rsi": {
        "window": 14,
        "oversold": 30,
        "overbought": 70,
        "position_size": 0.1       # 10% of available funds
    },
    
    # Spread Trading Strategy
    "spread_trading": {
        "spread_threshold": 0.5,    # Minimum spread percentage to consider
        "max_spread": 5.0,          # Maximum spread to consider (avoid illiquid markets)
        "volatility_limit": 0.05,   # Maximum allowed volatility
        "position_size": 0.1        # 10% of available funds
    },
    
    # Bollinger Bands Strategy
    "bollinger_bands": {
        "window": 20,
        "std": 2.0,
        "position_size": 0.1,       # 10% of available funds
        "exit_middle_band": True    # Exit when price crosses middle band
    }
}

# External API keys (for market analysis)
EXTERNAL_API_KEYS = {
    "cryptocompare": "YOUR_CRYPTOCOMPARE_API_KEY",
    "coinmarketcap": "YOUR_COINMARKETCAP_API_KEY",
    "twitter": "YOUR_TWITTER_API_KEY"
}

# Notification settings
NOTIFICATION_SETTINGS = {
    "enabled": False,
    "email": {
        "enabled": False,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_username": "your-email@gmail.com",
        "smtp_password": "your-app-password",
        "recipient": "your-email@gmail.com",
        "sender": "crypto-analyzer@example.com"
    },
    "telegram": {
        "enabled": False,
        "bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
        "chat_id": "YOUR_TELEGRAM_CHAT_ID"
    },
    "desktop": {
        "enabled": True
    }
}

# Backtesting settings
BACKTEST_SETTINGS = {
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 10000,
    "transaction_fee_percent": 0.25,
    "slippage_percent": 0.1,
    "data_granularity": "1h",  # Options: "1m", "5m", "15m", "1h", "4h", "1d"
    "include_weekends": True,
    "plot_results": True,
    "save_results": True,
    "results_dir": "backtest_results"
}

# Dashboard settings
DASHBOARD_SETTINGS = {
    "port": 8050,
    "host": "localhost",
    "debug": False,
    "theme": "dark",  # Options: "light", "dark"
    "refresh_interval": 60000,  # milliseconds
    "max_assets_display": 10,
    "default_timeframe": "1d",  # Options: "1h", "4h", "1d", "1w", "1m" (month)
    "chart_type": "candle"  # Options: "candle", "line", "ohlc"
}
