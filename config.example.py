"""
Configuration for Robinhood Crypto Analyzer

Copy this file to config.py and add your API credentials.
"""

# Your Robinhood API key from the Robinhood API Credentials Portal
API_KEY = "your-api-key-here"

# Your base64-encoded private key (generated with the key generation script)
BASE64_PRIVATE_KEY = "your-base64-private-key-here"

# Optional: Custom list of default symbols to analyze
DEFAULT_SYMBOLS = [
    "BTC-USD",  # Bitcoin
    "ETH-USD",  # Ethereum
    "SOL-USD",  # Solana
    "DOGE-USD", # Dogecoin
    "ADA-USD",  # Cardano
    "XRP-USD",  # XRP (Ripple)
    # Add more symbols as needed
]

# Optional: Settings for analysis
ANALYSIS_SETTINGS = {
    # Time intervals for trend analysis (in minutes)
    "time_intervals": [15, 60, 240, 1440],
    
    # Maximum number of orders to analyze
    "max_orders": 100,
    
    # Include canceled orders in analysis
    "include_canceled": True
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
    "sentiment_weight": 0.2,   # 20% weight for sentiment
    
    # Live trading parameters
    "check_interval": 300,     # Check for signals every 5 minutes
    "auto_trade": False,       # Whether to execute trades automatically
    "paper_trading": True,     # Use paper trading mode
    "data_lookback_days": 30   # Days of historical data to use
}

# External API keys (for market analysis)
EXTERNAL_API_KEYS = {
    "cryptocompare": "YOUR_CRYPTOCOMPARE_API_KEY",
    "coinmarketcap": "YOUR_COINMARKETCAP_API_KEY",
    "twitter": "YOUR_TWITTER_API_KEY"
}

# Trading Strategy Settings
STRATEGY_SETTINGS = {
    # Moving Average Crossover Strategy
    "ma_crossover": {
        "short_window": 20,
        "long_window": 50
    },
    
    # RSI Strategy
    "rsi": {
        "window": 14,
        "oversold": 30,
        "overbought": 70
    },
    
    # Spread Trading Strategy
    "spread_trading": {
        "spread_threshold": 0.5
    }
}
