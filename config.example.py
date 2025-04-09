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
