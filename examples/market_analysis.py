#!/usr/bin/env python3
"""
Market Analysis Example for Robinhood Crypto Analyzer

This script demonstrates how to use the Robinhood Crypto Analyzer
to perform real-time market analysis.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our modules
from crypto_api import RobinhoodCryptoAPI
from visualizers.price_charts import plot_price_comparison, plot_price_spread, simulate_price_history
from strategies.basic_strategies import MovingAverageCrossover, RelativeStrengthIndex

# Try to import the API credentials from config.py
try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import API_KEY, BASE64_PRIVATE_KEY
except ImportError:
    print("Error: API credentials not found.")
    print("Please create a config.py file with your API credentials.")
    sys.exit(1)

# Define the symbols to analyze
SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "ADA-USD"]

def main():
    """Main function."""
    print("Robinhood Crypto Market Analysis Example")
    print("=======================================")
    
    # Initialize the API client
    print("\nInitializing API client...")
    api = RobinhoodCryptoAPI(API_KEY, BASE64_PRIVATE_KEY)
    
    # Get account information
    print("\nGetting account information...")
    try:
        account_info = api.get_account()
        print(f"Account Number: {account_info.get('account_number', 'N/A')}")
        print(f"Status: {account_info.get('status', 'N/A')}")
        print(f"Buying Power: {account_info.get('buying_power', 'N/A')} {account_info.get('buying_power_currency', 'USD')}")
    except Exception as e:
        print(f"Error getting account information: {e}")
    
    # Get trading pairs
    print("\nGetting trading pairs...")
    try:
        trading_pairs = api.get_trading_pairs(*SYMBOLS)
        if "results" in trading_pairs:
            print(f"Found {len(trading_pairs['results'])} trading pairs:")
            for pair in trading_pairs["results"]:
                symbol = pair.get("symbol", "N/A")
                min_order = pair.get("min_order_size", "N/A")
                max_order = pair.get("max_order_size", "N/A")
                print(f"  - {symbol}: Min Order: {min_order}, Max Order: {max_order}")
    except Exception as e:
        print(f"Error getting trading pairs: {e}")
    
    # Get best prices
    print("\nGetting best prices...")
    try:
        best_prices = api.get_best_bid_ask(*SYMBOLS)
        
        if "results" in best_prices:
            print(f"Current prices ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):")
            
            # Prepare data for visualization
            price_data = {}
            
            for price in best_prices["results"]:
                symbol = price.get("symbol", "N/A")
                bid = float(price.get("bid_price", 0))
                ask = float(price.get("ask_price", 0))
                spread = ask - bid
                spread_percent = (spread / bid) * 100 if bid > 0 else 0
                
                price_data[symbol] = {
                    "bid": bid,
                    "ask": ask,
                    "mid_price": (bid + ask) / 2,
                    "spread_amount": spread,
                    "spread_percent": spread_percent
                }
                
                print(f"  - {symbol}: Bid: ${bid:.2f}, Ask: ${ask:.2f}, Spread: ${spread:.2f} ({spread_percent:.2f}%)")
            
            # Visualize price comparison
            print("\nGenerating price comparison chart...")
            fig = plot_price_comparison(price_data)
            plt.savefig("price_comparison.png")
            plt.close(fig)
            print("Chart saved to price_comparison.png")
            
            # Visualize price spreads
            print("\nGenerating price spread chart...")
            fig = plot_price_spread(price_data)
            plt.savefig("price_spread.png")
            plt.close(fig)
            print("Chart saved to price_spread.png")
            
            # Simulate price history for BTC-USD
            print("\nSimulating historical price data for BTC-USD...")
            btc_price = price_data["BTC-USD"]["mid_price"]
            fig, historical_data = simulate_price_history("BTC-USD", btc_price, days=30)
            plt.savefig("btc_price_history.png")
            plt.close(fig)
            print("Chart saved to btc_price_history.png")
            
            # Apply trading strategies
            print("\nApplying trading strategies to simulated data...")
            
            # Moving Average Crossover
            ma_strategy = MovingAverageCrossover(short_window=5, long_window=20)
            ma_signal = ma_strategy.generate_signal(historical_data)
            ma_results = ma_strategy.backtest(historical_data)
            
            print(f"MA Crossover Strategy Signal: {ma_signal}")
            print(f"  - Total Return: {ma_results['total_return']:.2%}")
            print(f"  - Annualized Return: {ma_results['annualized_return']:.2%}")
            print(f"  - Sharpe Ratio: {ma_results['sharpe_ratio']:.2f}")
            print(f"  - Number of Trades: {ma_results['num_trades']}")
            
            # RSI Strategy
            rsi_strategy = RelativeStrengthIndex(window=14, oversold=30, overbought=70)
            rsi_signal = rsi_strategy.generate_signal(historical_data)
            rsi_results = rsi_strategy.backtest(historical_data)
            
            print(f"RSI Strategy Signal: {rsi_signal}")
            print(f"  - Total Return: {rsi_results['total_return']:.2%}")
            print(f"  - Annualized Return: {rsi_results['annualized_return']:.2%}")
            print(f"  - Sharpe Ratio: {rsi_results['sharpe_ratio']:.2f}")
            print(f"  - Number of Trades: {rsi_results['num_trades']}")
            
    except Exception as e:
        print(f"Error analyzing prices: {e}")
    
    print("\nMarket analysis complete.")

if __name__ == "__main__":
    main()
