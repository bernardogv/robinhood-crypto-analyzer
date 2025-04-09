#!/usr/bin/env python3
"""
Portfolio Analysis Example for Robinhood Crypto Analyzer

This script demonstrates how to use the Robinhood Crypto Analyzer
to perform portfolio analysis.
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
from visualizers.price_charts import plot_portfolio_distribution

# Try to import the API credentials from config.py
try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import API_KEY, BASE64_PRIVATE_KEY
except ImportError:
    print("Error: API credentials not found.")
    print("Please create a config.py file with your API credentials.")
    sys.exit(1)

def main():
    """Main function."""
    print("Robinhood Crypto Portfolio Analysis Example")
    print("=========================================")
    
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
    
    # Get holdings
    print("\nGetting crypto holdings...")
    try:
        holdings = api.get_holdings()
        
        if "results" in holdings and holdings["results"]:
            print(f"Found {len(holdings['results'])} holdings:")
            
            # Prepare data for detailed analysis
            holdings_data = {}
            
            for holding in holdings["results"]:
                asset_code = holding.get("asset_code", "N/A")
                quantity = float(holding.get("quantity", 0))
                
                # Get current price for this asset
                best_price = api.get_best_bid_ask(f"{asset_code}-USD")
                
                current_price = 0
                if "results" in best_price and best_price["results"]:
                    bid = float(best_price["results"][0].get("bid_price", 0))
                    ask = float(best_price["results"][0].get("ask_price", 0))
                    current_price = (bid + ask) / 2
                
                current_value = quantity * current_price
                
                holdings_data[asset_code] = {
                    "quantity": quantity,
                    "current_price": current_price,
                    "current_value": current_value
                }
                
                print(f"  - {asset_code}: {quantity} @ ${current_price:.2f} = ${current_value:.2f}")
            
            # Calculate total portfolio value
            total_value = sum(data["current_value"] for data in holdings_data.values())
            print(f"\nTotal Portfolio Value: ${total_value:.2f}")
            
            # Calculate portfolio allocation
            print("\nPortfolio Allocation:")
            for asset_code, data in holdings_data.items():
                allocation = (data["current_value"] / total_value) * 100 if total_value > 0 else 0
                print(f"  - {asset_code}: ${data['current_value']:.2f} ({allocation:.2f}%)")
            
            # Visualize portfolio distribution
            print("\nGenerating portfolio distribution chart...")
            fig = plot_portfolio_distribution(holdings_data)
            plt.savefig("portfolio_distribution.png")
            plt.close(fig)
            print("Chart saved to portfolio_distribution.png")
            
            # Get recent orders
            print("\nGetting recent orders...")
            orders = api.get_orders()
            
            if "results" in orders and orders["results"]:
                print(f"Found {len(orders['results'])} orders:")
                
                # Group orders by state
                order_states = {}
                for order in orders["results"]:
                    state = order.get("state", "unknown")
                    if state not in order_states:
                        order_states[state] = []
                    order_states[state].append(order)
                
                # Print order summary
                for state, state_orders in order_states.items():
                    print(f"  - {state.capitalize()}: {len(state_orders)}")
                
                # Print recent filled orders
                if "filled" in order_states:
                    print("\nRecent filled orders:")
                    for order in sorted(order_states["filled"], key=lambda x: x.get("updated_at", ""), reverse=True)[:5]:
                        symbol = order.get("symbol", "N/A")
                        side = order.get("side", "N/A")
                        filled_quantity = order.get("filled_asset_quantity", "N/A")
                        average_price = order.get("average_price", "N/A")
                        date = order.get("updated_at", "N/A")
                        
                        print(f"  - {date}: {side.upper()} {filled_quantity} {symbol} @ ${average_price}")
            else:
                print("No orders found.")
        else:
            print("No holdings found.")
    except Exception as e:
        print(f"Error analyzing portfolio: {e}")
    
    print("\nPortfolio analysis complete.")

if __name__ == "__main__":
    main()
