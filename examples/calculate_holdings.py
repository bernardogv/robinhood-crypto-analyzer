#!/usr/bin/env python3
"""
Holdings Calculator for Robinhood Crypto Analyzer

This script focuses specifically on accurately calculating cryptocurrency holdings
from your Robinhood account, addressing issues with API data.
"""

import os
import sys
import json
import traceback
from datetime import datetime
from decimal import Decimal, getcontext

# Set decimal precision for financial calculations
getcontext().prec = 10

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our modules
from crypto_api import RobinhoodCryptoAPI

# Try to import the API credentials from config.py
try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import API_KEY, BASE64_PRIVATE_KEY
except ImportError:
    print("Error: API credentials not found.")
    print("Please create a config.py file with your API credentials.")
    sys.exit(1)

def calculate_holdings_from_orders(api, asset_code):
    """
    Calculate holdings for a specific asset from order history.
    This is a more reliable method when API data shows zero quantity.
    
    Args:
        api: Robinhood Crypto API client
        asset_code: Asset code (e.g., "BTC", "ETH")
        
    Returns:
        Decimal: Calculated quantity
    """
    try:
        # Get all orders
        all_orders = api.get_orders()
        
        if "results" not in all_orders:
            print(f"No order data found for {asset_code}")
            return Decimal('0')
        
        # Filter orders for this asset
        asset_orders = []
        for order in all_orders["results"]:
            if order.get("symbol") == f"{asset_code}-USD" and order.get("state") == "filled":
                asset_orders.append(order)
        
        if not asset_orders:
            print(f"No filled orders found for {asset_code}")
            return Decimal('0')
        
        # Calculate net quantity from orders
        net_quantity = Decimal('0')
        
        for order in asset_orders:
            # Extract order details using Decimal for precision
            filled_quantity = Decimal(str(order.get("filled_asset_quantity", "0")))
            side = order.get("side")
            
            # Update net quantity based on buy/sell
            if side == "buy":
                net_quantity += filled_quantity
            elif side == "sell":
                net_quantity -= filled_quantity
            
            # Print order details for debugging
            print(f"  - {order.get('updated_at')}: {side.upper()} {filled_quantity} {asset_code}")
        
        print(f"Calculated {asset_code} quantity from {len(asset_orders)} orders: {net_quantity}")
        return net_quantity
    
    except Exception as e:
        print(f"Error calculating holdings from orders for {asset_code}: {e}")
        traceback.print_exc()
        return Decimal('0')

def get_crypto_price(api, asset_code):
    """
    Get the current price for a cryptocurrency.
    
    Args:
        api: Robinhood Crypto API client
        asset_code: Asset code (e.g., "BTC", "ETH")
        
    Returns:
        Decimal: Current price
    """
    try:
        # Get best bid/ask prices
        best_price = api.get_best_bid_ask(f"{asset_code}-USD")
        
        if "results" not in best_price or not best_price["results"]:
            print(f"No price data found for {asset_code}")
            return Decimal('0')
        
        # Calculate mid price from bid and ask
        bid = Decimal(str(best_price["results"][0].get("bid_price", "0")))
        ask = Decimal(str(best_price["results"][0].get("ask_price", "0")))
        
        if bid > 0 and ask > 0:
            mid_price = (bid + ask) / Decimal('2')
            return mid_price
        else:
            print(f"Invalid price data for {asset_code}: Bid={bid}, Ask={ask}")
            return Decimal('0')
    
    except Exception as e:
        print(f"Error getting price for {asset_code}: {e}")
        traceback.print_exc()
        return Decimal('0')

def analyze_crypto_holdings(api):
    """
    Analyze cryptocurrency holdings with fallback to order calculation.
    
    Args:
        api: Robinhood Crypto API client
        
    Returns:
        Dict: Holdings analysis
    """
    try:
        print("\nAnalyzing crypto holdings...")
        # Get holdings from API
        holdings = api.get_holdings()
        
        # Debug: Print raw holdings data
        print("\nRaw holdings data:")
        print(json.dumps(holdings, indent=2))
        
        holdings_data = {}
        
        if "results" in holdings and holdings["results"]:
            print(f"\nFound {len(holdings['results'])} holdings in API data:")
            
            for holding in holdings["results"]:
                asset_code = holding.get("asset_code", "N/A")
                api_quantity = Decimal(str(holding.get("quantity", "0")))
                
                print(f"\n{asset_code} from API: {api_quantity}")
                
                # If API shows zero, calculate from orders
                quantity = api_quantity
                if quantity == 0:
                    print(f"{asset_code} shows zero in API data. Calculating from orders...")
                    quantity = calculate_holdings_from_orders(api, asset_code)
                
                # Get current price
                current_price = get_crypto_price(api, asset_code)
                
                # Calculate value
                current_value = quantity * current_price
                
                holdings_data[asset_code] = {
                    "quantity": float(quantity),  # Convert back to float for JSON compatibility
                    "current_price": float(current_price),
                    "current_value": float(current_value)
                }
        else:
            print("No holdings found in API data or unexpected data format.")
            print("Attempting to calculate holdings from order history...")
            
            # Common crypto assets to check
            assets_to_check = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA"]
            
            for asset_code in assets_to_check:
                quantity = calculate_holdings_from_orders(api, asset_code)
                
                if quantity > 0:
                    # Get current price
                    current_price = get_crypto_price(api, asset_code)
                    
                    # Calculate value
                    current_value = quantity * current_price
                    
                    holdings_data[asset_code] = {
                        "quantity": float(quantity),
                        "current_price": float(current_price),
                        "current_value": float(current_value)
                    }
        
        # Calculate total portfolio value
        total_value = sum(data["current_value"] for data in holdings_data.values())
        
        # Print summary
        print("\nHoldings Summary:")
        for asset_code, data in holdings_data.items():
            quantity = data["quantity"]
            price = data["current_price"]
            value = data["current_value"]
            allocation = (value / total_value * 100) if total_value > 0 else 0
            
            print(f"  - {asset_code}: {quantity:.8f} @ ${price:.2f} = ${value:.2f} ({allocation:.2f}%)")
        
        print(f"\nTotal Portfolio Value: ${total_value:.2f}")
        
        return holdings_data
    
    except Exception as e:
        print(f"Error analyzing holdings: {e}")
        traceback.print_exc()
        return {}

def main():
    """Main function."""
    print("Robinhood Crypto Holdings Calculator")
    print("===================================")
    
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
    
    # Analyze holdings
    holdings_data = analyze_crypto_holdings(api)
    
    # Save holdings data to file for reference
    with open("holdings_data.json", "w") as f:
        json.dump(holdings_data, f, indent=2)
    print("\nHoldings data saved to holdings_data.json")
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
