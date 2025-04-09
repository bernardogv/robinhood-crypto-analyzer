#!/usr/bin/env python3
"""
Robinhood Crypto Analyzer

A tool for analyzing leading cryptocurrencies using the Robinhood API.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import our modules
from crypto_api import RobinhoodCryptoAPI

# Check if config.py exists, if not print instructions
try:
    from config import API_KEY, BASE64_PRIVATE_KEY
except ImportError:
    print("Error: Configuration file not found or invalid.")
    print("Please create a config.py file with your API credentials:")
    print("API_KEY = 'your-api-key'")
    print("BASE64_PRIVATE_KEY = 'your-base64-encoded-private-key'")
    sys.exit(1)

# Constants
DEFAULT_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "ADA-USD"]

class CryptoAnalyzer:
    """
    Main class for analyzing cryptocurrency data from Robinhood.
    """
    
    def __init__(self, api_key: str, private_key: str):
        """
        Initialize the crypto analyzer.
        
        Args:
            api_key: Robinhood API key
            private_key: Base64-encoded private key
        """
        self.api = RobinhoodCryptoAPI(api_key, private_key)
        
    def get_account_overview(self) -> Dict[str, Any]:
        """
        Get an overview of the user's account.
        
        Returns:
            Account information
        """
        account_info = self.api.get_account()
        holdings = self.api.get_holdings()
        
        return {
            "account": account_info,
            "holdings": holdings
        }
    
    def get_market_overview(self, symbols: List[str] = DEFAULT_SYMBOLS) -> Dict[str, Any]:
        """
        Get an overview of the current market for specified symbols.
        
        Args:
            symbols: List of trading pair symbols
            
        Returns:
            Market information
        """
        trading_pairs = self.api.get_trading_pairs(*symbols)
        best_prices = self.api.get_best_bid_ask(*symbols)
        
        return {
            "trading_pairs": trading_pairs,
            "best_prices": best_prices
        }
    
    def analyze_price_spreads(self, symbols: List[str] = DEFAULT_SYMBOLS) -> Dict[str, Dict[str, float]]:
        """
        Analyze the price spreads for specified symbols.
        
        Args:
            symbols: List of trading pair symbols
            
        Returns:
            Spread analysis
        """
        best_prices = self.api.get_best_bid_ask(*symbols)
        
        spreads = {}
        if best_prices and "results" in best_prices:
            for result in best_prices["results"]:
                if "symbol" in result:
                    symbol = result["symbol"]
                    bid = float(result.get("bid_price", 0))
                    ask = float(result.get("ask_price", 0))
                    
                    # Calculate spread
                    if bid > 0 and ask > 0:
                        spread_amount = ask - bid
                        spread_percent = (spread_amount / bid) * 100
                        mid_price = (bid + ask) / 2
                        
                        spreads[symbol] = {
                            "bid": bid,
                            "ask": ask,
                            "mid_price": mid_price,
                            "spread_amount": spread_amount,
                            "spread_percent": spread_percent
                        }
        
        return spreads
    
    def analyze_crypto_holdings(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze the user's crypto holdings.
        
        Returns:
            Holdings analysis
        """
        holdings_data = self.api.get_holdings()
        holdings_analysis = {}
        
        if holdings_data and "results" in holdings_data:
            for holding in holdings_data["results"]:
                if "asset_code" in holding:
                    asset_code = holding["asset_code"]
                    quantity = float(holding.get("quantity", 0))
                    
                    # Get current price data for this asset
                    best_price = self.api.get_best_bid_ask(f"{asset_code}-USD")
                    
                    current_price = 0
                    if best_price and "results" in best_price and best_price["results"]:
                        # Use mid price for valuation
                        bid = float(best_price["results"][0].get("bid_price", 0))
                        ask = float(best_price["results"][0].get("ask_price", 0))
                        current_price = (bid + ask) / 2
                    
                    # Calculate current value
                    current_value = quantity * current_price
                    
                    holdings_analysis[asset_code] = {
                        "quantity": quantity,
                        "current_price": current_price,
                        "current_value": current_value,
                        "original_data": holding
                    }
        
        return holdings_analysis
    
    def analyze_orders(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze the user's orders.
        
        Returns:
            Orders analysis
        """
        orders_data = self.api.get_orders()
        
        orders_analysis = {
            "open_orders": [],
            "filled_orders": [],
            "canceled_orders": [],
            "other_orders": []
        }
        
        if orders_data and "results" in orders_data:
            for order in orders_data["results"]:
                state = order.get("state", "unknown")
                
                if state == "open":
                    orders_analysis["open_orders"].append(order)
                elif state == "filled":
                    orders_analysis["filled_orders"].append(order)
                elif state == "canceled":
                    orders_analysis["canceled_orders"].append(order)
                else:
                    orders_analysis["other_orders"].append(order)
        
        return orders_analysis

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Robinhood Crypto Analyzer")
    parser.add_argument("--account", action="store_true", help="Get account overview")
    parser.add_argument("--market", action="store_true", help="Get market overview")
    parser.add_argument("--spreads", action="store_true", help="Analyze price spreads")
    parser.add_argument("--holdings", action="store_true", help="Analyze holdings")
    parser.add_argument("--orders", action="store_true", help="Analyze orders")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, 
                        help=f"Specify symbols to analyze (default: {', '.join(DEFAULT_SYMBOLS)})")
    parser.add_argument("--output", "-o", help="Output file (JSON)")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any([args.account, args.market, args.spreads, args.holdings, args.orders]):
        parser.print_help()
        return
    
    analyzer = CryptoAnalyzer(API_KEY, BASE64_PRIVATE_KEY)
    results = {}
    
    # Run requested analyses
    if args.account:
        print("Getting account overview...")
        results["account"] = analyzer.get_account_overview()
    
    if args.market:
        print(f"Getting market overview for {', '.join(args.symbols)}...")
        results["market"] = analyzer.get_market_overview(args.symbols)
    
    if args.spreads:
        print(f"Analyzing price spreads for {', '.join(args.symbols)}...")
        results["spreads"] = analyzer.analyze_price_spreads(args.symbols)
    
    if args.holdings:
        print("Analyzing holdings...")
        results["holdings"] = analyzer.analyze_crypto_holdings()
    
    if args.orders:
        print("Analyzing orders...")
        results["orders"] = analyzer.analyze_orders()
    
    # Print results to stdout or save to file
    formatted_results = json.dumps(results, indent=2)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(formatted_results)
        print(f"Results saved to {args.output}")
    else:
        print(formatted_results)

if __name__ == "__main__":
    main()
