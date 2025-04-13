#!/usr/bin/env python3
"""
Robinhood Crypto Analyzer

A tool for analyzing leading cryptocurrencies using the Robinhood API.

This script provides command-line utilities for analyzing cryptocurrency market data,
portfolio holdings, price spreads, and order history.

Usage:
    python crypto_analyzer.py [options]

Examples:
    python crypto_analyzer.py --market --symbols BTC-USD ETH-USD
    python crypto_analyzer.py --spreads
    python crypto_analyzer.py --holdings --output holdings.json
"""

import argparse
import json
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Import our modules
from crypto_api import RobinhoodCryptoAPI
import analyzers
from utils import historical_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_analyzer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Check if config.py exists, if not print instructions
try:
    from config import API_KEY, BASE64_PRIVATE_KEY, DEFAULT_SYMBOLS
except ImportError:
    logger.error("Configuration file not found or invalid.")
    logger.error("Please create a config.py file with your API credentials:")
    logger.error("API_KEY = 'your-api-key'")
    logger.error("BASE64_PRIVATE_KEY = 'your-base64-encoded-private-key'")
    logger.error("DEFAULT_SYMBOLS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'ADA-USD']")
    sys.exit(1)


class CryptoAnalyzer:
    """
    Main class for analyzing cryptocurrency data from Robinhood.
    
    This class provides methods for retrieving and analyzing account information,
    market data, price spreads, holdings, and orders.
    """
    
    def __init__(self, api_key: str, private_key: str):
        """
        Initialize the crypto analyzer.
        
        Args:
            api_key: Robinhood API key
            private_key: Base64-encoded private key
        
        Raises:
            Exception: If initialization fails
        """
        try:
            self.api = RobinhoodCryptoAPI(api_key, private_key)
            logger.info("CryptoAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CryptoAnalyzer: {e}")
            raise

    def get_account_overview(self) -> Dict[str, Any]:
        """
        Get an overview of the user's account.
        
        Returns:
            Dict[str, Any]: Account information
        """
        try:
            logger.info("Retrieving account overview")
            account_info = self.api.get_account()
            holdings = self.api.get_holdings()
            
            return {
                "account": account_info,
                "holdings": holdings
            }
        except Exception as e:
            logger.error(f"Error getting account overview: {e}")
            return {"error": str(e)}
    
    def get_market_overview(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Get an overview of the current market for specified symbols.
        
        Args:
            symbols: List of trading pair symbols (default: use DEFAULT_SYMBOLS from config)
            
        Returns:
            Dict[str, Any]: Market information
        """
        if symbols is None:
            symbols = DEFAULT_SYMBOLS
            
        try:
            logger.info(f"Retrieving market overview for {symbols}")
            trading_pairs = self.api.get_trading_pairs(*symbols)
            best_prices = self.api.get_best_bid_ask(*symbols)
            
            # Use the analyzer module for additional market analysis
            if best_prices and "results" in best_prices:
                # Extract bid-ask data for analysis
                bid_ask_data = {}
                for result in best_prices.get("results", []):
                    if "symbol" in result:
                        symbol = result["symbol"]
                        bid_ask_data[symbol] = {
                            "bid": float(result.get("bid_price", 0)),
                            "ask": float(result.get("ask_price", 0))
                        }
                
                # Analyze the price spreads
                spread_analysis = analyzers.analyze_price_spreads(bid_ask_data)
                
                return {
                    "trading_pairs": trading_pairs,
                    "best_prices": best_prices,
                    "spread_analysis": spread_analysis
                }
            else:
                return {
                    "trading_pairs": trading_pairs,
                    "best_prices": best_prices
                }
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {"error": str(e)}

    def analyze_price_spreads(self, symbols: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Analyze the price spreads for specified symbols.
        
        Args:
            symbols: List of trading pair symbols (default: use DEFAULT_SYMBOLS from config)
            
        Returns:
            Dict[str, Dict[str, float]]: Spread analysis
        """
        if symbols is None:
            symbols = DEFAULT_SYMBOLS
            
        try:
            logger.info(f"Analyzing price spreads for {symbols}")
            best_prices = self.api.get_best_bid_ask(*symbols)
            
            # Extract bid-ask data for analysis
            bid_ask_data = {}
            if best_prices and "results" in best_prices:
                for result in best_prices.get("results", []):
                    if "symbol" in result:
                        symbol = result["symbol"]
                        bid_ask_data[symbol] = {
                            "bid": float(result.get("bid_price", 0)),
                            "ask": float(result.get("ask_price", 0))
                        }
            
            # Use the analyzer module for spread analysis
            return analyzers.analyze_price_spreads(bid_ask_data)
        except Exception as e:
            logger.error(f"Error analyzing price spreads: {e}")
            return {"error": str(e)}
    
    def analyze_crypto_holdings(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze the user's crypto holdings.
        
        Returns:
            Dict[str, Dict[str, Any]]: Holdings analysis
        """
        try:
            logger.info("Analyzing crypto holdings")
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
        except Exception as e:
            logger.error(f"Error analyzing crypto holdings: {e}")
            return {"error": str(e)}

    def analyze_orders(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze the user's orders.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Orders analysis
        """
        try:
            logger.info("Analyzing orders")
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
        except Exception as e:
            logger.error(f"Error analyzing orders: {e}")
            return {"error": str(e)}


def setup_argparse() -> argparse.ArgumentParser:
    """
    Set up command-line argument parsing.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Robinhood Crypto Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n")[2]  # Use the examples from the docstring
    )
    parser.add_argument("--account", action="store_true", help="Get account overview")
    parser.add_argument("--market", action="store_true", help="Get market overview")
    parser.add_argument("--spreads", action="store_true", help="Analyze price spreads")
    parser.add_argument("--holdings", action="store_true", help="Analyze holdings")
    parser.add_argument("--orders", action="store_true", help="Analyze orders")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, 
                      help=f"Specify symbols to analyze (default: {', '.join(DEFAULT_SYMBOLS)})")
    parser.add_argument("--output", "-o", help="Output file (JSON)")
    
    return parser


def main():
    """Main function that parses arguments and runs the analyzer."""
    # Parse command-line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any([args.account, args.market, args.spreads, args.holdings, args.orders]):
        parser.print_help()
        return
    
    try:
        # Initialize the analyzer
        analyzer = CryptoAnalyzer(API_KEY, BASE64_PRIVATE_KEY)
        results = {}
        
        # Run requested analyses
        if args.account:
            logger.info("Getting account overview...")
            results["account"] = analyzer.get_account_overview()
        
        if args.market:
            logger.info(f"Getting market overview for {', '.join(args.symbols)}...")
            results["market"] = analyzer.get_market_overview(args.symbols)
        
        if args.spreads:
            logger.info(f"Analyzing price spreads for {', '.join(args.symbols)}...")
            results["spreads"] = analyzer.analyze_price_spreads(args.symbols)
        
        if args.holdings:
            logger.info("Analyzing holdings...")
            results["holdings"] = analyzer.analyze_crypto_holdings()
        
        if args.orders:
            logger.info("Analyzing orders...")
            results["orders"] = analyzer.analyze_orders()
        
        # Print results to stdout or save to file
        formatted_results = json.dumps(results, indent=2)
        
        if args.output:
            with open(args.output, "w") as f:
                f.write(formatted_results)
            logger.info(f"Results saved to {args.output}")
        else:
            print(formatted_results)
            
        return 0
    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
