#!/usr/bin/env python3
"""
XRP Live Trading Example

This example demonstrates how to run the XRP live trading strategy
with real-time data from Robinhood.
"""

import argparse
import sys
import json
from datetime import datetime
import time

from crypto_api import RobinhoodCryptoAPI
from strategies.xrp_live_trading_strategy import XRPLiveTradingStrategy

# Check if config.py exists, if not print instructions
try:
    from config import API_KEY, BASE64_PRIVATE_KEY
except ImportError:
    print("Error: Configuration file not found or invalid.")
    print("Please create a config.py file with your API credentials:")
    print("API_KEY = 'your-api-key'")
    print("BASE64_PRIVATE_KEY = 'your-base64-encoded-private-key'")
    sys.exit(1)

def main():
    """Main function to run XRP live trading."""
    parser = argparse.ArgumentParser(description="XRP Live Trading Example")
    parser.add_argument("--hours", type=float, default=24.0,
                        help="Number of hours to run the trading strategy")
    parser.add_argument("--mode", choices=["live", "paper", "backtest"], default="paper",
                        help="Trading mode: live for real trading, paper for simulation, backtest for historical testing")
    parser.add_argument("--auto", action="store_true",
                        help="Enable automatic trading")
    parser.add_argument("--output", "-o", help="Output file for trade history (JSON)")
    parser.add_argument("--interval", type=int, default=300,
                        help="Check interval in seconds (default: 300)")
    parser.add_argument("--max-position", type=float, default=0.1,
                        help="Maximum position size as fraction of available funds (default: 0.1)")
    parser.add_argument("--stop-loss", type=float, default=0.05,
                        help="Stop loss percentage (default: 0.05)")
    parser.add_argument("--take-profit", type=float, default=0.15,
                        help="Take profit percentage (default: 0.15)")
    
    args = parser.parse_args()
    
    # Initialize the API client
    api = RobinhoodCryptoAPI(API_KEY, BASE64_PRIVATE_KEY)
    
    # Create the strategy instance
    strategy = XRPLiveTradingStrategy(
        api=api,
        max_position_size=args.max_position,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        check_interval=args.interval,
        auto_trade=args.auto,
        paper_trading=(args.mode == "paper")
    )
    
    print(f"XRP Live Trading Strategy initialized")
    print(f"Mode: {args.mode}")
    print(f"Auto-trading: {'Enabled' if args.auto else 'Disabled'}")
    print(f"Running for: {args.hours} hours")
    print(f"Maximum position size: {args.max_position * 100:.1f}% of funds")
    print(f"Stop loss: {args.stop_loss * 100:.1f}%")
    print(f"Take profit: {args.take_profit * 100:.1f}%")
    print()
    
    # Warning for live trading
    if args.mode == "live" and args.auto:
        print("WARNING: You are about to start LIVE TRADING with real money!")
        print("Press Ctrl+C within 10 seconds to cancel...")
        try:
            for i in range(10, 0, -1):
                print(f"Starting in {i} seconds...", end="\r")
                time.sleep(1)
            print("Starting live trading now...        ")
        except KeyboardInterrupt:
            print("\nLive trading canceled by user.")
            return
    
    try:
        if args.mode in ["live", "paper"]:
            # Run live trading
            start_time = datetime.now()
            print(f"Starting at: {start_time}")
            
            strategy.run_live_trading(duration_hours=args.hours)
            
            end_time = datetime.now()
            print(f"Completed at: {end_time}")
            print(f"Total runtime: {(end_time - start_time).total_seconds() / 3600:.2f} hours")
            
        elif args.mode == "backtest":
            # Run backtest
            print("Running backtest...")
            backtest_results = strategy.backtest()
            
            print(f"XRP Trading Strategy Backtest Results")
            print(f"Total Return: {backtest_results['total_return']:.2f}%")
            print(f"Win Rate: {backtest_results['win_rate']*100:.2f}%")
            print(f"Profit Factor: {backtest_results['profit_factor']:.2f}")
            print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
            print(f"Number of Trades: {len(backtest_results['trades'])}")
        
        # Save trade history if requested
        if args.output:
            strategy.save_trade_history(args.output)
            print(f"Trade history saved to {args.output}")
        
    except KeyboardInterrupt:
        print("\nTrading stopped by user.")
    except Exception as e:
        print(f"Error: {str(e)}")
    
if __name__ == "__main__":
    main()
