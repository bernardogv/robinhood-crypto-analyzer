#!/usr/bin/env python3
"""
XRP Trading Visualization

This script visualizes the results of the XRP trading strategy,
including price charts, trading signals, and performance metrics.
"""

import argparse
import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

from crypto_api import RobinhoodCryptoAPI
from strategies.xrp_live_trading_strategy import XRPLiveTradingStrategy
from utils.xrp_market_analysis import XRPMarketAnalysis

# Check if config.py exists, if not print instructions
try:
    from config import API_KEY, BASE64_PRIVATE_KEY
except ImportError:
    print("Error: Configuration file not found or invalid.")
    print("Please create a config.py file with your API credentials:")
    print("API_KEY = 'your-api-key'")
    print("BASE64_PRIVATE_KEY = 'your-base64-encoded-private-key'")
    sys.exit(1)

def plot_trading_results(history_file: str, output_dir: str = "charts"):
    """
    Plot trading results from a history file.
    
    Args:
        history_file: Path to trade history JSON file
        output_dir: Directory to save charts
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load trade history
        with open(history_file, 'r') as f:
            trade_history = json.load(f)
        
        if not trade_history:
            print("No trades found in history file.")
            return
            
        # Convert trade times to datetime
        for trade in trade_history:
            if 'time' in trade:
                trade['time'] = datetime.fromisoformat(trade['time'])
                
        # Get market data for the trading period
        market_analyzer = XRPMarketAnalysis()
        start_time = min([trade['time'] for trade in trade_history if 'time' in trade])
        end_time = max([trade['time'] for trade in trade_history if 'time' in trade])
        
        days = (end_time - start_time).days + 1
        price_data = market_analyzer.get_xrp_price_data(days=max(30, days))
        
        # Filter price data to the trading period
        price_data = price_data[(price_data['timestamp'] >= start_time) & 
                                (price_data['timestamp'] <= end_time + timedelta(days=1))]
        
        # Plot price chart with trades
        plt.figure(figsize=(14, 8))
        
        # Plot price
        plt.plot(price_data['timestamp'], price_data['close'], color='blue', alpha=0.7, label='XRP Price')
        
        # Plot buy trades
        buy_trades = [t for t in trade_history if t.get('action') == 'buy']
        for trade in buy_trades:
            if 'price' in trade and 'time' in trade:
                plt.scatter(trade['time'], trade['price'], color='green', marker='^', s=100, alpha=0.8)
                plt.annotate('Buy', (trade['time'], trade['price']), 
                             xytext=(0, 10), textcoords='offset points', 
                             ha='center', va='bottom', color='green')
        
        # Plot sell trades
        sell_trades = [t for t in trade_history if t.get('action') in ['sell', 'stop_loss', 'take_profit']]
        for trade in sell_trades:
            if 'price' in trade and 'time' in trade:
                color = 'red'
                if trade.get('action') == 'take_profit':
                    color = 'purple'
                elif trade.get('action') == 'stop_loss':
                    color = 'orange'
                    
                plt.scatter(trade['time'], trade['price'], color=color, marker='v', s=100, alpha=0.8)
                plt.annotate(trade.get('action', 'Sell'), (trade['time'], trade['price']), 
                             xytext=(0, -15), textcoords='offset points', 
                             ha='center', va='top', color=color)
        
        # Format chart
        plt.title('XRP Trading Strategy Results', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days//10)))
        plt.gcf().autofmt_xdate()
        
        # Save chart
        price_chart_path = os.path.join(output_dir, 'xrp_trading_price_chart.png')
        plt.savefig(price_chart_path, dpi=300, bbox_inches='tight')
        print(f"Price chart saved to {price_chart_path}")
        plt.close()
        
        # Plot performance chart
        plt.figure(figsize=(14, 8))
        
        # Calculate cumulative returns
        cumulative_return = 0
        returns = []
        timestamps = []
        
        for trade in trade_history:
            if trade.get('action') in ['sell', 'stop_loss', 'take_profit'] and 'profit_loss' in trade:
                cumulative_return += trade['profit_loss']
                returns.append(cumulative_return)
                timestamps.append(trade['time'])
        
        if returns:
            plt.plot(timestamps, returns, color='green', marker='o', linestyle='-', alpha=0.8)
            
            # Add annotations for significant trades
            for i, trade in enumerate([t for t in trade_history if t.get('action') in ['sell', 'stop_loss', 'take_profit'] and 'profit_loss' in t]):
                if abs(trade['profit_loss']) > 10 or i == len(returns) - 1:  # Significant trade or last trade
                    plt.annotate(f"${trade['profit_loss']:.2f}", 
                                 (trade['time'], returns[i]), 
                                 xytext=(10, 10 if trade['profit_loss'] > 0 else -10),
                                 textcoords='offset points',
                                 ha='left', va='center')
            
            # Format chart
            plt.title('XRP Trading Strategy Cumulative Profit/Loss', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Cumulative P&L (USD)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add horizontal line at 0
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days//10)))
            plt.gcf().autofmt_xdate()
            
            # Save chart
            performance_chart_path = os.path.join(output_dir, 'xrp_trading_performance.png')
            plt.savefig(performance_chart_path, dpi=300, bbox_inches='tight')
            print(f"Performance chart saved to {performance_chart_path}")
            plt.close()
        else:
            print("No completed trades found for performance chart.")
        
        # Calculate and display statistics
        print("\nTrading Statistics:")
        
        total_trades = len([t for t in trade_history if t.get('action') in ['buy', 'sell', 'stop_loss', 'take_profit']])
        buy_count = len([t for t in trade_history if t.get('action') == 'buy'])
        sell_count = len([t for t in trade_history if t.get('action') in ['sell', 'stop_loss', 'take_profit']])
        
        print(f"Total trades: {total_trades}")
        print(f"Buy trades: {buy_count}")
        print(f"Sell trades: {sell_count}")
        
        if sell_count > 0:
            # Calculate profit/loss statistics
            profits = [t.get('profit_loss', 0) for t in trade_history if t.get('action') in ['sell', 'stop_loss', 'take_profit'] and 'profit_loss' in t]
            profit_pcts = [t.get('profit_loss_pct', 0) for t in trade_history if t.get('action') in ['sell', 'stop_loss', 'take_profit'] and 'profit_loss_pct' in t]
            
            total_profit = sum(profits)
            avg_profit_pct = np.mean(profit_pcts) if profit_pcts else 0
            
            win_trades = [p for p in profit_pcts if p > 0]
            loss_trades = [p for p in profit_pcts if p <= 0]
            
            win_rate = len(win_trades) / len(profit_pcts) if profit_pcts else 0
            
            print(f"Total profit/loss: ${total_profit:.2f}")
            print(f"Average profit/loss: {avg_profit_pct:.2f}%")
            print(f"Win rate: {win_rate:.2%}")
            print(f"Best trade: {max(profit_pcts) if profit_pcts else 0:.2f}%")
            print(f"Worst trade: {min(profit_pcts) if profit_pcts else 0:.2f}%")
            
            # Plot profit distribution
            if profit_pcts:
                plt.figure(figsize=(10, 6))
                plt.hist(profit_pcts, bins=10, alpha=0.7, color='blue', edgecolor='black')
                
                plt.title('XRP Trading Profit/Loss Distribution', fontsize=16)
                plt.xlabel('Profit/Loss (%)', fontsize=12)
                plt.ylabel('Number of Trades', fontsize=12)
                plt.grid(True, alpha=0.3)
                
                # Add vertical line at 0
                plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                
                # Save chart
                dist_chart_path = os.path.join(output_dir, 'xrp_trading_profit_distribution.png')
                plt.savefig(dist_chart_path, dpi=300, bbox_inches='tight')
                print(f"Profit distribution chart saved to {dist_chart_path}")
                plt.close()
        else:
            print("No completed trades found for statistics.")
    
    except Exception as e:
        print(f"Error plotting trading results: {str(e)}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="XRP Trading Visualization")
    parser.add_argument("--history", required=True, help="Path to trade history JSON file")
    parser.add_argument("--output", default="charts", help="Directory to save charts")
    
    args = parser.parse_args()
    
    plot_trading_results(args.history, args.output)

if __name__ == "__main__":
    main()
