#!/usr/bin/env python3
"""
XRP Trading Strategy Example

This script demonstrates how to use the XRP Advanced Trading Strategy
to analyze and trade XRP on Robinhood.

Usage:
    python examples/xrp_trading_example.py
"""

import sys
import os
import logging
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from crypto_api import RobinhoodCryptoAPI
from strategies.xrp_advanced_strategy import XRPAdvancedStrategy
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("xrp_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('xrp_trading_example')

def backtest_strategy():
    """Run a backtest of the XRP trading strategy."""
    logger.info("Starting XRP trading strategy backtest")
    
    # Initialize API client (use demo mode for backtesting)
    api = RobinhoodCryptoAPI(config.API_KEY, config.BASE64_PRIVATE_KEY)
    
    # Create strategy instance using parameters from config
    strategy = XRPAdvancedStrategy(api, **config.XRP_STRATEGY_CONFIG)
    
    # Run backtest
    results = strategy.backtest()
    
    # Print results
    logger.info(f"Backtest Results:")
    logger.info(f"Total Return: {results['total_return']:.2f}%")
    logger.info(f"Win Rate: {results['win_rate']*100:.2f}%")
    logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"Number of Trades: {len(results['trades'])}")
    
    # Visualize the results
    visualize_results(results)
    
    return results

def visualize_results(results):
    """
    Visualize the backtest results.
    
    Args:
        results: Backtest results dictionary
    """
    if not results['daily_returns']:
        logger.warning("No daily returns data to visualize")
        return
    
    # Convert daily returns to DataFrame
    df = pd.DataFrame(results['daily_returns'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot price history
    ax1.plot(df.index, df['price'], label='XRP Price')
    ax1.set_title('XRP Price History')
    ax1.set_ylabel('Price (USD)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot strategy returns
    cumulative_returns = []
    cum_return = 1.0
    for daily_return in df['return']:
        if pd.notna(daily_return):
            cum_return *= (1 + daily_return)
        cumulative_returns.append(cum_return)
    
    ax2.plot(df.index, [(cr - 1) * 100 for cr in cumulative_returns], label='Strategy Returns')
    ax2.set_title('XRP Strategy Cumulative Returns')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Return (%)')
    ax2.grid(True)
    ax2.legend()
    
    # Add trades to the price chart
    if results['trades']:
        buy_dates = []
        buy_prices = []
        sell_dates = []
        sell_prices = []
        
        for trade in results['trades']:
            if trade.get('action') == 'buy':
                buy_dates.append(trade.get('date'))
                buy_prices.append(trade.get('entry_price'))
            elif trade.get('action') in ['sell', 'stop_loss', 'take_profit']:
                sell_dates.append(trade.get('date'))
                sell_prices.append(trade.get('exit_price'))
        
        # Convert to datetime if string
        buy_dates = [pd.to_datetime(d) if isinstance(d, str) else d for d in buy_dates]
        sell_dates = [pd.to_datetime(d) if isinstance(d, str) else d for d in sell_dates]
        
        ax1.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy Signal')
        ax1.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell Signal')
        ax1.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('xrp_strategy_backtest.png')
    logger.info("Saved backtest visualization to xrp_strategy_backtest.png")
    plt.close()

def live_trading_simulation():
    """
    Simulate live trading with the XRP strategy.
    This function demonstrates how the strategy would be used in a live setting,
    but doesn't actually execute trades.
    """
    logger.info("Starting XRP trading strategy live simulation")
    
    # Initialize API client
    api = RobinhoodCryptoAPI(config.API_KEY, config.BASE64_PRIVATE_KEY)
    
    # Create strategy instance
    strategy = XRPAdvancedStrategy(api, **config.XRP_STRATEGY_CONFIG)
    
    # Simulation parameters
    sim_duration = 10  # Number of iterations for the simulation
    sleep_time = 2     # Time between iterations in seconds (would be longer in real scenario)
    
    logger.info(f"Running simulation for {sim_duration} iterations")
    
    for i in range(sim_duration):
        logger.info(f"Simulation iteration {i+1}/{sim_duration}")
        
        try:
            # Generate signal
            signal = strategy.generate_signal()
            logger.info(f"Signal: {signal['signal']} at price {signal['price']:.4f}, RSI: {signal['rsi']:.2f}, MACD Hist: {signal['macd_histogram']:.6f}")
            
            # Execute trade based on signal (simulated)
            trade_result = strategy.execute_trade(signal)
            if trade_result['action'] != 'hold':
                logger.info(f"Trade executed: {trade_result}")
            
            # Check for stop loss or take profit if in position
            if strategy.current_position is not None:
                logger.info(f"Currently in position: Entry price: {strategy.entry_price:.4f}, Stop loss: {strategy.stop_loss_price:.4f}, Take profit: {strategy.take_profit_price:.4f}")
                
                # Get current price (simulated for this example)
                # In a real implementation, we would use the API to get current price
                current_price = signal['price'] * (1 + (np.random.random() - 0.5) * 0.02)  # Random price change
                
                # Update dynamic stop loss
                strategy.update_dynamic_stop_loss(current_price)
                
                # Check if stop loss or take profit triggered
                exit_signal = strategy.check_stop_loss_take_profit(current_price)
                if exit_signal is not None:
                    logger.info(f"Exit triggered: {exit_signal}")
            
            # Sleep before next iteration
            time.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
    
    logger.info("Simulation completed")

def main():
    """Main function to run the example."""
    logger.info("XRP Trading Strategy Example")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='XRP Trading Strategy Example')
    parser.add_argument('--mode', choices=['backtest', 'simulate'], default='backtest',
                       help='Mode to run the example (backtest or simulate)')
    args = parser.parse_args()
    
    if args.mode == 'backtest':
        # Run backtest
        backtest_strategy()
    elif args.mode == 'simulate':
        # Run live trading simulation
        live_trading_simulation()
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1
    
    return 0

if __name__ == "__main__":
    # Fix for matplotlib on some systems
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    
    sys.exit(main())