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
import numpy as np
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

# Enhanced XRP strategy configuration with more aggressive parameters
ENHANCED_XRP_STRATEGY_CONFIG = {
    "symbol": "XRP-USD",
    "rsi_window": 14,
    "rsi_oversold": 40,      # Increased from 30 to be more sensitive
    "rsi_overbought": 60,    # Decreased from 70 to be more sensitive
    "bb_window": 15,         # Decreased from 20 for faster response
    "bb_std": 1.8,          # Decreased from 2.0 to tighten bands
    "macd_fast": 8,          # Decreased from 12 for faster response
    "macd_slow": 20,         # Decreased from 26 for faster response
    "macd_signal": 9,
    "volatility_window": 15, # Decreased from 20
    "max_position_size": 0.1,
    "stop_loss_pct": 0.04,   # Tightened from 0.05
    "take_profit_pct": 0.12, # Decreased from 0.15 for more frequent profits
    "sentiment_weight": 0.3  # Increased from 0.2 for more influence
}

def generate_mock_price_data(days=60, volatility_factor=1.5):
    """
    Generate mock price data for XRP for testing without API connection.
    
    Args:
        days: Number of days of data to generate
        volatility_factor: Multiplier for price volatility
        
    Returns:
        DataFrame with mock price history
    """
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate realistic XRP price data based on current trends
    # Start with a base price around current XRP value
    base_price = 2.20
    
    # Add a trend (combination of upward and downward movements)
    t = np.linspace(0, 1, len(dates))
    trend = 0.2 * np.sin(2 * np.pi * t) + 0.3 * t
    
    # Add some cyclical patterns (sine waves of different frequencies)
    cycle1 = 0.25 * np.sin(np.linspace(0, 6 * np.pi, len(dates)))  # Increased amplitude
    cycle2 = 0.15 * np.sin(np.linspace(0, 15 * np.pi, len(dates)))  # Increased amplitude and frequency
    
    # Add some random volatility (increased)
    volatility = np.random.normal(0, 0.08 * volatility_factor, len(dates))
    
    # Combine components
    prices = base_price + trend + cycle1 + cycle2 + volatility
    
    # Ensure prices are positive
    prices = np.maximum(prices, 0.5)
    
    # Generate realistic volumes with higher volatility
    volumes = np.random.uniform(80000000, 250000000, len(dates))
    volumes = volumes * (1 + 0.7 * np.sin(np.linspace(0, 8 * np.pi, len(dates))))  # Add cyclical pattern to volume
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Volume': volumes
    })
    
    return df

class MockXRPStrategy(XRPAdvancedStrategy):
    """
    A mock implementation of XRPAdvancedStrategy that overrides API-dependent methods
    to work with simulated data.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mock_price_data = None
        self._mock_sentiment = None
    
    def set_mock_data(self, price_data, sentiment=None):
        """Set mock price data and sentiment."""
        self._mock_price_data = price_data
        self._mock_sentiment = sentiment
    
    def get_price_history(self, days=30):
        """Override to use mock data instead of API calls."""
        if self._mock_price_data is None:
            self._mock_price_data = generate_mock_price_data(days=90, volatility_factor=1.5)
        
        return self._mock_price_data.iloc[-days:]
    
    def get_market_sentiment(self):
        """Override to use mock sentiment data."""
        if self._mock_sentiment is not None:
            return self._mock_sentiment
        
        # Generate synthetic sentiment data that correlates somewhat with price changes
        recent_data = self._mock_price_data.iloc[-14:]
        price_change = (recent_data['Price'].iloc[-1] / recent_data['Price'].iloc[0]) - 1
        
        # Base sentiment on recent price change plus random noise
        sentiment_base = np.tanh(price_change * 3)  # Scale and bound between -1 and 1
        noise = np.random.normal(0, 0.3)  # Add some randomness
        sentiment = sentiment_base + noise
        sentiment = max(-1, min(1, sentiment))  # Ensure between -1 and 1
        
        return sentiment
    
    def calculate_indicators(self, price_history):
        """Enhanced indicator calculation with more signal generation."""
        df = super().calculate_indicators(price_history)
        
        # Add some additional indicators that might help with signal generation
        
        # Add Rate of Change (ROC)
        df['ROC'] = price_history['Price'].pct_change(5) * 100
        
        # Add Average Directional Index (ADX) - simplified version
        plus_dm = price_history['Price'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm = price_history['Price'].diff()
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        tr = pd.DataFrame()
        tr['h-l'] = price_history['Price'].diff().abs()  # Using price diff as a simple approximation
        tr['h-c'] = price_history['Price'].shift(1).diff().abs()
        tr['l-c'] = price_history['Price'].shift(1).diff().abs()
        tr['tr'] = tr.max(axis=1)
        
        plus_di = 100 * plus_dm.rolling(window=14).mean() / tr['tr'].rolling(window=14).mean()
        minus_di = 100 * minus_dm.rolling(window=14).mean() / tr['tr'].rolling(window=14).mean()
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.rolling(window=14).mean()
        
        return df

def backtest_strategy():
    """Run a backtest of the XRP trading strategy with enhanced parameters."""
    logger.info("Starting XRP trading strategy backtest")
    
    # Initialize API client (use demo mode for backtesting)
    api = RobinhoodCryptoAPI(config.API_KEY, config.BASE64_PRIVATE_KEY)
    
    # Create strategy instance with enhanced parameters
    logger.info(f"Using enhanced strategy configuration: {ENHANCED_XRP_STRATEGY_CONFIG}")
    strategy = MockXRPStrategy(api, **ENHANCED_XRP_STRATEGY_CONFIG)
    
    # Generate mock price data with increased volatility
    mock_price_data = generate_mock_price_data(days=90, volatility_factor=1.8)
    strategy.set_mock_data(mock_price_data)
    
    # Run backtest
    results = strategy.backtest()
    
    # Print results
    logger.info(f"Backtest Results:")
    logger.info(f"Total Return: {results['total_return']:.2f}%")
    logger.info(f"Win Rate: {results['win_rate']*100:.2f}%")
    logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"Number of Trades: {len(results['trades'])}")
    
    if len(results['trades']) > 0:
        trade_summaries = []
        for i, trade in enumerate(results['trades'], 1):
            if 'profit_loss_pct' in trade:
                trade_summaries.append(
                    f"Trade {i}: {trade['action']} at {trade['exit_price']:.4f}, PnL: {trade['profit_loss_pct']:.2f}%"
                )
            else:
                trade_summaries.append(
                    f"Trade {i}: {trade['action']} at {trade.get('entry_price', 0):.4f}"
                )
        
        logger.info("Trade Summary:")
        for summary in trade_summaries:
            logger.info(summary)
    
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
    
    # Create strategy instance with enhanced parameters
    logger.info(f"Using enhanced strategy configuration: {ENHANCED_XRP_STRATEGY_CONFIG}")
    strategy = MockXRPStrategy(api, **ENHANCED_XRP_STRATEGY_CONFIG)
    
    # Generate mock price data for the simulation
    mock_prices = generate_mock_price_data(days=30, volatility_factor=1.8)
    strategy.set_mock_data(mock_prices)
    
    # Simulation parameters
    sim_duration = 10  # Number of iterations for the simulation
    sleep_time = 2     # Time between iterations in seconds
    
    logger.info(f"Running simulation for {sim_duration} iterations")
    
    # Generate an initial price (latest price from our mock data)
    current_price = mock_prices.iloc[-1]['Price']
    last_price = current_price
    
    for i in range(sim_duration):
        logger.info(f"Simulation iteration {i+1}/{sim_duration}")
        
        try:
            # Generate a slightly updated price with random movement
            # Current price moves up or down by up to 3%
            price_change = (np.random.random() - 0.5) * 0.06  # -3% to +3%
            current_price = last_price * (1 + price_change)
            last_price = current_price
            
            # Create a signal manually to simulate what would come from the strategy
            signal = {
                'date': datetime.now(),
                'price': current_price,
                'signal': np.random.choice(['buy', 'sell', 'hold'], p=[0.3, 0.3, 0.4]),  # More active trading
                'rsi': np.random.uniform(20, 80),
                'bb_width': np.random.uniform(0.02, 0.06),
                'macd_histogram': np.random.uniform(-0.02, 0.02),
                'volatility': np.random.uniform(0.01, 0.05),
                'sentiment': np.random.uniform(-0.5, 0.5),
                'combined_score': np.random.uniform(-1, 1),
                'position_size': np.random.uniform(0.05, 0.1)
            }
            
            logger.info(f"Signal: {signal['signal']} at price {signal['price']:.4f}, RSI: {signal['rsi']:.2f}, MACD Hist: {signal['macd_histogram']:.6f}")
            
            # Execute trade based on signal (simulated)
            trade_result = strategy.execute_trade(signal)
            if trade_result['action'] != 'hold':
                logger.info(f"Trade executed: {trade_result}")
            
            # Check for stop loss or take profit if in position
            if strategy.current_position is not None:
                logger.info(f"Currently in position: Entry price: {strategy.entry_price:.4f}, Stop loss: {strategy.stop_loss_price:.4f}, Take profit: {strategy.take_profit_price:.4f}")
                
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
    
    sys.exit(main())