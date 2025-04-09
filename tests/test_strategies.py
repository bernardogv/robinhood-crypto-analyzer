#!/usr/bin/env python3
"""
Tests for the trading strategies
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import strategies
from strategies.basic_strategies import MovingAverageCrossover, RelativeStrengthIndex, SpreadTradingStrategy

class TestTradingStrategies(unittest.TestCase):
    """Test cases for trading strategies"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample price history data
        self.dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
        
        # Create a typical price pattern with some clear crossovers
        prices = []
        for i in range(100):
            # Create a price pattern: rising then falling then rising again
            if i < 30:
                price = 100 + i * 2  # Rising trend
            elif i < 60:
                price = 160 - (i - 30) * 2  # Falling trend
            else:
                price = 100 + (i - 60) * 2  # Rising trend again
            
            # Add some noise
            noise = np.random.normal(0, 5)
            prices.append(price + noise)
        
        self.price_history = pd.DataFrame({
            'Date': self.dates,
            'Price': prices
        })
        
        # Create sample spread data
        self.spread_data = {
            'BTC-USD': {
                'bid': 60000.0,
                'ask': 60300.0,
                'mid_price': 60150.0,
                'spread_amount': 300.0,
                'spread_percent': 0.5
            },
            'ETH-USD': {
                'bid': 3000.0,
                'ask': 3060.0,
                'mid_price': 3030.0,
                'spread_amount': 60.0,
                'spread_percent': 2.0
            },
            'SOL-USD': {
                'bid': 150.0,
                'ask': 151.5,
                'mid_price': 150.75,
                'spread_amount': 1.5,
                'spread_percent': 1.0
            },
            'XRP-USD': {
                'bid': 1.80,
                'ask': 1.86,
                'mid_price': 1.83,
                'spread_amount': 0.06,
                'spread_percent': 3.3
            }
        }
    
    def test_moving_average_crossover_signals(self):
        """Test MA crossover strategy signal generation"""
        # Create strategy with short window of 5 and long window of 20
        ma_strategy = MovingAverageCrossover(short_window=5, long_window=20)
        
        # Generate signal at the beginning (should be "hold" due to insufficient data)
        signal = ma_strategy.generate_signal(self.price_history.iloc[:10])
        self.assertEqual(signal, "hold", "Signal should be 'hold' with insufficient data")
        
        # Generate signals for the full dataset
        signal = ma_strategy.generate_signal(self.price_history)
        self.assertIn(signal, ["buy", "sell", "hold"], "Invalid signal returned")
        
        # Make sure signals were recorded
        self.assertGreater(len(ma_strategy.signals), 0, "No signals recorded")
        
        # Test backtest method
        results = ma_strategy.backtest(self.price_history)
        
        # Check if backtest results contain required keys
        required_keys = ["data", "buy_signals", "sell_signals", "num_trades", 
                        "total_return", "annualized_return", "sharpe_ratio"]
        for key in required_keys:
            self.assertIn(key, results, f"Backtest results missing key: {key}")
    
    def test_relative_strength_index_signals(self):
        """Test RSI strategy signal generation"""
        # Create RSI strategy
        rsi_strategy = RelativeStrengthIndex(window=14, oversold=30, overbought=70)
        
        # Generate signal at the beginning (should be "hold" due to insufficient data)
        signal = rsi_strategy.generate_signal(self.price_history.iloc[:10])
        self.assertEqual(signal, "hold", "Signal should be 'hold' with insufficient data")
        
        # Generate signals for the full dataset
        signal = rsi_strategy.generate_signal(self.price_history)
        self.assertIn(signal, ["buy", "sell", "hold"], "Invalid signal returned")
        
        # Make sure signals were recorded
        self.assertGreater(len(rsi_strategy.signals), 0, "No signals recorded")
        
        # Check RSI calculation
        rsi = rsi_strategy.calculate_rsi(self.price_history)
        self.assertGreater(len(rsi), 0, "RSI calculation failed")
        
        # Verify RSI values are in expected range
        self.assertTrue(all(0 <= value <= 100 for value in rsi.dropna()), 
                       "RSI values outside expected range [0, 100]")
        
        # Test backtest method
        results = rsi_strategy.backtest(self.price_history)
        
        # Check if backtest results contain required keys
        required_keys = ["data", "buy_signals", "sell_signals", "num_trades", 
                        "total_return", "annualized_return", "sharpe_ratio"]
        for key in required_keys:
            self.assertIn(key, results, f"Backtest results missing key: {key}")
    
    def test_spread_trading_strategy(self):
        """Test spread trading strategy"""
        # Create spread trading strategy
        spread_strategy = SpreadTradingStrategy(spread_threshold=2.0)
        
        # Generate signals
        signals = spread_strategy.generate_signal(self.spread_data)
        
        # Verify all symbols have signals
        for symbol in self.spread_data.keys():
            self.assertIn(symbol, signals, f"No signal generated for {symbol}")
            self.assertIn(signals[symbol], ["avoid", "consider"], f"Invalid signal for {symbol}")
        
        # Check signals based on spread threshold
        # XRP and ETH should be "avoid" due to spread > 2.0%
        self.assertEqual(signals["XRP-USD"], "avoid", "XRP should be 'avoid' due to high spread")
        self.assertEqual(signals["ETH-USD"], "avoid", "ETH should be 'avoid' due to high spread")
        
        # BTC and SOL should be "consider" due to spread < 2.0%
        self.assertEqual(signals["BTC-USD"], "consider", "BTC should be 'consider' due to low spread")
        self.assertEqual(signals["SOL-USD"], "consider", "SOL should be 'consider' due to low spread")
        
        # Test ranking of opportunities
        opportunities = spread_strategy.rank_trading_opportunities(self.spread_data)
        
        # Verify opportunities are ranked correctly (lower spread = higher score)
        self.assertEqual(opportunities[0]["symbol"], "BTC-USD", "BTC should be ranked highest (lowest spread)")
        self.assertEqual(opportunities[-1]["symbol"], "XRP-USD", "XRP should be ranked lowest (highest spread)")

    def test_strategy_performance_metrics(self):
        """Test calculation of strategy performance metrics"""
        # Create strategy with short window of 5 and long window of 20
        ma_strategy = MovingAverageCrossover(short_window=5, long_window=20)
        
        # Perform backtest
        results = ma_strategy.backtest(self.price_history)
        
        # Check that performance metrics are calculated and within reasonable ranges
        self.assertIsInstance(results["total_return"], float)
        self.assertIsInstance(results["annualized_return"], float)
        self.assertIsInstance(results["sharpe_ratio"], float)
        
        # Create RSI strategy and backtest
        rsi_strategy = RelativeStrengthIndex()
        rsi_results = rsi_strategy.backtest(self.price_history)
        
        # Compare performance metrics between strategies
        self.assertIsNotNone(results["total_return"])
        self.assertIsNotNone(rsi_results["total_return"])

    def test_strategy_edge_cases(self):
        """Test strategies with edge case inputs"""
        # Test with empty dataframe
        empty_data = pd.DataFrame(columns=["Date", "Price"])
        
        # Test MA strategy with empty data
        ma_strategy = MovingAverageCrossover()
        ma_signal = ma_strategy.generate_signal(empty_data)
        self.assertEqual(ma_signal, "hold", "Should return 'hold' for empty data")
        
        # Test RSI strategy with empty data
        rsi_strategy = RelativeStrengthIndex()
        rsi_signal = rsi_strategy.generate_signal(empty_data)
        self.assertEqual(rsi_signal, "hold", "Should return 'hold' for empty data")
        
        # Test with very short history (less than window)
        short_data = self.price_history.iloc[:5]
        
        # Test MA strategy with short data
        ma_signal = ma_strategy.generate_signal(short_data)
        self.assertEqual(ma_signal, "hold", "Should return 'hold' for short data")
        
        # Test RSI strategy with short data
        rsi_signal = rsi_strategy.generate_signal(short_data)
        self.assertEqual(rsi_signal, "hold", "Should return 'hold' for short data")
        
        # Test spread strategy with empty data
        spread_strategy = SpreadTradingStrategy()
        signals = spread_strategy.generate_signal({})
        self.assertEqual(signals, {}, "Should return empty dict for empty spread data")

    def test_backtest_results_format(self):
        """Test format of backtest results"""
        ma_strategy = MovingAverageCrossover(short_window=5, long_window=20)
        results = ma_strategy.backtest(self.price_history)
        
        # Check that the results dataframe has the expected columns
        expected_columns = ["Price", "ShortMA", "LongMA", "Signal", "Position", 
                           "Returns", "Strategy_Returns", "Cumulative_Returns", 
                           "Cumulative_Strategy_Returns"]
        for column in expected_columns:
            self.assertIn(column, results["data"].columns, f"Missing column: {column}")
        
        # Verify that buy signals dataframe has rows if there were crossovers
        if not results["buy_signals"].empty:
            self.assertEqual(results["buy_signals"]["Signal"].iloc[0], 1, 
                            "Buy signals should have Signal = 1")
        
        # Verify that sell signals dataframe has rows if there were crossovers
        if not results["sell_signals"].empty:
            self.assertEqual(results["sell_signals"]["Signal"].iloc[0], -1, 
                            "Sell signals should have Signal = -1")

if __name__ == '__main__':
    unittest.main()
