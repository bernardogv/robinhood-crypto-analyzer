#!/usr/bin/env python3
"""
Basic Trading Strategies for Robinhood Crypto Analyzer

This module provides implementations of basic trading strategies
for cryptocurrency trading on Robinhood.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

class Strategy:
    """Base class for trading strategies."""
    
    def __init__(self, name: str):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
        """
        self.name = name
        self.signals = []  # List of trading signals generated
    
    def generate_signal(self, data: Any) -> str:
        """
        Generate a trading signal based on the strategy.
        
        Args:
            data: Data to analyze
            
        Returns:
            str: "buy", "sell", or "hold"
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def backtest(self, price_history: pd.DataFrame) -> Dict[str, Any]:
        """
        Backtest the strategy on historical price data.
        
        Args:
            price_history: DataFrame with historical price data
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        raise NotImplementedError("Subclasses must implement this method")


class MovingAverageCrossover(Strategy):
    """
    Moving Average Crossover Strategy
    
    Generates buy signals when the short-term moving average crosses above
    the long-term moving average, and sell signals when it crosses below.
    """
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        Initialize the strategy.
        
        Args:
            short_window: Short-term moving average window
            long_window: Long-term moving average window
        """
        super().__init__(f"MA Crossover ({short_window}, {long_window})")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signal(self, price_history: pd.DataFrame) -> str:
        """
        Generate a trading signal based on the MA crossover strategy.
        
        Args:
            price_history: DataFrame with historical price data
            
        Returns:
            str: "buy", "sell", or "hold"
        """
        if len(price_history) < self.long_window:
            return "hold"  # Not enough data
        
        # Calculate moving averages
        short_ma = price_history["Price"].rolling(window=self.short_window).mean()
        long_ma = price_history["Price"].rolling(window=self.long_window).mean()
        
        # Get the most recent values
        current_short_ma = short_ma.iloc[-1]
        current_long_ma = long_ma.iloc[-1]
        
        # Get the previous values
        prev_short_ma = short_ma.iloc[-2]
        prev_long_ma = long_ma.iloc[-2]
        
        # Check for crossover
        if prev_short_ma < prev_long_ma and current_short_ma > current_long_ma:
            signal = "buy"  # Bullish crossover
        elif prev_short_ma > prev_long_ma and current_short_ma < current_long_ma:
            signal = "sell"  # Bearish crossover
        else:
            signal = "hold"  # No crossover
        
        # Record the signal
        self.signals.append({
            "date": price_history.iloc[-1]["Date"],
            "price": price_history.iloc[-1]["Price"],
            "signal": signal,
            "short_ma": current_short_ma,
            "long_ma": current_long_ma
        })
        
        return signal
    
    def backtest(self, price_history: pd.DataFrame) -> Dict[str, Any]:
        """
        Backtest the strategy on historical price data.
        
        Args:
            price_history: DataFrame with historical price data
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        if len(price_history) < self.long_window:
            return {"error": "Not enough data for backtesting"}
        
        # Create a copy of the price history
        data = price_history.copy()
        
        # Calculate moving averages
        data["ShortMA"] = data["Price"].rolling(window=self.short_window).mean()
        data["LongMA"] = data["Price"].rolling(window=self.long_window).mean()
        
        # Initialize signal column
        data["Signal"] = 0
        
        # Generate signals
        data.loc[data["ShortMA"] > data["LongMA"], "Signal"] = 1  # Buy signal
        data.loc[data["ShortMA"] < data["LongMA"], "Signal"] = -1  # Sell signal
        
        # Create a "Position" column (1 = long, -1 = short, 0 = no position)
        data["Position"] = data["Signal"].diff()
        
        # Calculate returns
        data["Returns"] = data["Price"].pct_change()
        data["Strategy_Returns"] = data["Position"].shift(1) * data["Returns"]
        
        # Calculate cumulative returns
        data["Cumulative_Returns"] = (1 + data["Returns"]).cumprod()
        data["Cumulative_Strategy_Returns"] = (1 + data["Strategy_Returns"]).cumprod()
        
        # Calculate statistics
        total_return = data["Cumulative_Strategy_Returns"].iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(data)) - 1
        daily_returns = data["Strategy_Returns"].dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        
        # Count trades
        trades = data["Position"].dropna()
        trades = trades[trades != 0]
        num_trades = len(trades)
        
        # Get buy and sell signals for plotting
        buy_signals = data[data["Position"] == 1]
        sell_signals = data[data["Position"] == -1]
        
        return {
            "data": data,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "num_trades": num_trades,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio
        }


class RelativeStrengthIndex(Strategy):
    """
    Relative Strength Index (RSI) Strategy
    
    Generates buy signals when RSI is below the oversold threshold,
    and sell signals when RSI is above the overbought threshold.
    """
    
    def __init__(self, window: int = 14, oversold: int = 30, overbought: int = 70):
        """
        Initialize the strategy.
        
        Args:
            window: RSI calculation window
            oversold: Oversold threshold
            overbought: Overbought threshold
        """
        super().__init__(f"RSI ({window}, {oversold}, {overbought})")
        self.window = window
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, price_history: pd.DataFrame) -> pd.Series:
        """
        Calculate the Relative Strength Index.
        
        Args:
            price_history: DataFrame with historical price data
            
        Returns:
            pd.Series: RSI values
        """
        # Calculate price changes
        delta = price_history["Price"].diff()
        
        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=self.window).mean()
        avg_loss = loss.rolling(window=self.window).mean()
        
        # Calculate relative strength
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signal(self, price_history: pd.DataFrame) -> str:
        """
        Generate a trading signal based on the RSI strategy.
        
        Args:
            price_history: DataFrame with historical price data
            
        Returns:
            str: "buy", "sell", or "hold"
        """
        if len(price_history) <= self.window:
            return "hold"  # Not enough data
        
        # Calculate RSI
        rsi = self.calculate_rsi(price_history)
        
        # Get the most recent RSI value
        current_rsi = rsi.iloc[-1]
        
        # Generate signal
        if current_rsi <= self.oversold:
            signal = "buy"  # Oversold condition
        elif current_rsi >= self.overbought:
            signal = "sell"  # Overbought condition
        else:
            signal = "hold"  # No clear signal
        
        # Record the signal
        self.signals.append({
            "date": price_history.iloc[-1]["Date"],
            "price": price_history.iloc[-1]["Price"],
            "signal": signal,
            "rsi": current_rsi
        })
        
        return signal
    
    def backtest(self, price_history: pd.DataFrame) -> Dict[str, Any]:
        """
        Backtest the strategy on historical price data.
        
        Args:
            price_history: DataFrame with historical price data
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        if len(price_history) <= self.window:
            return {"error": "Not enough data for backtesting"}
        
        # Create a copy of the price history
        data = price_history.copy()
        
        # Calculate RSI
        data["RSI"] = self.calculate_rsi(data)
        
        # Initialize signal and position columns
        data["Signal"] = 0  # 0 = no signal
        data.loc[data["RSI"] <= self.oversold, "Signal"] = 1  # Buy signal
        data.loc[data["RSI"] >= self.overbought, "Signal"] = -1  # Sell signal
        
        # Create a "Position" column
        data["Position"] = 0
        
        # Implement a state machine for positions
        position = 0  # 0 = no position, 1 = long position, -1 = short position
        
        for i in range(len(data)):
            signal = data.iloc[i]["Signal"]
            
            if signal == 1 and position <= 0:  # Buy signal and not long
                position = 1
            elif signal == -1 and position >= 0:  # Sell signal and not short
                position = -1
            
            data.iloc[i, data.columns.get_loc("Position")] = position
        
        # Calculate returns
        data["Returns"] = data["Price"].pct_change()
        data["Strategy_Returns"] = data["Position"].shift(1) * data["Returns"]
        
        # Calculate cumulative returns
        data["Cumulative_Returns"] = (1 + data["Returns"]).cumprod()
        data["Cumulative_Strategy_Returns"] = (1 + data["Strategy_Returns"]).cumprod()
        
        # Calculate statistics
        total_return = data["Cumulative_Strategy_Returns"].iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(data)) - 1
        daily_returns = data["Strategy_Returns"].dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        
        # Count trades
        position_changes = data["Position"].diff()
        num_trades = len(position_changes[position_changes != 0])
        
        # Get buy and sell signals for plotting
        buy_signals = data[data["Signal"] == 1]
        sell_signals = data[data["Signal"] == -1]
        
        return {
            "data": data,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "num_trades": num_trades,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio
        }


class SpreadTradingStrategy(Strategy):
    """
    Spread Trading Strategy
    
    A strategy that aims to profit from the bid-ask spread differences
    among different cryptocurrencies.
    """
    
    def __init__(self, spread_threshold: float = 0.5):
        """
        Initialize the strategy.
        
        Args:
            spread_threshold: Minimum spread percentage to trigger a signal
        """
        super().__init__(f"Spread Trading ({spread_threshold}%)")
        self.spread_threshold = spread_threshold
    
    def generate_signal(self, spread_data: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """
        Generate trading signals based on the spread trading strategy.
        
        Args:
            spread_data: Dictionary mapping symbols to spread data
            
        Returns:
            Dict[str, str]: Trading signals for each symbol
        """
        signals = {}
        
        for symbol, data in spread_data.items():
            spread_percent = data.get("spread_percent", 0)
            
            if spread_percent >= self.spread_threshold:
                signals[symbol] = "avoid"  # Spread too high, avoid trading
            else:
                signals[symbol] = "consider"  # Spread acceptable, consider trading
            
            # Record the signal
            self.signals.append({
                "date": datetime.now(),
                "symbol": symbol,
                "signal": signals[symbol],
                "spread_percent": spread_percent
            })
        
        return signals
    
    def rank_trading_opportunities(self, spread_data: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Rank trading opportunities based on spread data.
        
        Args:
            spread_data: Dictionary mapping symbols to spread data
            
        Returns:
            List[Dict[str, Any]]: Ranked trading opportunities
        """
        opportunities = []
        
        for symbol, data in spread_data.items():
            spread_percent = data.get("spread_percent", 0)
            bid = data.get("bid", 0)
            ask = data.get("ask", 0)
            
            opportunities.append({
                "symbol": symbol,
                "spread_percent": spread_percent,
                "bid": bid,
                "ask": ask,
                "score": 1 / (spread_percent + 0.1)  # Higher score for lower spread
            })
        
        # Sort by score (descending)
        opportunities.sort(key=lambda x: x["score"], reverse=True)
        
        return opportunities
    
    def backtest(self, spread_history: List[Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
        """
        Backtest the strategy on historical spread data.
        
        Args:
            spread_history: List of spread data snapshots over time
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        # Implementation of backtest logic for spread trading
        # This would require historical spread data and a more complex simulation
        
        return {
            "message": "Spread trading backtest would require historical spread data over time",
            "spread_threshold": self.spread_threshold
        }
