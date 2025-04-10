#!/usr/bin/env python3
"""
XRP Trading Strategy for Robinhood Crypto Analyzer

This module implements an advanced trading strategy specifically optimized for
XRP trading based on current market conditions and technical analysis.

The strategy combines:
1. Multiple indicator signals including RSI, Bollinger Bands, and MACD
2. Volatility-based position sizing
3. Dynamic stop-loss mechanism
4. XRP-specific market sentiment integration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import requests
from typing import Dict, List, Any, Optional, Tuple

from crypto_api import RobinhoodCryptoAPI

class XRPAdvancedStrategy:
    """
    Advanced strategy for trading XRP using a combination of technical indicators,
    volatility-based position sizing, and sentiment analysis.
    """
    
    def __init__(
        self,
        api: RobinhoodCryptoAPI,
        symbol: str = "XRP-USD",
        rsi_window: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        bb_window: int = 20,
        bb_std: float = 2.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        volatility_window: int = 20,
        max_position_size: float = 0.1,  # 10% of available funds
        stop_loss_pct: float = 0.05,     # 5% stop loss
        take_profit_pct: float = 0.15,   # 15% take profit
        sentiment_weight: float = 0.2    # 20% weight for sentiment
    ):
        """
        Initialize the XRP trading strategy.
        
        Args:
            api: RobinhoodCryptoAPI instance
            symbol: Trading pair symbol
            rsi_window: RSI calculation window
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            bb_window: Bollinger Bands window
            bb_std: Bollinger Bands standard deviation
            macd_fast: MACD fast EMA period
            macd_slow: MACD slow EMA period
            macd_signal: MACD signal line period
            volatility_window: Window for volatility calculation
            max_position_size: Maximum position size as percentage of funds
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            sentiment_weight: Weight of sentiment in trading decision
        """
        self.api = api
        self.symbol = symbol
        self.name = f"XRP Advanced Strategy ({rsi_window}, {bb_window}, {macd_fast}:{macd_slow})"
        
        # Strategy parameters
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.volatility_window = volatility_window
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.sentiment_weight = sentiment_weight
        
        # Trading state
        self.signals = []
        self.current_position = None
        self.last_signal = None
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None
        
        # Market data cache
        self.price_history = None
        self.last_update = None
        self.fear_greed_index = None

    def get_price_history(self, days: int = 30) -> pd.DataFrame:
        """
        Get historical price data for XRP.
        
        Args:
            days: Number of days of historical data
            
        Returns:
            DataFrame with historical price data
        """
        # In a real implementation, we would use the Robinhood API
        # to fetch historical price data. For this example, we'll
        # simulate fetching data and create a placeholder.
        
        if self.price_history is not None and self.last_update is not None:
            time_diff = datetime.now() - self.last_update
            if time_diff.total_seconds() < 3600:  # Cache for 1 hour
                return self.price_history
        
        # Get current price info
        current_price_data = self.api.get_best_bid_ask([self.symbol])
        
        # In a real implementation, we would fetch historical data
        # For this example, we'll create a simulated dataframe
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        # Simulate realistic XRP price data based on current conditions
        prices = np.linspace(2.0, float(current_price_data['bid_price']), len(dates))
        # Add some random noise to simulate volatility
        prices = prices + np.random.normal(0, 0.05, len(dates))
        
        # Create DataFrame
        self.price_history = pd.DataFrame({
            'Date': dates,
            'Price': prices,
            'Volume': np.random.uniform(50000000, 200000000, len(dates))
        })
        
        self.last_update = datetime.now()
        return self.price_history

    def calculate_indicators(self, price_history: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the price history.
        
        Args:
            price_history: DataFrame with price history
            
        Returns:
            DataFrame with added technical indicators
        """
        df = price_history.copy()
        
        # Calculate RSI
        delta = df['Price'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=self.rsi_window).mean()
        avg_loss = loss.rolling(window=self.rsi_window).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        df['SMA'] = df['Price'].rolling(window=self.bb_window).mean()
        df['STD'] = df['Price'].rolling(window=self.bb_window).std()
        df['UpperBand'] = df['SMA'] + (df['STD'] * self.bb_std)
        df['LowerBand'] = df['SMA'] - (df['STD'] * self.bb_std)
        df['BandWidth'] = (df['UpperBand'] - df['LowerBand']) / df['SMA']
        
        # Calculate MACD
        df['EMA_Fast'] = df['Price'].ewm(span=self.macd_fast, adjust=False).mean()
        df['EMA_Slow'] = df['Price'].ewm(span=self.macd_slow, adjust=False).mean()
        df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']
        df['Signal_Line'] = df['MACD'].ewm(span=self.macd_signal, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        
        # Calculate volatility (using ATR-like measure)
        df['Volatility'] = df['Price'].rolling(window=self.volatility_window).std() / df['Price']
        
        return df

    def get_market_sentiment(self) -> float:
        """
        Get market sentiment for XRP from external sources.
        
        Returns:
            Sentiment score from -1 (extremely bearish) to 1 (extremely bullish)
        """
        # In a real implementation, we would fetch sentiment data from APIs
        # For this example, we'll use a simulated fear and greed index value
        
        # Cache sentiment for 4 hours
        if self.fear_greed_index is not None and self.last_update is not None:
            time_diff = datetime.now() - self.last_update
            if time_diff.total_seconds() < 14400:  # 4 hours
                return self.fear_greed_index
        
        # Simulate sentiment: 0 (extreme fear) to 100 (extreme greed)
        # Currently markets show somewhat neutral sentiment
        self.fear_greed_index = np.random.normal(45, 10)
        self.fear_greed_index = max(0, min(100, self.fear_greed_index))
        
        # Normalize to -1 to 1 scale
        normalized_sentiment = (self.fear_greed_index - 50) / 50
        
        return normalized_sentiment

    def calculate_position_size(self, volatility: float) -> float:
        """
        Calculate position size based on volatility.
        
        Args:
            volatility: Current price volatility
            
        Returns:
            Position size as percentage of available funds
        """
        # Inverse relationship with volatility - smaller positions in high volatility
        position_size = self.max_position_size * (1 - min(volatility * 5, 0.8))
        return max(0.01, position_size)  # Minimum 1% position

    def generate_signal(self, price_history: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate a trading signal based on the strategy.
        
        Args:
            price_history: Optional price history dataframe
            
        Returns:
            Trading signal with details
        """
        if price_history is None:
            price_history = self.get_price_history()
        
        # Calculate indicators
        indicators = self.calculate_indicators(price_history)
        current_data = indicators.iloc[-1]
        
        # Get market sentiment
        sentiment = self.get_market_sentiment()
        
        # Determine indicator signals
        # RSI signal
        if current_data['RSI'] <= self.rsi_oversold:
            rsi_signal = 1  # Bullish
        elif current_data['RSI'] >= self.rsi_overbought:
            rsi_signal = -1  # Bearish
        else:
            rsi_signal = 0  # Neutral
        
        # Bollinger Bands signal
        if current_data['Price'] <= current_data['LowerBand']:
            bb_signal = 1  # Bullish
        elif current_data['Price'] >= current_data['UpperBand']:
            bb_signal = -1  # Bearish
        else:
            bb_signal = 0  # Neutral
        
        # MACD signal
        if current_data['MACD'] > current_data['Signal_Line'] and current_data['MACD_Histogram'] > 0:
            macd_signal = 1  # Bullish
        elif current_data['MACD'] < current_data['Signal_Line'] and current_data['MACD_Histogram'] < 0:
            macd_signal = -1  # Bearish
        else:
            macd_signal = 0  # Neutral
        
        # Combine signals with weights
        # RSI: 30%, BB: 20%, MACD: 30%, Sentiment: 20%
        combined_signal = (
            0.3 * rsi_signal +
            0.2 * bb_signal +
            0.3 * macd_signal +
            self.sentiment_weight * sentiment
        )
        
        # Determine trading action
        if combined_signal >= 0.4:
            signal = "buy"
        elif combined_signal <= -0.4:
            signal = "sell"
        else:
            signal = "hold"
        
        # Calculate position size based on volatility
        position_size = self.calculate_position_size(current_data['Volatility'])
        
        # Record the signal
        signal_data = {
            "date": datetime.now(),
            "price": current_data['Price'],
            "signal": signal,
            "rsi": current_data['RSI'],
            "bb_width": current_data['BandWidth'],
            "macd_histogram": current_data['MACD_Histogram'],
            "volatility": current_data['Volatility'],
            "sentiment": sentiment,
            "combined_score": combined_signal,
            "position_size": position_size
        }
        self.signals.append(signal_data)
        self.last_signal = signal_data
        
        return signal_data

    def execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trade based on the signal.
        
        Args:
            signal: Trading signal
            
        Returns:
            Trade execution details
        """
        if signal['signal'] == "buy" and self.current_position is None:
            # Calculate quantity based on position size and available funds
            # In a real implementation, we would get account balance
            available_funds = 10000  # Example value
            
            investment_amount = available_funds * signal['position_size']
            quantity = investment_amount / signal['price']
            
            # Execute buy order
            # In a real implementation, we would use the API to place an order
            # order = self.api.place_market_order(self.symbol, "buy", str(quantity))
            
            # Set entry price, stop loss, and take profit
            self.entry_price = signal['price']
            self.stop_loss_price = self.entry_price * (1 - self.stop_loss_pct)
            self.take_profit_price = self.entry_price * (1 + self.take_profit_pct)
            
            self.current_position = {
                "entry_time": datetime.now(),
                "entry_price": self.entry_price,
                "quantity": quantity,
                "stop_loss": self.stop_loss_price,
                "take_profit": self.take_profit_price
            }
            
            return {
                "action": "buy",
                "entry_price": self.entry_price,
                "quantity": quantity,
                "stop_loss": self.stop_loss_price,
                "take_profit": self.take_profit_price,
                "investment": investment_amount
            }
            
        elif signal['signal'] == "sell" and self.current_position is not None:
            # Execute sell order
            # In a real implementation, we would use the API to place an order
            # order = self.api.place_market_order(self.symbol, "sell", str(self.current_position["quantity"]))
            
            exit_price = signal['price']
            profit_loss = (exit_price - self.entry_price) * self.current_position["quantity"]
            profit_loss_pct = (exit_price / self.entry_price - 1) * 100
            
            result = {
                "action": "sell",
                "entry_price": self.entry_price,
                "exit_price": exit_price,
                "quantity": self.current_position["quantity"],
                "profit_loss": profit_loss,
                "profit_loss_pct": profit_loss_pct,
                "holding_time": (datetime.now() - self.current_position["entry_time"]).total_seconds() / 3600  # hours
            }
            
            # Reset position
            self.current_position = None
            self.entry_price = None
            self.stop_loss_price = None
            self.take_profit_price = None
            
            return result
            
        return {"action": "hold"}

    def check_stop_loss_take_profit(self, current_price: float) -> Optional[Dict[str, Any]]:
        """
        Check if stop loss or take profit has been triggered.
        
        Args:
            current_price: Current XRP price
            
        Returns:
            Trade execution details if triggered, None otherwise
        """
        if self.current_position is None:
            return None
        
        if current_price <= self.stop_loss_price:
            # Stop loss triggered
            # In a real implementation, we would use the API to place a sell order
            # order = self.api.place_market_order(self.symbol, "sell", str(self.current_position["quantity"]))
            
            profit_loss = (current_price - self.entry_price) * self.current_position["quantity"]
            profit_loss_pct = (current_price / self.entry_price - 1) * 100
            
            result = {
                "action": "stop_loss",
                "entry_price": self.entry_price,
                "exit_price": current_price,
                "quantity": self.current_position["quantity"],
                "profit_loss": profit_loss,
                "profit_loss_pct": profit_loss_pct,
                "holding_time": (datetime.now() - self.current_position["entry_time"]).total_seconds() / 3600  # hours
            }
            
            # Reset position
            self.current_position = None
            self.entry_price = None
            self.stop_loss_price = None
            self.take_profit_price = None
            
            return result
            
        elif current_price >= self.take_profit_price:
            # Take profit triggered
            # In a real implementation, we would use the API to place a sell order
            # order = self.api.place_market_order(self.symbol, "sell", str(self.current_position["quantity"]))
            
            profit_loss = (current_price - self.entry_price) * self.current_position["quantity"]
            profit_loss_pct = (current_price / self.entry_price - 1) * 100
            
            result = {
                "action": "take_profit",
                "entry_price": self.entry_price,
                "exit_price": current_price,
                "quantity": self.current_position["quantity"],
                "profit_loss": profit_loss,
                "profit_loss_pct": profit_loss_pct,
                "holding_time": (datetime.now() - self.current_position["entry_time"]).total_seconds() / 3600  # hours
            }
            
            # Reset position
            self.current_position = None
            self.entry_price = None
            self.stop_loss_price = None
            self.take_profit_price = None
            
            return result
            
        return None

    def update_dynamic_stop_loss(self, current_price: float) -> None:
        """
        Update stop loss dynamically based on price movement (trailing stop).
        
        Args:
            current_price: Current XRP price
        """
        if self.current_position is None or self.stop_loss_price is None:
            return
        
        # Calculate the minimum stop loss (original stop loss)
        min_stop_loss = self.entry_price * (1 - self.stop_loss_pct)
        
        # Calculate the new potential stop loss (trailing)
        # If price has moved 5% in our favor, move stop loss to break even
        if current_price >= self.entry_price * 1.05:
            new_stop_loss = max(self.entry_price, current_price * 0.95)
        else:
            new_stop_loss = min_stop_loss
        
        # Update stop loss if new one is higher
        if new_stop_loss > self.stop_loss_price:
            self.stop_loss_price = new_stop_loss

    def run_strategy(self, days_to_run: int = 30) -> Dict[str, Any]:
        """
        Run the strategy over a period of time.
        
        Args:
            days_to_run: Number of days to simulate
            
        Returns:
            Strategy performance metrics
        """
        # In a real implementation, this would run in a loop checking for signals
        # For this simulation, we'll generate signals and execute trades for historical data
        
        price_history = self.get_price_history(days=days_to_run + 30)  # Get extra history for indicators
        
        # Slice the dataframe to the required period
        start_index = len(price_history) - days_to_run if len(price_history) > days_to_run else 0
        simulation_data = price_history.iloc[start_index:].copy()
        
        trades = []
        daily_returns = []
        
        self.current_position = None
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None
        
        for i in range(len(simulation_data)):
            date = simulation_data.iloc[i]['Date']
            current_price = simulation_data.iloc[i]['Price']
            
            # Check if we need to exit based on stop loss or take profit
            if self.current_position is not None:
                self.update_dynamic_stop_loss(current_price)
                exit_signal = self.check_stop_loss_take_profit(current_price)
                if exit_signal is not None:
                    exit_signal['date'] = date
                    trades.append(exit_signal)
            
            # Generate signal based on indicators using data up to current day
            historical_data = price_history.iloc[:start_index + i + 1]
            signal = self.generate_signal(historical_data)
            signal['date'] = date
            
            # Execute trade based on signal
            trade_result = self.execute_trade(signal)
            trade_result['date'] = date
            
            if trade_result['action'] != 'hold':
                trades.append(trade_result)
            
            # Calculate daily return
            if self.current_position is not None:
                daily_return = (current_price / simulation_data.iloc[i-1]['Price'] - 1) if i > 0 else 0
            else:
                daily_return = 0
            
            daily_returns.append({
                'date': date,
                'price': current_price,
                'return': daily_return,
                'position': 'long' if self.current_position is not None else 'none'
            })
        
        # Calculate performance metrics
        if len(trades) > 0:
            win_trades = [t for t in trades if t.get('profit_loss_pct', 0) > 0]
            loss_trades = [t for t in trades if t.get('profit_loss_pct', 0) <= 0]
            
            win_rate = len(win_trades) / len(trades) if len(trades) > 0 else 0
            avg_profit = np.mean([t.get('profit_loss_pct', 0) for t in win_trades]) if len(win_trades) > 0 else 0
            avg_loss = np.mean([t.get('profit_loss_pct', 0) for t in loss_trades]) if len(loss_trades) > 0 else 0
            profit_factor = abs(sum([t.get('profit_loss', 0) for t in win_trades]) / sum([t.get('profit_loss', 0) for t in loss_trades])) if sum([t.get('profit_loss', 0) for t in loss_trades]) != 0 else float('inf')
            
            # Calculate total return
            total_return = 1.0
            for dr in daily_returns:
                if dr['position'] == 'long':
                    total_return *= (1 + dr['return'])
            
            total_return = (total_return - 1) * 100  # Convert to percentage
            
            # Calculate Sharpe Ratio
            returns_series = [dr['return'] for dr in daily_returns if dr['position'] == 'long']
            sharpe_ratio = np.sqrt(252) * np.mean(returns_series) / np.std(returns_series) if len(returns_series) > 0 and np.std(returns_series) > 0 else 0
            
            return {
                'trades': trades,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'daily_returns': daily_returns
            }
        else:
            return {
                'trades': [],
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'daily_returns': daily_returns
            }

    def backtest(self, price_history: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Backtest the strategy on historical price data.
        
        Args:
            price_history: Optional price history dataframe
            
        Returns:
            Backtest results
        """
        if price_history is None:
            price_history = self.get_price_history(days=60)  # Default to 60 days for backtest
        
        # Run the strategy over the historical data
        return self.run_strategy(days_to_run=len(price_history))


def main():
    """
    Main function to demonstrate the XRP trading strategy.
    """
    # In a real implementation, you would configure this with your API keys
    api_key = "YOUR_API_KEY"
    base64_private_key = "YOUR_BASE64_PRIVATE_KEY"
    
    api = RobinhoodCryptoAPI(api_key, base64_private_key)
    
    # Create the XRP trading strategy
    strategy = XRPAdvancedStrategy(api)
    
    # Run backtest
    backtest_results = strategy.backtest()
    
    # Print results
    print(f"XRP Trading Strategy Backtest Results")
    print(f"Total Return: {backtest_results['total_return']:.2f}%")
    print(f"Win Rate: {backtest_results['win_rate']*100:.2f}%")
    print(f"Profit Factor: {backtest_results['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"Number of Trades: {len(backtest_results['trades'])}")
    
    # In a real implementation, you might want to visualize the results
    # or save them to a file for further analysis


if __name__ == "__main__":
    main()
