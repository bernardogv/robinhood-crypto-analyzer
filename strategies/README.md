# Trading Strategies

This directory contains implementations of various cryptocurrency trading strategies for use with the Robinhood Crypto Analyzer.

## Strategy Types

### Basic Strategies

Located in `basic_strategies.py`:

1. **Moving Average Crossover**
   - Uses two moving averages (short-term and long-term) to generate buy/sell signals
   - Buy signal: Short-term MA crosses above long-term MA
   - Sell signal: Short-term MA crosses below long-term MA
   - Customizable parameters: short window, long window

2. **Relative Strength Index (RSI)**
   - Uses the RSI indicator to identify overbought/oversold conditions
   - Buy signal: RSI falls below the oversold threshold
   - Sell signal: RSI rises above the overbought threshold
   - Customizable parameters: window period, oversold threshold, overbought threshold

3. **Spread Trading**
   - Analyzes bid-ask spreads across different cryptocurrencies to identify arbitrage opportunities
   - Signals to trade or avoid based on spread percentage
   - Customizable parameters: spread threshold

### Advanced Strategies

Located in `xrp_advanced_strategy.py`:

1. **XRP Advanced Trading Strategy**
   - Multi-indicator approach combining RSI, Bollinger Bands, and MACD
   - Volatility-based position sizing
   - Dynamic stop-loss mechanism (trailing stops)
   - Market sentiment integration
   - Comprehensive backtesting capabilities

## Usage

### Basic Usage

```python
from strategies import MovingAverageCrossover, RelativeStrengthIndex, SpreadTradingStrategy

# Initialize a strategy
ma_strategy = MovingAverageCrossover(short_window=20, long_window=50)

# Generate a signal based on price history
signal = ma_strategy.generate_signal(price_history_dataframe)
print(f"Trading signal: {signal}")  # Output: "buy", "sell", or "hold"

# Backtest the strategy
backtest_results = ma_strategy.backtest(price_history_dataframe)
```

### Using the XRP Advanced Strategy

```python
from crypto_api import RobinhoodCryptoAPI
from strategies import XRPAdvancedStrategy
from config import API_KEY, BASE64_PRIVATE_KEY, XRP_STRATEGY_CONFIG

# Initialize API client
api = RobinhoodCryptoAPI(API_KEY, BASE64_PRIVATE_KEY)

# Initialize the XRP strategy with configuration from config.py
strategy = XRPAdvancedStrategy(api, **XRP_STRATEGY_CONFIG)

# Generate a signal
signal = strategy.generate_signal()
print(f"XRP signal: {signal['signal']}")

# Backtest the strategy
backtest_results = strategy.backtest()
print(f"Total return: {backtest_results['total_return']:.2f}%")
print(f"Win rate: {backtest_results['win_rate']*100:.2f}%")
```

## Customizing Strategies

### Modifying Strategy Parameters

All strategies are designed to be easily customizable by modifying the parameters in `config.py`:

```python
# In config.py
STRATEGY_SETTINGS = {
    "ma_crossover": {
        "short_window": 20,
        "long_window": 50
    },
    "rsi": {
        "window": 14,
        "oversold": 30,
        "overbought": 70
    },
    # Other strategy settings...
}
```

### Creating a Custom Strategy

To implement a custom strategy:

1. Subclass the `Strategy` base class
2. Implement the `generate_signal()` method to produce trading signals
3. Implement the `backtest()` method to evaluate strategy performance

Example:

```python
from strategies import Strategy
import pandas as pd

class MyCustomStrategy(Strategy):
    def __init__(self, param1=10, param2=20):
        super().__init__("My Custom Strategy")
        self.param1 = param1
        self.param2 = param2
    
    def generate_signal(self, price_history):
        # Implement your signal generation logic
        # ...
        return signal  # "buy", "sell", or "hold"
    
    def backtest(self, price_history):
        # Implement your backtesting logic
        # ...
        return results
```

## Performance Metrics

The backtesting framework calculates the following performance metrics:

- **Total Return**: Overall percentage return of the strategy
- **Annualized Return**: Return normalized to a yearly basis
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Win Rate**: Percentage of winning trades
- **Average Profit**: Average percentage gain on winning trades
- **Average Loss**: Average percentage loss on losing trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Number of Trades**: Total number of trades executed

## Risk Management

The strategies implement various risk management techniques:

1. **Position Sizing**: Limit exposure to a percentage of available capital
2. **Stop Loss**: Automatically exit positions at predefined loss levels
3. **Take Profit**: Automatically exit positions at predefined profit levels
4. **Trailing Stops**: Dynamically adjust stop loss levels as price moves in favor
5. **Volatility-Based Position Sizing**: Reduce position size in high-volatility periods
