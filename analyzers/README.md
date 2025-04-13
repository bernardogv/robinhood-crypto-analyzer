# Market Analyzers

This directory contains modules for analyzing cryptocurrency market data, including price trends, spread analysis, and identifying trading opportunities.

## Modules Overview

### Market Analyzer (`market_analyzer.py`)

The Market Analyzer module provides functions for analyzing cryptocurrency price data, spreads, and identifying potential trading opportunities.

#### Key Functions:

1. **`analyze_price_trends()`**
   - Analyzes price trends using moving averages and momentum indicators
   - Identifies bullish, bearish, or neutral market conditions
   - Calculates metrics like volatility and momentum

2. **`analyze_price_spreads()`**
   - Analyzes bid-ask spreads for trading pairs
   - Calculates spread amount, percentage, and mid-price
   - Useful for identifying liquid markets and potential arbitrage opportunities

3. **`identify_market_opportunities()`**
   - Identifies potential trading opportunities based on spread and volatility
   - Ranks opportunities based on a scoring system
   - Filters opportunities based on minimum spread and maximum volatility thresholds

## Usage

### Basic Usage

```python
import pandas as pd
from analyzers import analyze_price_trends, analyze_price_spreads, identify_market_opportunities

# Analyze price trends
price_history = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=100),
    'Price': [100 + i + (i*i*0.01) for i in range(100)]
})
trend_analysis = analyze_price_trends(price_history, window_short=5, window_long=20)
print(f"Current trend: {trend_analysis['trend']}")

# Analyze price spreads
bid_ask_data = {
    "BTC-USD": {"bid": 60000.0, "ask": 60050.0},
    "ETH-USD": {"bid": 3000.0, "ask": 3010.0},
    "XRP-USD": {"bid": 0.50, "ask": 0.51}
}
spread_analysis = analyze_price_spreads(bid_ask_data)
for symbol, data in spread_analysis.items():
    print(f"{symbol}: Spread = {data['spread_percent']:.2f}%")

# Identify trading opportunities
volatility_data = {
    "BTC-USD": 0.02,
    "ETH-USD": 0.03,
    "XRP-USD": 0.04
}
opportunities = identify_market_opportunities(
    spread_analysis, 
    volatility_data,
    min_spread_percent=0.1,
    max_volatility=0.05
)
print("Trading opportunities:")
for opp in opportunities:
    print(f"{opp['symbol']}: Score = {opp['score']:.2f}")
```

### Integration with Robinhood API

The analyzers are designed to work seamlessly with the Robinhood Crypto API client:

```python
from crypto_api import RobinhoodCryptoAPI
from analyzers import analyze_price_spreads
from config import API_KEY, BASE64_PRIVATE_KEY

# Initialize API client
api = RobinhoodCryptoAPI(API_KEY, BASE64_PRIVATE_KEY)

# Get market data
symbols = ["BTC-USD", "ETH-USD", "XRP-USD"]
best_prices = api.get_best_bid_ask(*symbols)

# Extract bid-ask data
bid_ask_data = {}
if best_prices and "results" in best_prices:
    for result in best_prices.get("results", []):
        if "symbol" in result:
            symbol = result["symbol"]
            bid_ask_data[symbol] = {
                "bid": float(result.get("bid_price", 0)),
                "ask": float(result.get("ask_price", 0))
            }

# Analyze spread data
spread_analysis = analyze_price_spreads(bid_ask_data)
for symbol, data in spread_analysis.items():
    print(f"{symbol}:")
    print(f"  Bid: ${data['bid']:.2f}")
    print(f"  Ask: ${data['ask']:.2f}")
    print(f"  Spread: ${data['spread_amount']:.2f} ({data['spread_percent']:.2f}%)")
    print()
```

## Extending the Analyzers

### Creating a New Analyzer

To create a new analyzer module:

1. Create a new Python file in the `analyzers` directory
2. Define your analysis functions
3. Add imports to `__init__.py` to make functions accessible

Example of a simple sentiment analyzer:

```python
# sentiment_analyzer.py
from typing import Dict, Any, List
import numpy as np

def analyze_sentiment(text_data: List[str]) -> Dict[str, float]:
    """
    Simple sentiment analysis for cryptocurrency-related text.
    
    Args:
        text_data: List of text strings to analyze
        
    Returns:
        Dictionary with sentiment scores
    """
    # Simplified example - in a real implementation, you would use
    # a proper NLP library or sentiment analysis model
    positive_words = ['bullish', 'up', 'gain', 'positive', 'buy', 'growth']
    negative_words = ['bearish', 'down', 'loss', 'negative', 'sell', 'drop']
    
    sentiment_scores = []
    for text in text_data:
        text = text.lower()
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count + neg_count > 0:
            score = (pos_count - neg_count) / (pos_count + neg_count)
        else:
            score = 0.0
        
        sentiment_scores.append(score)
    
    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
    
    return {
        "average_sentiment": avg_sentiment,
        "individual_scores": sentiment_scores,
        "sample_count": len(text_data)
    }
```

Then update `__init__.py`:

```python
from analyzers.market_analyzer import (
    analyze_price_trends,
    analyze_price_spreads,
    identify_market_opportunities
)
from analyzers.sentiment_analyzer import analyze_sentiment

__all__ = [
    'analyze_price_trends',
    'analyze_price_spreads',
    'identify_market_opportunities',
    'analyze_sentiment'
]
```

## Best Practices

1. **Proper Type Hints**: Use Python type hints to make your analyzer functions more robust and self-documenting
2. **Comprehensive Documentation**: Include detailed docstrings for all functions
3. **Error Handling**: Add appropriate error handling to provide meaningful error messages
4. **Statistical Validation**: Consider adding statistical validation for analysis results
5. **Performance Optimization**: For large datasets, consider optimizing your analysis functions for performance
