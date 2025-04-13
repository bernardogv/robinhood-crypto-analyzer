"""
Market Analyzer Module

This module provides functions for analyzing cryptocurrency market data,
including price analysis, spread analysis, and market trend identification.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

def analyze_price_trends(
    price_history: pd.DataFrame, 
    window_short: int = 5, 
    window_long: int = 20
) -> Dict[str, Any]:
    """
    Analyze price trends using moving averages and momentum indicators.
    
    Args:
        price_history: DataFrame with historical price data
        window_short: Short-term moving average window
        window_long: Long-term moving average window
        
    Returns:
        Dictionary with trend analysis results
    """
    if len(price_history) < window_long:
        return {"error": "Not enough data for trend analysis"}
    
    # Create a copy of the price history
    df = price_history.copy()
    
    # Calculate moving averages
    df['SMA_short'] = df['Price'].rolling(window=window_short).mean()
    df['SMA_long'] = df['Price'].rolling(window=window_long).mean()
    
    # Calculate momentum (price change)
    df['momentum_1d'] = df['Price'].pct_change(1)
    df['momentum_5d'] = df['Price'].pct_change(5)
    
    # Determine trend
    current_price = df['Price'].iloc[-1]
    sma_short = df['SMA_short'].iloc[-1]
    sma_long = df['SMA_long'].iloc[-1]
    
    if sma_short > sma_long:
        trend = "bullish"
    elif sma_short < sma_long:
        trend = "bearish"
    else:
        trend = "neutral"
    
    # Calculate additional metrics
    volatility = df['Price'].rolling(window=window_long).std() / df['Price'].rolling(window=window_long).mean()
    current_volatility = volatility.iloc[-1]
    
    # Recent momentum
    recent_momentum = df['momentum_5d'].iloc[-1] * 100  # As percentage
    
    return {
        "trend": trend,
        "current_price": current_price,
        "sma_short": sma_short,
        "sma_long": sma_long,
        "recent_momentum": recent_momentum,
        "current_volatility": current_volatility,
        "data": df
    }

def analyze_price_spreads(
    bid_ask_data: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Analyze bid-ask spreads for trading pairs.
    
    Args:
        bid_ask_data: Dictionary mapping symbols to bid and ask price data
        
    Returns:
        Dictionary with spread analysis for each symbol
    """
    spreads = {}
    
    for symbol, data in bid_ask_data.items():
        bid = data.get("bid", 0)
        ask = data.get("ask", 0)
        
        if bid > 0 and ask > 0:
            spread_amount = ask - bid
            spread_percent = (spread_amount / bid) * 100
            mid_price = (bid + ask) / 2
            
            spreads[symbol] = {
                "bid": bid,
                "ask": ask,
                "mid_price": mid_price,
                "spread_amount": spread_amount,
                "spread_percent": spread_percent
            }
    
    return spreads

def identify_market_opportunities(
    spread_data: Dict[str, Dict[str, float]],
    volatility_data: Dict[str, float],
    min_spread_percent: float = 0.5,
    max_volatility: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Identify potential trading opportunities based on spread and volatility.
    
    Args:
        spread_data: Dictionary with spread analysis for each symbol
        volatility_data: Dictionary mapping symbols to volatility values
        min_spread_percent: Minimum spread percentage to consider
        max_volatility: Maximum volatility to consider
        
    Returns:
        List of potential trading opportunities
    """
    opportunities = []
    
    for symbol, data in spread_data.items():
        spread_percent = data.get("spread_percent", 0)
        volatility = volatility_data.get(symbol, 1.0)
        
        # Calculate opportunity score (higher is better)
        # We want high spread and low volatility
        if spread_percent >= min_spread_percent and volatility <= max_volatility:
            score = spread_percent / (volatility * 100)
            
            opportunities.append({
                "symbol": symbol,
                "spread_percent": spread_percent,
                "volatility": volatility,
                "score": score,
                "mid_price": data.get("mid_price", 0)
            })
    
    # Sort by score (descending)
    opportunities.sort(key=lambda x: x["score"], reverse=True)
    
    return opportunities
