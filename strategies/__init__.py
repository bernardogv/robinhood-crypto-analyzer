"""
Trading Strategies Module

This module contains implementations of various trading strategies for cryptocurrency trading,
including basic strategies like Moving Average Crossover and RSI, and advanced strategies
like the XRP-specific trading strategy.
"""

from strategies.basic_strategies import (
    Strategy,
    MovingAverageCrossover,
    RelativeStrengthIndex,
    SpreadTradingStrategy
)

from strategies.xrp_advanced_strategy import XRPAdvancedStrategy

__all__ = [
    'Strategy',
    'MovingAverageCrossover',
    'RelativeStrengthIndex',
    'SpreadTradingStrategy',
    'XRPAdvancedStrategy'
]
