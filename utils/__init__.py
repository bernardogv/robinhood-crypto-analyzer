"""
Utilities Module

This module provides various utility functions for the Robinhood Crypto Analyzer,
including key generation, historical data handling, risk metrics, and transaction cost analysis.
"""

from utils.generate_keys import generate_key_pair, save_key_pair
from utils.historical_data import (
    fetch_historical_data,
    process_historical_data,
    save_historical_data
)
from utils.risk_metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_value_at_risk
)
from utils.transaction_costs import (
    calculate_transaction_cost,
    estimate_slippage
)
from utils.xrp_market_analysis import (
    analyze_xrp_market,
    get_xrp_sentiment
)

__all__ = [
    # Key generation
    'generate_key_pair', 
    'save_key_pair',
    
    # Historical data
    'fetch_historical_data',
    'process_historical_data',
    'save_historical_data',
    
    # Risk metrics
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_max_drawdown',
    'calculate_value_at_risk',
    
    # Transaction costs
    'calculate_transaction_cost',
    'estimate_slippage',
    
    # XRP market analysis
    'analyze_xrp_market',
    'get_xrp_sentiment'
]
