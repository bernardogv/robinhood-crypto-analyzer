#!/usr/bin/env python3
"""
Transaction Cost Calculator

This module provides utilities for calculating and simulating various trading costs
including spreads, fees, and slippage to make backtesting more realistic.
"""

import random
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger('transaction_costs')

class TransactionCostCalculator:
    """
    Calculates transaction costs for trades, including:
    - Trading fees
    - Bid-ask spread costs
    - Slippage based on order size and liquidity
    """
    
    def __init__(
        self,
        fee_rate: float = 0.0025,  # 0.25% default trading fee (Robinhood)
        spread_factor: float = 0.001,  # 0.1% default spread
        slippage_factor: float = 0.0005,  # 0.05% default slippage
        volume_impact: float = 0.1,  # How much volume impacts slippage
        min_fee: float = 0.0,  # Minimum fee per trade
        max_slippage: float = 0.01,  # 1% maximum slippage
        volatility_impact: float = 0.5  # How much volatility impacts slippage
    ):
        """
        Initialize the transaction cost calculator.
        
        Args:
            fee_rate: Trading fee as a decimal (e.g., 0.0025 for 0.25%)
            spread_factor: Typical bid-ask spread as a decimal
            slippage_factor: Base slippage factor
            volume_impact: How much volume impacts slippage (0-1)
            min_fee: Minimum fee per trade
            max_slippage: Maximum slippage allowed
            volatility_impact: How much volatility impacts slippage (0-1)
        """
        self.fee_rate = fee_rate
        self.spread_factor = spread_factor
        self.slippage_factor = slippage_factor
        self.volume_impact = volume_impact
        self.min_fee = min_fee
        self.max_slippage = max_slippage
        self.volatility_impact = volatility_impact
    
    def calculate_trading_fee(self, order_value: float) -> float:
        """
        Calculate the trading fee for an order.
        
        Args:
            order_value: Total value of the order
            
        Returns:
            Trading fee amount
        """
        fee = order_value * self.fee_rate
        return max(fee, self.min_fee)
    
    def calculate_spread_cost(self, price: float, side: str = 'buy') -> float:
        """
        Calculate the cost of the bid-ask spread.
        
        Args:
            price: Current price
            side: Order side ('buy' or 'sell')
            
        Returns:
            Adjusted price after applying spread
        """
        half_spread = price * self.spread_factor / 2
        
        if side.lower() == 'buy':
            # Buy at the ask price (higher)
            return price + half_spread
        else:
            # Sell at the bid price (lower)
            return price - half_spread
    
    def calculate_slippage(
        self,
        price: float,
        side: str,
        order_value: float,
        market_volume: float,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate price slippage based on order size, market volume, and volatility.
        
        Args:
            price: Current price
            side: Order side ('buy' or 'sell')
            order_value: Total value of the order
            market_volume: Daily market volume in the same currency
            volatility: Optional price volatility (standard deviation)
            
        Returns:
            Adjusted price after applying slippage
        """
        # Base slippage
        base_slippage = self.slippage_factor
        
        # Volume-based slippage
        volume_ratio = min(order_value / market_volume, 1.0) if market_volume > 0 else 0
        volume_slippage = base_slippage * (1 + self.volume_impact * volume_ratio * 10)
        
        # Volatility-based slippage (if provided)
        volatility_slippage = 0
        if volatility is not None:
            # Higher volatility means more slippage
            # Normalize volatility to a reasonable range (0-0.1)
            norm_volatility = min(volatility * 10, 0.1)
            volatility_slippage = base_slippage * self.volatility_impact * norm_volatility
        
        # Combine slippage factors
        total_slippage = min(base_slippage + volume_slippage + volatility_slippage, self.max_slippage)
        
        # Add some randomness to simulate real-world conditions
        randomness = random.uniform(0.8, 1.2)
        total_slippage *= randomness
        
        # Apply slippage to the price
        if side.lower() == 'buy':
            # Higher price when buying
            return price * (1 + total_slippage)
        else:
            # Lower price when selling
            return price * (1 - total_slippage)
    
    def calculate_total_cost(
        self,
        price: float,
        side: str,
        quantity: float,
        market_volume: float,
        volatility: Optional[float] = None,
        include_fees: bool = True,
        include_spread: bool = True,
        include_slippage: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total transaction cost and execution price.
        
        Args:
            price: Current market price
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            market_volume: Daily market volume
            volatility: Optional price volatility
            include_fees: Whether to include trading fees
            include_spread: Whether to include bid-ask spread
            include_slippage: Whether to include price slippage
            
        Returns:
            Tuple of (execution_price, cost_breakdown)
        """
        execution_price = price
        order_value = price * quantity
        
        # Transaction cost components
        costs = {
            'base_price': price,
            'fee': 0.0,
            'spread_cost': 0.0,
            'slippage_cost': 0.0,
            'total_cost_pct': 0.0
        }
        
        # Apply spread if enabled
        if include_spread:
            spread_price = self.calculate_spread_cost(price, side)
            spread_cost = abs(spread_price - price) * quantity
            execution_price = spread_price
            costs['spread_cost'] = spread_cost
        
        # Apply slippage if enabled
        if include_slippage:
            slippage_price = self.calculate_slippage(
                execution_price, side, order_value, market_volume, volatility
            )
            slippage_cost = abs(slippage_price - execution_price) * quantity
            execution_price = slippage_price
            costs['slippage_cost'] = slippage_cost
        
        # Calculate fee if enabled
        if include_fees:
            fee = self.calculate_trading_fee(execution_price * quantity)
            costs['fee'] = fee
        
        # Calculate total costs
        total_cost = costs['fee'] + costs['spread_cost'] + costs['slippage_cost']
        costs['total_cost'] = total_cost
        
        # Calculate cost as percentage of trade value
        costs['total_cost_pct'] = (total_cost / order_value) * 100 if order_value > 0 else 0
        
        # Final execution price after all costs
        if side.lower() == 'buy':
            # For buy orders, the effective price is higher due to costs
            effective_price = execution_price
        else:
            # For sell orders, the effective price is lower due to costs
            effective_price = execution_price
        
        return effective_price, costs


class RobinhoodCostModel(TransactionCostCalculator):
    """
    Transaction cost model specifically tuned for Robinhood trading.
    """
    
    def __init__(self):
        """Initialize with Robinhood-specific parameters."""
        super().__init__(
            fee_rate=0.0,  # Robinhood claims zero commission
            spread_factor=0.0025,  # 0.25% spread (Robinhood makes money on the spread)
            slippage_factor=0.0015,  # 0.15% base slippage
            volume_impact=0.15,
            min_fee=0.0,
            max_slippage=0.015,  # 1.5% max slippage
            volatility_impact=0.6
        )


def calculate_trade_costs_for_backtest(
    trades: List[Dict[str, Any]],
    price_data: pd.DataFrame,
    cost_model: Optional[TransactionCostCalculator] = None
) -> List[Dict[str, Any]]:
    """
    Calculate realistic costs for a list of backtest trades.
    
    Args:
        trades: List of trade dictionaries
        price_data: DataFrame with price and volume data
        cost_model: Transaction cost model to use (default: RobinhoodCostModel)
        
    Returns:
        List of trades with updated costs and prices
    """
    # Use default cost model if not provided
    if cost_model is None:
        cost_model = RobinhoodCostModel()
    
    # Calculate volatility if 'Price' column exists
    volatility = None
    if 'Price' in price_data.columns:
        # Calculate 14-day rolling volatility
        volatility = price_data['Price'].pct_change().rolling(window=14).std().iloc[-1]
    
    updated_trades = []
    
    for trade in trades:
        # Skip if not a buy or sell trade
        if 'action' not in trade or trade['action'] not in ['buy', 'sell', 'stop_loss', 'take_profit']:
            updated_trades.append(trade)
            continue
        
        # Map action to side
        side = 'buy' if trade['action'] == 'buy' else 'sell'
        
        # Get original price
        original_price = trade.get('entry_price') if side == 'buy' else trade.get('exit_price', 0)
        
        # Get quantity
        quantity = trade.get('quantity', 0)
        
        # Get market volume (if available)
        trade_date = trade.get('date')
        market_volume = 0
        
        if trade_date is not None and 'Date' in price_data.columns and 'Volume' in price_data.columns:
            # Find closest date in price data
            if isinstance(trade_date, str):
                trade_date = pd.to_datetime(trade_date)
            
            price_data['Date'] = pd.to_datetime(price_data['Date'])
            closest_idx = (price_data['Date'] - trade_date).abs().idxmin()
            market_volume = price_data.loc[closest_idx, 'Volume']
        
        # Calculate costs
        try:
            execution_price, costs = cost_model.calculate_total_cost(
                original_price, side, quantity, market_volume, volatility
            )
            
            # Update the trade with cost info
            if side == 'buy':
                trade['original_entry_price'] = trade.get('entry_price')
                trade['entry_price'] = execution_price
            else:
                trade['original_exit_price'] = trade.get('exit_price')
                trade['exit_price'] = execution_price
            
            # Add cost breakdown to trade
            trade['transaction_costs'] = costs
            
            # Recalculate profit/loss if applicable
            if side == 'sell' and 'entry_price' in trade:
                entry_price = trade.get('entry_price', 0)
                exit_price = execution_price
                quantity = trade.get('quantity', 0)
                
                profit_loss = (exit_price - entry_price) * quantity - costs['fee']
                profit_loss_pct = ((exit_price / entry_price) - 1) * 100
                
                trade['profit_loss'] = profit_loss
                trade['profit_loss_pct'] = profit_loss_pct
            
            updated_trades.append(trade)
            
        except Exception as e:
            logger.error(f"Error calculating costs for trade: {e}")
            updated_trades.append(trade)
    
    return updated_trades


def main():
    """Test the transaction cost calculator."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a cost model
    cost_model = RobinhoodCostModel()
    
    # Test with a sample buy order
    price = 2.50  # XRP price in USD
    quantity = 1000  # Amount of XRP
    market_volume = 20000000  # Daily trading volume
    volatility = 0.02  # 2% daily volatility
    
    buy_price, buy_costs = cost_model.calculate_total_cost(
        price, 'buy', quantity, market_volume, volatility
    )
    
    print(f"Buy Order (Original price: ${price:.4f}, Quantity: {quantity}):")
    print(f"Execution price: ${buy_price:.4f}")
    print(f"Total cost: ${buy_costs['total_cost']:.2f} ({buy_costs['total_cost_pct']:.2f}%)")
    print(f"Breakdown - Fees: ${buy_costs['fee']:.2f}, Spread: ${buy_costs['spread_cost']:.2f}, Slippage: ${buy_costs['slippage_cost']:.2f}")
    
    # Test with a sample sell order
    sell_price, sell_costs = cost_model.calculate_total_cost(
        price, 'sell', quantity, market_volume, volatility
    )
    
    print(f"\nSell Order (Original price: ${price:.4f}, Quantity: {quantity}):")
    print(f"Execution price: ${sell_price:.4f}")
    print(f"Total cost: ${sell_costs['total_cost']:.2f} ({sell_costs['total_cost_pct']:.2f}%)")
    print(f"Breakdown - Fees: ${sell_costs['fee']:.2f}, Spread: ${sell_costs['spread_cost']:.2f}, Slippage: ${sell_costs['slippage_cost']:.2f}")


if __name__ == "__main__":
    main()
