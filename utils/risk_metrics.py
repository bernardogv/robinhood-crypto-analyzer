#!/usr/bin/env python3
"""
Risk Metrics Calculator

This module provides utilities for calculating various risk metrics and statistics
for trading strategies, including drawdown, Sharpe ratio, Sortino ratio, VaR, and more.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

logger = logging.getLogger('risk_metrics')

# Constants
TRADING_DAYS_PER_YEAR = 252  # Standard trading days in a year
RISK_FREE_RATE = 0.02  # Default risk-free rate (2%)

class RiskMetricsCalculator:
    """
    Calculates various risk metrics for a trading strategy.
    """
    
    @staticmethod
    def calculate_returns_series(
        trades: List[Dict[str, Any]], 
        daily_returns: List[Dict[str, Any]]
    ) -> pd.Series:
        """
        Convert trade list and daily returns to a proper returns series.
        
        Args:
            trades: List of trade dictionaries
            daily_returns: List of daily return dictionaries
            
        Returns:
            Series of daily returns
        """
        # Extract daily returns with dates
        if daily_returns:
            df = pd.DataFrame(daily_returns)
            
            # Handle date formatting
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            # Get the returns series
            if 'return' in df.columns:
                return df['return']
        
        # If no daily returns, calculate from trades
        if trades:
            # Extract profit/loss from trades
            trade_df = pd.DataFrame(trades)
            if 'profit_loss_pct' in trade_df.columns and 'date' in trade_df.columns:
                trade_df['date'] = pd.to_datetime(trade_df['date'])
                trade_df.set_index('date', inplace=True)
                # Convert percentage to decimal
                return trade_df['profit_loss_pct'] / 100
        
        # If no valid data, return empty series
        return pd.Series()
    
    @staticmethod
    def calculate_cumulative_returns(returns_series: pd.Series) -> pd.Series:
        """
        Calculate cumulative returns from a returns series.
        
        Args:
            returns_series: Series of daily returns (decimal)
            
        Returns:
            Series of cumulative returns
        """
        if returns_series.empty:
            return pd.Series()
        
        # Calculate cumulative returns
        return (1 + returns_series).cumprod() - 1
    
    @staticmethod
    def calculate_drawdowns(returns_series: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Calculate drawdowns from a returns series.
        
        Args:
            returns_series: Series of daily returns
            
        Returns:
            Tuple of (drawdown_series, drawdown_stats)
        """
        if returns_series.empty:
            return pd.Series(), {"max_drawdown": 0, "avg_drawdown": 0, "drawdown_duration": 0}
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns_series).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.cummax()
        
        # Calculate drawdowns
        drawdowns = (cumulative_returns / running_max - 1)
        
        # Calculate drawdown statistics
        max_drawdown = drawdowns.min()
        avg_drawdown = drawdowns[drawdowns < 0].mean() if any(drawdowns < 0) else 0
        
        # Calculate drawdown duration
        in_drawdown = drawdowns < 0
        drawdown_duration = 0
        current_duration = 0
        
        for is_drawdown in in_drawdown:
            if is_drawdown:
                current_duration += 1
            else:
                drawdown_duration = max(drawdown_duration, current_duration)
                current_duration = 0
        
        # In case we're currently in a drawdown
        drawdown_duration = max(drawdown_duration, current_duration)
        
        drawdown_stats = {
            "max_drawdown": max_drawdown,
            "avg_drawdown": avg_drawdown,
            "drawdown_duration": drawdown_duration
        }
        
        return drawdowns, drawdown_stats
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns_series: pd.Series, 
        risk_free_rate: float = RISK_FREE_RATE,
        annualize: bool = True
    ) -> float:
        """
        Calculate the Sharpe ratio.
        
        Args:
            returns_series: Series of daily returns
            risk_free_rate: Annual risk-free rate (default: 0.02 or 2%)
            annualize: Whether to annualize the ratio
            
        Returns:
            Sharpe ratio
        """
        if returns_series.empty or returns_series.std() == 0:
            return 0
        
        # Calculate excess returns
        excess_returns = returns_series - (risk_free_rate / TRADING_DAYS_PER_YEAR)
        
        # Calculate Sharpe ratio
        daily_sharpe = excess_returns.mean() / returns_series.std()
        
        # Annualize if requested
        if annualize:
            return daily_sharpe * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        return daily_sharpe
    
    @staticmethod
    def calculate_sortino_ratio(
        returns_series: pd.Series, 
        risk_free_rate: float = RISK_FREE_RATE,
        annualize: bool = True
    ) -> float:
        """
        Calculate the Sortino ratio (similar to Sharpe but penalizes only downside volatility).
        
        Args:
            returns_series: Series of daily returns
            risk_free_rate: Annual risk-free rate
            annualize: Whether to annualize the ratio
            
        Returns:
            Sortino ratio
        """
        if returns_series.empty:
            return 0
        
        # Calculate excess returns
        excess_returns = returns_series - (risk_free_rate / TRADING_DAYS_PER_YEAR)
        
        # Calculate downside deviation
        negative_returns = returns_series[returns_series < 0]
        downside_deviation = negative_returns.std()
        
        # If there are no negative returns, return a large number
        if len(negative_returns) == 0 or downside_deviation == 0:
            return float('inf') if excess_returns.mean() > 0 else 0
        
        # Calculate Sortino ratio
        daily_sortino = excess_returns.mean() / downside_deviation
        
        # Annualize if requested
        if annualize:
            return daily_sortino * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        return daily_sortino
    
    @staticmethod
    def calculate_calmar_ratio(
        returns_series: pd.Series, 
        drawdowns: Optional[pd.Series] = None,
        annualize: bool = True
    ) -> float:
        """
        Calculate the Calmar ratio (annualized return / maximum drawdown).
        
        Args:
            returns_series: Series of daily returns
            drawdowns: Optional pre-calculated drawdowns
            annualize: Whether to annualize the returns
            
        Returns:
            Calmar ratio
        """
        if returns_series.empty:
            return 0
        
        # Calculate annualized return
        ann_return = (1 + returns_series.mean()) ** TRADING_DAYS_PER_YEAR - 1 if annualize else returns_series.mean()
        
        # Get maximum drawdown
        if drawdowns is None:
            _, drawdown_stats = RiskMetricsCalculator.calculate_drawdowns(returns_series)
            max_drawdown = drawdown_stats['max_drawdown']
        else:
            max_drawdown = drawdowns.min()
        
        # Avoid division by zero
        if max_drawdown == 0:
            return 0 if ann_return == 0 else float('inf') if ann_return > 0 else float('-inf')
        
        # Calculate Calmar ratio (use absolute value of max drawdown since it's negative)
        return ann_return / abs(max_drawdown)
    
    @staticmethod
    def calculate_var(
        returns_series: pd.Series, 
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate the Value at Risk (VaR).
        
        Args:
            returns_series: Series of daily returns
            confidence_level: Confidence level for VaR calculation (default: 0.95 or 95%)
            
        Returns:
            Value at Risk
        """
        if returns_series.empty:
            return 0
        
        # Calculate VaR
        return returns_series.quantile(1 - confidence_level)
    
    @staticmethod
    def calculate_cvar(
        returns_series: pd.Series, 
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate the Conditional Value at Risk (CVaR) or Expected Shortfall.
        
        Args:
            returns_series: Series of daily returns
            confidence_level: Confidence level for CVaR calculation
            
        Returns:
            Conditional Value at Risk
        """
        if returns_series.empty:
            return 0
        
        # Calculate VaR
        var = RiskMetricsCalculator.calculate_var(returns_series, confidence_level)
        
        # Calculate CVaR (expected shortfall beyond VaR)
        return returns_series[returns_series <= var].mean()
    
    @staticmethod
    def calculate_win_rate(trades: List[Dict[str, Any]]) -> float:
        """
        Calculate the win rate (percentage of winning trades).
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Win rate as a percentage
        """
        if not trades:
            return 0
        
        # Count winning trades
        winning_trades = sum(1 for trade in trades if trade.get('profit_loss_pct', 0) > 0)
        
        # Calculate win rate
        return (winning_trades / len(trades)) * 100
    
    @staticmethod
    def calculate_profit_factor(trades: List[Dict[str, Any]]) -> float:
        """
        Calculate the profit factor (gross profit / gross loss).
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Profit factor
        """
        if not trades:
            return 0
        
        # Calculate gross profit and loss
        gross_profit = sum(trade.get('profit_loss', 0) for trade in trades if trade.get('profit_loss', 0) > 0)
        gross_loss = sum(abs(trade.get('profit_loss', 0)) for trade in trades if trade.get('profit_loss', 0) < 0)
        
        # Calculate profit factor
        return gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
    
    @staticmethod
    def calculate_expectancy(trades: List[Dict[str, Any]]) -> float:
        """
        Calculate the expectancy (average profit per trade).
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Expectancy
        """
        if not trades:
            return 0
        
        # Calculate total profit/loss
        total_pnl = sum(trade.get('profit_loss', 0) for trade in trades)
        
        # Calculate expectancy
        return total_pnl / len(trades)
    
    @staticmethod
    def calculate_risk_metrics(
        trades: List[Dict[str, Any]], 
        daily_returns: List[Dict[str, Any]],
        risk_free_rate: float = RISK_FREE_RATE
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics for a trading strategy.
        
        Args:
            trades: List of trade dictionaries
            daily_returns: List of daily return dictionaries
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary of risk metrics
        """
        # Get returns series
        returns_series = RiskMetricsCalculator.calculate_returns_series(trades, daily_returns)
        
        # If we have no returns data, return basic trade metrics only
        if returns_series.empty:
            return {
                "win_rate": RiskMetricsCalculator.calculate_win_rate(trades),
                "profit_factor": RiskMetricsCalculator.calculate_profit_factor(trades),
                "expectancy": RiskMetricsCalculator.calculate_expectancy(trades),
                "num_trades": len(trades)
            }
        
        # Calculate drawdowns
        drawdowns, drawdown_stats = RiskMetricsCalculator.calculate_drawdowns(returns_series)
        
        # Calculate comprehensive metrics
        metrics = {
            # Return metrics
            "total_return": ((1 + returns_series).prod() - 1) * 100,  # as percentage
            "annualized_return": ((1 + returns_series.mean()) ** TRADING_DAYS_PER_YEAR - 1) * 100,  # as percentage
            "volatility": returns_series.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100,  # as percentage
            
            # Risk-adjusted return metrics
            "sharpe_ratio": RiskMetricsCalculator.calculate_sharpe_ratio(returns_series, risk_free_rate),
            "sortino_ratio": RiskMetricsCalculator.calculate_sortino_ratio(returns_series, risk_free_rate),
            "calmar_ratio": RiskMetricsCalculator.calculate_calmar_ratio(returns_series, drawdowns),
            
            # Drawdown metrics
            "max_drawdown": drawdown_stats["max_drawdown"] * 100,  # as percentage
            "avg_drawdown": drawdown_stats["avg_drawdown"] * 100 if drawdown_stats["avg_drawdown"] != 0 else 0,  # as percentage
            "drawdown_duration": drawdown_stats["drawdown_duration"],
            
            # Risk metrics
            "var_95": abs(RiskMetricsCalculator.calculate_var(returns_series, 0.95)) * 100,  # as percentage
            "cvar_95": abs(RiskMetricsCalculator.calculate_cvar(returns_series, 0.95)) * 100,  # as percentage
            
            # Trade metrics
            "win_rate": RiskMetricsCalculator.calculate_win_rate(trades),
            "profit_factor": RiskMetricsCalculator.calculate_profit_factor(trades),
            "expectancy": RiskMetricsCalculator.calculate_expectancy(trades),
            "num_trades": len(trades)
        }
        
        return metrics


def main():
    """Test the risk metrics calculator."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample trades
    trades = [
        {"action": "buy", "entry_price": 2.50, "quantity": 1000, "date": "2025-01-01"},
        {"action": "sell", "entry_price": 2.50, "exit_price": 2.75, "quantity": 1000, "profit_loss": 250, "profit_loss_pct": 10.0, "date": "2025-01-10"},
        {"action": "buy", "entry_price": 2.60, "quantity": 1000, "date": "2025-01-15"},
        {"action": "sell", "entry_price": 2.60, "exit_price": 2.50, "quantity": 1000, "profit_loss": -100, "profit_loss_pct": -3.8, "date": "2025-01-25"},
        {"action": "buy", "entry_price": 2.45, "quantity": 1000, "date": "2025-02-01"},
        {"action": "sell", "entry_price": 2.45, "exit_price": 2.80, "quantity": 1000, "profit_loss": 350, "profit_loss_pct": 14.3, "date": "2025-02-15"}
    ]
    
    # Create sample daily returns
    daily_returns = []
    dates = pd.date_range(start="2025-01-01", end="2025-02-15")
    
    for date in dates:
        # For simplicity, only days with positions have non-zero returns
        is_position_day = False
        for trade in trades:
            trade_date = pd.to_datetime(trade["date"])
            if trade["action"] == "buy" and date >= trade_date:
                is_position_day = True
            elif trade["action"] == "sell" and date >= trade_date:
                is_position_day = False
        
        # Assign random returns for position days
        if is_position_day:
            daily_return = np.random.normal(0.001, 0.01)  # 0.1% mean, 1% std
        else:
            daily_return = 0
        
        daily_returns.append({
            "date": date,
            "return": daily_return,
            "position": "long" if is_position_day else "none"
        })
    
    # Calculate risk metrics
    metrics = RiskMetricsCalculator.calculate_risk_metrics(trades, daily_returns)
    
    # Print results
    print("Risk Metrics Results:")
    for metric, value in metrics.items():
        print(f"- {metric}: {value:.4f}" if isinstance(value, float) else f"- {metric}: {value}")


if __name__ == "__main__":
    main()
