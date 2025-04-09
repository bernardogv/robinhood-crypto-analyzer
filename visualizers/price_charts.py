#!/usr/bin/env python3
"""
Price Chart Visualizations for Robinhood Crypto Analyzer

This module provides functions for visualizing cryptocurrency price data.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Set seaborn style for better aesthetics
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)

def plot_price_comparison(price_data: Dict[str, Dict[str, float]], save_path: Optional[str] = None):
    """
    Plot a comparison of cryptocurrency prices.
    
    Args:
        price_data: Dictionary mapping symbols to price data
        save_path: Optional path to save the chart image
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Create a figure and axis
    fig, ax = plt.subplots()
    
    # Extract data for plotting
    symbols = []
    bid_prices = []
    ask_prices = []
    mid_prices = []
    
    for symbol, data in price_data.items():
        symbols.append(symbol.split("-")[0])  # Remove the "-USD" part
        bid_prices.append(data.get("bid", 0))
        ask_prices.append(data.get("ask", 0))
        mid_prices.append(data.get("mid_price", 0))
    
    # Sort by mid price (descending)
    sorted_indices = np.argsort(mid_prices)[::-1]
    symbols = [symbols[i] for i in sorted_indices]
    bid_prices = [bid_prices[i] for i in sorted_indices]
    ask_prices = [ask_prices[i] for i in sorted_indices]
    mid_prices = [mid_prices[i] for i in sorted_indices]
    
    # Create a range for the x-axis
    x = np.arange(len(symbols))
    width = 0.25
    
    # Create the bars
    ax.bar(x - width, bid_prices, width, label="Bid Price")
    ax.bar(x, mid_prices, width, label="Mid Price")
    ax.bar(x + width, ask_prices, width, label="Ask Price")
    
    # Add labels and title
    ax.set_xlabel("Cryptocurrency")
    ax.set_ylabel("Price (USD)")
    ax.set_title("Cryptocurrency Price Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(symbols)
    ax.legend()
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter('${x:,.2f}')
    
    # Adjust layout and save if a save path is provided
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to {save_path}")
    
    return fig

def plot_price_spread(spread_data: Dict[str, Dict[str, float]], save_path: Optional[str] = None):
    """
    Plot the bid-ask spreads of cryptocurrencies.
    
    Args:
        spread_data: Dictionary mapping symbols to spread data
        save_path: Optional path to save the chart image
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Create a figure and axis
    fig, ax = plt.subplots()
    
    # Extract data for plotting
    symbols = []
    spread_percents = []
    
    for symbol, data in spread_data.items():
        symbols.append(symbol.split("-")[0])  # Remove the "-USD" part
        spread_percents.append(data.get("spread_percent", 0))
    
    # Sort by spread percentage (ascending)
    sorted_indices = np.argsort(spread_percents)
    symbols = [symbols[i] for i in sorted_indices]
    spread_percents = [spread_percents[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    ax.barh(symbols, spread_percents, color=sns.color_palette("viridis", len(symbols)))
    
    # Add labels and title
    ax.set_xlabel("Spread (%)")
    ax.set_ylabel("Cryptocurrency")
    ax.set_title("Bid-Ask Spread Comparison")
    
    # Add data labels
    for i, v in enumerate(spread_percents):
        ax.text(v + 0.05, i, f"{v:.2f}%", va="center")
    
    # Adjust layout and save if a save path is provided
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to {save_path}")
    
    return fig

def plot_portfolio_distribution(holdings_data: Dict[str, Dict[str, Any]], save_path: Optional[str] = None):
    """
    Plot the distribution of cryptocurrencies in a portfolio.
    
    Args:
        holdings_data: Dictionary mapping asset codes to holdings data
        save_path: Optional path to save the chart image
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Extract data for plotting
    asset_codes = []
    values = []
    
    for asset_code, data in holdings_data.items():
        current_value = data.get("current_value", 0)
        if current_value > 0:
            asset_codes.append(asset_code)
            values.append(current_value)
    
    # Create a pie chart
    fig, ax = plt.subplots()
    
    # Calculate total portfolio value
    total_value = sum(values)
    
    # Create the pie chart
    wedges, texts, autotexts = ax.pie(
        values, 
        labels=asset_codes, 
        autopct=lambda p: f'{p:.1f}%\n(${p * total_value / 100:.2f})', 
        startangle=90,
        colors=sns.color_palette("viridis", len(asset_codes))
    )
    
    # Enhance the text properties
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")
    
    # Add title
    ax.set_title(f"Portfolio Distribution (Total: ${total_value:.2f})")
    
    # Adjust layout and save if a save path is provided
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to {save_path}")
    
    return fig

def simulate_price_history(
    symbol: str, 
    current_price: float, 
    days: int = 30, 
    volatility: float = 0.02,
    save_path: Optional[str] = None
):
    """
    Simulate price history for demonstration purposes.
    In a real application, this would be replaced with actual historical data.
    
    Args:
        symbol: Trading pair symbol
        current_price: Current price
        days: Number of days to simulate
        volatility: Daily price volatility
        save_path: Optional path to save the chart image
    
    Returns:
        Tuple[matplotlib.figure.Figure, pd.DataFrame]: The figure object and simulated data
    """
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    # Initialize with the current price and work backwards
    np.random.seed(42)  # For reproducibility
    prices = [current_price]
    
    for _ in range(days):
        # Random daily return with drift based on volatility
        daily_return = np.random.normal(0, volatility)
        previous_price = prices[-1]
        new_price = previous_price / (1 + daily_return)  # Working backwards
        prices.append(new_price)
    
    # Reverse to get chronological order
    prices.reverse()
    
    # Create a DataFrame
    df = pd.DataFrame({
        "Date": dates,
        "Price": prices
    })
    
    # Plot the data
    fig, ax = plt.subplots()
    
    ax.plot(df["Date"], df["Price"], label=symbol, linewidth=2)
    
    # Add labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.set_title(f"{symbol} Simulated Price History (Last {days} Days)")
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter('${x:,.2f}')
    
    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Adjust layout and save if a save path is provided
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to {save_path}")
    
    return fig, df
