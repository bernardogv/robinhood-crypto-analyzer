#!/usr/bin/env python3
"""
Dashboard Visualization for Robinhood Crypto Analyzer

This module provides a Dash web application for visualizing cryptocurrency data.
"""

import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class CryptoDashboard:
    """
    A Dash web application for visualizing cryptocurrency data.
    """
    
    def __init__(self, api_client):
        """
        Initialize the dashboard.
        
        Args:
            api_client: Robinhood Crypto API client
        """
        self.api = api_client
        self.app = Dash(__name__, title="Robinhood Crypto Analyzer")
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Robinhood Crypto Analyzer Dashboard"),
            
            html.Div([
                html.H2("Market Overview"),
                html.Button("Refresh Data", id="refresh-button", n_clicks=0),
                html.Div(id="last-update-time"),
                
                dcc.Tabs([
                    dcc.Tab(label="Price Comparison", children=[
                        dcc.Graph(id="price-comparison-chart")
                    ]),
                    dcc.Tab(label="Spread Analysis", children=[
                        dcc.Graph(id="spread-chart")
                    ]),
                    dcc.Tab(label="Portfolio", children=[
                        dcc.Graph(id="portfolio-chart")
                    ])
                ])
            ]),
            
            html.Div([
                html.H2("Cryptocurrency Details"),
                dcc.Dropdown(
                    id="crypto-dropdown",
                    options=[
                        {"label": "Bitcoin (BTC)", "value": "BTC-USD"},
                        {"label": "Ethereum (ETH)", "value": "ETH-USD"},
                        {"label": "Solana (SOL)", "value": "SOL-USD"},
                        {"label": "Dogecoin (DOGE)", "value": "DOGE-USD"},
                        {"label": "Cardano (ADA)", "value": "ADA-USD"}
                    ],
                    value="BTC-USD"
                ),
                dcc.Graph(id="price-history-chart"),
                
                html.Div([
                    html.H3("Trading Signals"),
                    html.Div(id="trading-signals")
                ])
            ])
        ], style={"maxWidth": "1200px", "margin": "0 auto", "padding": "20px"})
    
    def setup_callbacks(self):
        """Set up the dashboard callbacks."""
        
        @self.app.callback(
            [Output("last-update-time", "children"),
             Output("price-comparison-chart", "figure"),
             Output("spread-chart", "figure"),
             Output("portfolio-chart", "figure")],
            [Input("refresh-button", "n_clicks")]
        )
        def update_market_overview(n_clicks):
            """
            Update the market overview charts.
            
            Args:
                n_clicks: Number of button clicks
                
            Returns:
                Tuple of updated outputs
            """
            # Get current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Get market data
            market_data = self.get_market_data()
            
            # Get portfolio data
            portfolio_data = self.get_portfolio_data()
            
            # Create price comparison chart
            price_chart = self.create_price_comparison_chart(market_data)
            
            # Create spread chart
            spread_chart = self.create_spread_chart(market_data)
            
            # Create portfolio chart
            portfolio_chart = self.create_portfolio_chart(portfolio_data)
            
            return (
                f"Last Updated: {current_time}",
                price_chart,
                spread_chart,
                portfolio_chart
            )
        
        @self.app.callback(
            [Output("price-history-chart", "figure"),
             Output("trading-signals", "children")],
            [Input("crypto-dropdown", "value"),
             Input("refresh-button", "n_clicks")]
        )
        def update_crypto_details(symbol, n_clicks):
            """
            Update the cryptocurrency details.
            
            Args:
                symbol: Trading pair symbol
                n_clicks: Number of button clicks
                
            Returns:
                Tuple of updated outputs
            """
            # Get price history (simulated for demo)
            price_history = self.get_simulated_price_history(symbol)
            
            # Create price history chart
            price_history_chart = self.create_price_history_chart(symbol, price_history)
            
            # Generate trading signals
            signals_html = self.generate_trading_signals(symbol, price_history)
            
            return price_history_chart, signals_html
    
    def get_market_data(self) -> Dict[str, Dict[str, float]]:
        """
        Get market data from the API.
        
        Returns:
            Dict[str, Dict[str, float]]: Market data
        """
        symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "ADA-USD"]
        
        try:
            best_prices = self.api.get_best_bid_ask(*symbols)
            
            market_data = {}
            
            if "results" in best_prices:
                for price in best_prices["results"]:
                    symbol = price.get("symbol", "N/A")
                    bid = float(price.get("bid_price", 0))
                    ask = float(price.get("ask_price", 0))
                    spread = ask - bid
                    spread_percent = (spread / bid) * 100 if bid > 0 else 0
                    
                    market_data[symbol] = {
                        "bid": bid,
                        "ask": ask,
                        "mid_price": (bid + ask) / 2,
                        "spread_amount": spread,
                        "spread_percent": spread_percent
                    }
            
            return market_data
        except Exception as e:
            print(f"Error getting market data: {e}")
            return {}
    
    def get_portfolio_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get portfolio data from the API.
        
        Returns:
            Dict[str, Dict[str, Any]]: Portfolio data
        """
        try:
            holdings = self.api.get_holdings()
            
            portfolio_data = {}
            
            if "results" in holdings:
                for holding in holdings["results"]:
                    asset_code = holding.get("asset_code", "N/A")
                    quantity = float(holding.get("quantity", 0))
                    
                    # Get current price for this asset
                    best_price = self.api.get_best_bid_ask(f"{asset_code}-USD")
                    
                    current_price = 0
                    if "results" in best_price and best_price["results"]:
                        bid = float(best_price["results"][0].get("bid_price", 0))
                        ask = float(best_price["results"][0].get("ask_price", 0))
                        current_price = (bid + ask) / 2
                    
                    current_value = quantity * current_price
                    
                    portfolio_data[asset_code] = {
                        "quantity": quantity,
                        "current_price": current_price,
                        "current_value": current_value
                    }
            
            return portfolio_data
        except Exception as e:
            print(f"Error getting portfolio data: {e}")
            return {}
    
    def get_simulated_price_history(self, symbol: str) -> pd.DataFrame:
        """
        Get simulated price history for a symbol.
        In a real application, this would use actual historical data.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            pd.DataFrame: Price history data
        """
        # Try to get current price from the API
        current_price = None
        try:
            best_price = self.api.get_best_bid_ask(symbol)
            if "results" in best_price and best_price["results"]:
                bid = float(best_price["results"][0].get("bid_price", 0))
                ask = float(best_price["results"][0].get("ask_price", 0))
                current_price = (bid + ask) / 2
        except Exception:
            pass
        
        # Use a default price if API call failed
        if not current_price:
            default_prices = {
                "BTC-USD": 65000,
                "ETH-USD": 3500,
                "SOL-USD": 150,
                "DOGE-USD": 0.12,
                "ADA-USD": 0.45
            }
            current_price = default_prices.get(symbol, 100)
        
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        
        # Initialize with the current price and work backwards
        import numpy as np
        np.random.seed(hash(symbol) % 10000)  # Different seed for each symbol
        volatility = 0.02  # Daily price volatility
        
        prices = [current_price]
        
        for _ in range(len(dates) - 1):
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
        
        return df
    
    def create_price_comparison_chart(self, market_data: Dict[str, Dict[str, float]]) -> go.Figure:
        """
        Create a price comparison chart.
        
        Args:
            market_data: Market data
            
        Returns:
            go.Figure: Plotly figure
        """
        # Extract data for plotting
        symbols = []
        bid_prices = []
        ask_prices = []
        mid_prices = []
        
        for symbol, data in market_data.items():
            symbols.append(symbol.split("-")[0])  # Remove the "-USD" part
            bid_prices.append(data.get("bid", 0))
            ask_prices.append(data.get("ask", 0))
            mid_prices.append(data.get("mid_price", 0))
        
        # Create a DataFrame
        df = pd.DataFrame({
            "Symbol": symbols,
            "Bid Price": bid_prices,
            "Mid Price": mid_prices,
            "Ask Price": ask_prices
        })
        
        # Sort by mid price (descending)
        df = df.sort_values("Mid Price", ascending=False)
        
        # Create a bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df["Symbol"],
            y=df["Bid Price"],
            name="Bid Price",
            marker_color="lightgreen"
        ))
        
        fig.add_trace(go.Bar(
            x=df["Symbol"],
            y=df["Mid Price"],
            name="Mid Price",
            marker_color="lightskyblue"
        ))
        
        fig.add_trace(go.Bar(
            x=df["Symbol"],
            y=df["Ask Price"],
            name="Ask Price",
            marker_color="lightcoral"
        ))
        
        fig.update_layout(
            title="Cryptocurrency Price Comparison",
            xaxis_title="Cryptocurrency",
            yaxis_title="Price (USD)",
            barmode="group",
            yaxis=dict(
                tickprefix="$"
            )
        )
        
        return fig
    
    def create_spread_chart(self, market_data: Dict[str, Dict[str, float]]) -> go.Figure:
        """
        Create a spread chart.
        
        Args:
            market_data: Market data
            
        Returns:
            go.Figure: Plotly figure
        """
        # Extract data for plotting
        symbols = []
        spread_percents = []
        
        for symbol, data in market_data.items():
            symbols.append(symbol.split("-")[0])  # Remove the "-USD" part
            spread_percents.append(data.get("spread_percent", 0))
        
        # Create a DataFrame
        df = pd.DataFrame({
            "Symbol": symbols,
            "Spread Percent": spread_percents
        })
        
        # Sort by spread percentage (ascending)
        df = df.sort_values("Spread Percent")
        
        # Create a horizontal bar chart
        fig = px.bar(
            df,
            x="Spread Percent",
            y="Symbol",
            orientation="h",
            title="Bid-Ask Spread Comparison",
            labels={"Spread Percent": "Spread (%)", "Symbol": "Cryptocurrency"},
            color="Spread Percent",
            color_continuous_scale="Viridis"
        )
        
        fig.update_traces(texttemplate="%{x:.2f}%", textposition="outside")
        
        return fig
    
    def create_portfolio_chart(self, portfolio_data: Dict[str, Dict[str, Any]]) -> go.Figure:
        """
        Create a portfolio distribution chart.
        
        Args:
            portfolio_data: Portfolio data
            
        Returns:
            go.Figure: Plotly figure
        """
        # Extract data for plotting
        asset_codes = []
        values = []
        
        for asset_code, data in portfolio_data.items():
            current_value = data.get("current_value", 0)
            if current_value > 0:
                asset_codes.append(asset_code)
                values.append(current_value)
        
        # Create a DataFrame
        df = pd.DataFrame({
            "Asset": asset_codes,
            "Value": values
        })
        
        # Calculate total portfolio value
        total_value = sum(values)
        
        # Create a pie chart
        fig = px.pie(
            df,
            values="Value",
            names="Asset",
            title=f"Portfolio Distribution (Total: ${total_value:.2f})",
            hover_data={"Value": ":.2f"},
            labels={"Value": "Value (USD)"}
        )
        
        fig.update_traces(textinfo="percent+label")
        
        return fig
    
    def create_price_history_chart(self, symbol: str, price_history: pd.DataFrame) -> go.Figure:
        """
        Create a price history chart.
        
        Args:
            symbol: Trading pair symbol
            price_history: Price history data
            
        Returns:
            go.Figure: Plotly figure
        """
        # Create a line chart
        fig = px.line(
            price_history,
            x="Date",
            y="Price",
            title=f"{symbol} Price History (Last 30 Days)",
            labels={"Price": "Price (USD)", "Date": "Date"}
        )
        
        fig.update_yaxes(tickprefix="$")
        
        return fig
    
    def generate_trading_signals(self, symbol: str, price_history: pd.DataFrame) -> html.Div:
        """
        Generate trading signals.
        
        Args:
            symbol: Trading pair symbol
            price_history: Price history data
            
        Returns:
            html.Div: HTML div with trading signals
        """
        from strategies.basic_strategies import MovingAverageCrossover, RelativeStrengthIndex
        
        # Apply trading strategies
        ma_strategy = MovingAverageCrossover(short_window=5, long_window=20)
        ma_signal = ma_strategy.generate_signal(price_history)
        
        rsi_strategy = RelativeStrengthIndex(window=14, oversold=30, overbought=70)
        rsi_signal = rsi_strategy.generate_signal(price_history)
        
        # Create signal indicators
        signal_colors = {
            "buy": "green",
            "sell": "red",
            "hold": "gray"
        }
        
        ma_color = signal_colors.get(ma_signal, "gray")
        rsi_color = signal_colors.get(rsi_signal, "gray")
        
        # Create HTML content
        return html.Div([
            html.Div([
                html.H4("Moving Average Crossover"),
                html.Div([
                    html.Span("Signal: "),
                    html.Span(ma_signal.upper(), style={"color": ma_color, "fontWeight": "bold"})
                ]),
                html.Div("5-day MA crosses 20-day MA")
            ], style={"marginBottom": "20px"}),
            
            html.Div([
                html.H4("Relative Strength Index (RSI)"),
                html.Div([
                    html.Span("Signal: "),
                    html.Span(rsi_signal.upper(), style={"color": rsi_color, "fontWeight": "bold"})
                ]),
                html.Div("14-day RSI with 30/70 thresholds")
            ])
        ])
    
    def run_server(self, debug=True, port=8050):
        """
        Run the Dash server.
        
        Args:
            debug: Debug mode
            port: Server port
        """
        self.app.run_server(debug=debug, port=port)


# Example usage
if __name__ == "__main__":
    import os
    import sys
    
    # Add the parent directory to the path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Import our modules
    from crypto_api import RobinhoodCryptoAPI
    
    # Try to import the API credentials from config.py
    try:
        from config import API_KEY, BASE64_PRIVATE_KEY
        
        # Initialize the API client
        api = RobinhoodCryptoAPI(API_KEY, BASE64_PRIVATE_KEY)
        
        # Create and run the dashboard
        dashboard = CryptoDashboard(api)
        dashboard.run_server()
    except ImportError:
        print("Error: API credentials not found.")
        print("Please create a config.py file with your API credentials.")
        sys.exit(1)
