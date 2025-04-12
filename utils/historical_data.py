#!/usr/bin/env python3
"""
Historical Data Fetcher

This module provides utilities for fetching real historical cryptocurrency data
from various public APIs for more accurate backtesting and analysis.
"""

import os
import json
import time
import logging
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger('historical_data')

# Constants
DEFAULT_CACHE_DIR = "data/historical"
DEFAULT_CACHE_EXPIRY = 24 * 60 * 60  # 24 hours in seconds


class HistoricalDataFetcher:
    """
    Fetches historical price data from various APIs and provides caching mechanisms
    to avoid excessive API calls.
    """
    
    def __init__(self, 
                 cache_dir: str = DEFAULT_CACHE_DIR, 
                 cache_expiry: int = DEFAULT_CACHE_EXPIRY,
                 api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize the historical data fetcher.
        
        Args:
            cache_dir: Directory to store cached data
            cache_expiry: Cache expiry time in seconds
            api_keys: Dictionary of API keys for various data sources
        """
        self.cache_dir = cache_dir
        self.cache_expiry = cache_expiry
        self.api_keys = api_keys or {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_historical_data(self, 
                            symbol: str, 
                            days: int = 90, 
                            interval: str = 'daily',
                            source: str = 'cryptocompare',
                            force_refresh: bool = False) -> pd.DataFrame:
        """
        Get historical price data for a cryptocurrency.
        
        Args:
            symbol: Trading pair symbol (e.g., "XRP-USD" or "XRP")
            days: Number of days of historical data to fetch
            interval: Data interval ('daily', 'hourly', etc.)
            source: Data source ('cryptocompare', 'coingecko', etc.)
            force_refresh: Force refresh of cached data
            
        Returns:
            DataFrame with historical price data
        """
        # Clean up symbol formatting
        base_symbol = symbol.split('-')[0] if '-' in symbol else symbol
        quote_symbol = symbol.split('-')[1] if '-' in symbol else 'USD'
        
        # Check cache first
        cache_file = self._get_cache_path(base_symbol, quote_symbol, interval, days)
        if not force_refresh and self._is_cache_valid(cache_file):
            logger.info(f"Loading cached data for {symbol} from {cache_file}")
            return pd.read_csv(cache_file, parse_dates=['Date'])
        
        # Fetch from appropriate source
        if source.lower() == 'cryptocompare':
            df = self._fetch_from_cryptocompare(base_symbol, quote_symbol, days, interval)
        elif source.lower() == 'coingecko':
            df = self._fetch_from_coingecko(base_symbol, quote_symbol, days, interval)
        else:
            raise ValueError(f"Unsupported data source: {source}")
        
        # Cache the data
        self._cache_data(df, cache_file)
        
        return df
    
    def _get_cache_path(self, base: str, quote: str, interval: str, days: int) -> str:
        """Generate the cache file path."""
        return os.path.join(self.cache_dir, f"{base}_{quote}_{interval}_{days}days.csv")
    
    def _is_cache_valid(self, cache_file: str) -> bool:
        """Check if cache file exists and is not expired."""
        if not os.path.exists(cache_file):
            return False
        
        file_age = time.time() - os.path.getmtime(cache_file)
        return file_age < self.cache_expiry
    
    def _cache_data(self, df: pd.DataFrame, cache_file: str) -> None:
        """Cache the data to a CSV file."""
        df.to_csv(cache_file, index=False)
        logger.info(f"Cached data to {cache_file}")
    
    def _fetch_from_cryptocompare(self, 
                                 base: str, 
                                 quote: str = 'USD', 
                                 days: int = 90,
                                 interval: str = 'daily') -> pd.DataFrame:
        """
        Fetch historical data from CryptoCompare API.
        
        Args:
            base: Base currency symbol (e.g., "XRP")
            quote: Quote currency symbol (e.g., "USD")
            days: Number of days of historical data
            interval: Data interval ('daily', 'hourly', etc.)
            
        Returns:
            DataFrame with historical data
        """
        logger.info(f"Fetching {interval} data for {base}/{quote} for the last {days} days from CryptoCompare")
        
        # Determine the right API endpoint and parameters
        if interval.lower() == 'daily':
            endpoint = f"https://min-api.cryptocompare.com/data/v2/histoday"
            limit = min(days, 2000)  # API limit
        elif interval.lower() == 'hourly':
            endpoint = f"https://min-api.cryptocompare.com/data/v2/histohour"
            limit = min(days * 24, 2000)  # API limit
        else:
            raise ValueError(f"Unsupported interval: {interval}")
        
        # API parameters
        params = {
            'fsym': base,
            'tsym': quote,
            'limit': limit,
            'toTs': int(datetime.now().timestamp())
        }
        
        # Add API key if available
        if 'cryptocompare' in self.api_keys:
            params['api_key'] = self.api_keys['cryptocompare']
        
        # Make the request
        response = requests.get(endpoint, params=params)
        
        if response.status_code != 200:
            logger.error(f"Error fetching data: {response.text}")
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        data = response.json()
        
        if data['Response'] != 'Success':
            logger.error(f"API error: {data['Message']}")
            raise Exception(f"API error: {data['Message']}")
        
        # Convert to DataFrame
        df = pd.DataFrame(data['Data']['Data'])
        
        # Convert timestamp to datetime
        df['Date'] = pd.to_datetime(df['time'], unit='s')
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Price',  # Use close as the main price
            'volumeto': 'Volume'
        })
        
        # Select and reorder columns
        columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Volume']
        return df[columns]
    
    def _fetch_from_coingecko(self, 
                             base: str, 
                             quote: str = 'usd', 
                             days: int = 90, 
                             interval: str = 'daily') -> pd.DataFrame:
        """
        Fetch historical data from CoinGecko API.
        
        Args:
            base: Base currency symbol (e.g., "xrp")
            quote: Quote currency symbol (e.g., "usd")
            days: Number of days of historical data
            interval: Data interval ('daily', 'hourly', etc.)
            
        Returns:
            DataFrame with historical data
        """
        logger.info(f"Fetching {interval} data for {base}/{quote} for the last {days} days from CoinGecko")
        
        # CoinGecko uses lowercase symbols and requires the coin ID
        base = base.lower()
        quote = quote.lower()
        
        # Get coin ID if necessary
        coin_id = self._get_coingecko_coin_id(base)
        
        # Determine the right API parameters
        if interval.lower() == 'daily':
            days_param = min(days, 365)  # Max days supported
        elif interval.lower() == 'hourly':
            days_param = min(days, 90)  # Max days for hourly
        else:
            raise ValueError(f"Unsupported interval: {interval}")
        
        # API endpoint
        endpoint = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        
        # API parameters
        params = {
            'vs_currency': quote,
            'days': days_param,
            'interval': 'daily' if interval.lower() == 'daily' else 'hourly'
        }
        
        # Make the request
        response = requests.get(endpoint, params=params)
        
        if response.status_code != 200:
            logger.error(f"Error fetching data: {response.text}")
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        data = response.json()
        
        # Extract price and volume data
        prices = data['prices']
        volumes = data['total_volumes']
        
        # Ensure we have the same number of price and volume entries
        min_len = min(len(prices), len(volumes))
        prices = prices[:min_len]
        volumes = volumes[:min_len]
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': [p[0] for p in prices],
            'Price': [p[1] for p in prices],
            'Volume': [v[1] for v in volumes]
        })
        
        # Convert timestamp to datetime
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Forward-fill missing values
        df = df.fillna(method='ffill')
        
        # Select columns
        columns = ['Date', 'Price', 'Volume']
        return df[columns]
    
    def _get_coingecko_coin_id(self, symbol: str) -> str:
        """Get CoinGecko coin ID from symbol."""
        # Common mappings for popular coins
        mappings = {
            'btc': 'bitcoin',
            'eth': 'ethereum',
            'xrp': 'ripple',
            'ada': 'cardano',
            'sol': 'solana',
            'doge': 'dogecoin'
        }
        
        if symbol.lower() in mappings:
            return mappings[symbol.lower()]
        
        # For other coins, we need to fetch the coin list
        # This is a simplified version - in production, you'd want to cache this
        try:
            response = requests.get("https://api.coingecko.com/api/v3/coins/list")
            if response.status_code == 200:
                coins = response.json()
                for coin in coins:
                    if coin['symbol'].lower() == symbol.lower():
                        return coin['id']
            
            raise ValueError(f"Could not find CoinGecko ID for symbol: {symbol}")
        except Exception as e:
            logger.error(f"Error fetching CoinGecko coin list: {e}")
            # Fallback to using the symbol as the ID
            return symbol.lower()


def get_fear_greed_index(days: Optional[int] = None) -> pd.DataFrame:
    """
    Get the Fear & Greed Index for cryptocurrency market sentiment.
    
    Args:
        days: Number of days of historical data, or None for current value only
        
    Returns:
        DataFrame with date and index value
    """
    logger.info(f"Fetching Fear & Greed Index for the past {days} days" if days else "Fetching current Fear & Greed Index")
    
    # Use the Alternative.me API
    if days:
        url = f"https://api.alternative.me/fng/?limit={days}&format=json"
    else:
        url = "https://api.alternative.me/fng/?limit=1&format=json"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            logger.error(f"API request failed: {response.text}")
            return pd.DataFrame({'Date': [datetime.now()], 'Value': [50], 'Classification': ['Neutral']})
        
        data = response.json()
        
        # Extract relevant fields
        records = []
        for item in data['data']:
            timestamp = int(item['timestamp'])
            value = int(item['value'])
            classification = item['value_classification']
            dt = datetime.fromtimestamp(timestamp)
            records.append({
                'Date': dt,
                'Value': value,
                'Classification': classification
            })
        
        # Create DataFrame
        df = pd.DataFrame(records)
        return df
    
    except Exception as e:
        logger.error(f"Error fetching Fear & Greed Index: {e}")
        # Return a default neutral value on error
        return pd.DataFrame({'Date': [datetime.now()], 'Value': [50], 'Classification': ['Neutral']})


def get_normalized_sentiment(days: int = 30) -> float:
    """
    Get a normalized market sentiment score from -1 to 1.
    
    Args:
        days: Number of days to average
        
    Returns:
        Sentiment score from -1 (extreme fear) to 1 (extreme greed)
    """
    try:
        # Get Fear & Greed data
        fg_data = get_fear_greed_index(days)
        
        # Calculate average
        avg_value = fg_data['Value'].mean()
        
        # Normalize to -1 to 1 scale (0 to 100 becomes -1 to 1)
        normalized = (avg_value - 50) / 50
        
        return normalized
    
    except Exception as e:
        logger.error(f"Error calculating normalized sentiment: {e}")
        return 0.0  # Neutral value on error


def main():
    """Test the historical data fetcher."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize fetcher
    fetcher = HistoricalDataFetcher()
    
    # Test with XRP
    xrp_data = fetcher.get_historical_data('XRP', days=30, interval='daily')
    print(f"XRP data shape: {xrp_data.shape}")
    print(xrp_data.head())
    
    # Test Fear & Greed Index
    fg_data = get_fear_greed_index(days=30)
    print(f"Fear & Greed data shape: {fg_data.shape}")
    print(fg_data.head())
    
    # Test normalized sentiment
    sentiment = get_normalized_sentiment()
    print(f"Normalized sentiment: {sentiment}")


if __name__ == "__main__":
    main()
