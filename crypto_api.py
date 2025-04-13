"""
Robinhood Crypto API Client

This module provides a client class for interacting with the Robinhood Crypto API,
handling authentication, making API requests, and providing methods for various API endpoints.

Classes:
    RobinhoodCryptoAPI: Main client class for the Robinhood Crypto API.
"""

import base64
import datetime
import json
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import uuid
import requests
from nacl.signing import SigningKey

# Configure logging
logger = logging.getLogger(__name__)

class RobinhoodCryptoAPI:
    """
    A client for interacting with the Robinhood Crypto API.
    
    This class handles authentication and provides methods for various API endpoints
    related to cryptocurrency trading on Robinhood.
    """
    
    def __init__(self, api_key: str, base64_private_key: str):
        """
        Initialize the Robinhood Crypto API client.
        
        Args:
            api_key: Your Robinhood API key
            base64_private_key: Your base64-encoded private key
        
        Raises:
            ValueError: If the private key is invalid
            Exception: If initialization fails
        """
        try:
            self.api_key = api_key
            private_key_seed = base64.b64decode(base64_private_key)
            self.private_key = SigningKey(private_key_seed)
            self.base_url = "https://trading.robinhood.com"
            logger.info("RobinhoodCryptoAPI client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RobinhoodCryptoAPI: {e}")
            raise

    @staticmethod
    def _get_current_timestamp() -> int:
        """
        Get the current UTC timestamp in seconds.
        
        Returns:
            int: Current UTC timestamp
        """
        return int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp())

    @staticmethod
    def get_query_params(key: str, *args: Optional[str]) -> str:
        """
        Create query parameters for API requests.
        
        Args:
            key: The parameter key
            *args: Parameter values
            
        Returns:
            str: A formatted query string
        """
        if not args:
            return ""

        params = []
        for arg in args:
            params.append(f"{key}={arg}")

        return "?" + "&".join(params)

    def get_authorization_header(
            self, method: str, path: str, body: str, timestamp: int
        ) -> Dict[str, str]:
        """
        Get authorization headers for API requests.
        
        Args:
            method: HTTP method
            path: API endpoint path
            body: Request body
            timestamp: Current timestamp
            
        Returns:
            Dict[str, str]: Authorization headers
        
        Raises:
            Exception: If signing fails
        """
        try:
            message_to_sign = f"{self.api_key}{timestamp}{path}{method}{body}"
            signed = self.private_key.sign(message_to_sign.encode("utf-8"))

            return {
                "x-api-key": self.api_key,
                "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
                "x-timestamp": str(timestamp),
                "Content-Type": "application/json; charset=utf-8"
            }
        except Exception as e:
            logger.error(f"Error creating authorization headers: {e}")
            raise

    def make_api_request(self, method: str, path: str, body: str = "") -> Any:
        """
        Make an API request to the Robinhood Crypto API.
        
        Args:
            method: HTTP method
            path: API endpoint path
            body: Request body
            
        Returns:
            Any: API response or None if request fails
        
        Raises:
            requests.RequestException: For request errors
            ValueError: For invalid response data
            Exception: For other errors
        """
        timestamp = self._get_current_timestamp()
        headers = self.get_authorization_header(method, path, body, timestamp)
        url = self.base_url + path

        try:
            response = {}
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            elif method == "POST":
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=json.loads(body) if body else {}, 
                    timeout=10
                )
            
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API request error: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                logger.error(f"Response text: {e.response.text}")
            return None
        except ValueError as e:
            logger.error(f"Error parsing API response: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error making API request: {e}")
            return None

    # Account endpoints
    def get_account(self) -> Dict[str, Any]:
        """
        Get the user's Robinhood Crypto account details.
        
        Returns:
            Dict[str, Any]: Account information or empty dict if request fails
        """
        path = "/api/v1/crypto/trading/accounts/"
        result = self.make_api_request("GET", path)
        if result is None:
            logger.warning("Failed to get account information")
            return {}
        return result
    
    # Market data endpoints
    def get_trading_pairs(self, *symbols: Optional[str]) -> Dict[str, Any]:
        """
        Get information about trading pairs.
        
        Args:
            *symbols: Trading pair symbols (e.g., "BTC-USD", "ETH-USD")
            
        Returns:
            Dict[str, Any]: Trading pair information or empty dict if request fails
        """
        query_params = self.get_query_params("symbol", *symbols)
        path = f"/api/v1/crypto/trading/trading_pairs/{query_params}"
        result = self.make_api_request("GET", path)
        if result is None:
            logger.warning(f"Failed to get trading pairs for {symbols}")
            return {}
        return result

    def get_best_bid_ask(self, *symbols: Optional[str]) -> Dict[str, Any]:
        """
        Get the best bid and ask prices for specified trading pairs.
        
        Args:
            *symbols: Trading pair symbols (e.g., "BTC-USD", "ETH-USD")
            
        Returns:
            Dict[str, Any]: Bid and ask price information or empty dict if request fails
        """
        query_params = self.get_query_params("symbol", *symbols)
        path = f"/api/v1/crypto/marketdata/best_bid_ask/{query_params}"
        result = self.make_api_request("GET", path)
        if result is None:
            logger.warning(f"Failed to get best bid/ask for {symbols}")
            return {}
        return result

    def get_estimated_price(self, symbol: str, side: str, quantity: str) -> Dict[str, Any]:
        """
        Get an estimated price for a potential order.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC-USD")
            side: Order side ("bid" or "ask")
            quantity: Order quantity
            
        Returns:
            Dict[str, Any]: Estimated price information or empty dict if request fails
        """
        path = f"/api/v1/crypto/marketdata/estimated_price/?symbol={symbol}&side={side}&quantity={quantity}"
        result = self.make_api_request("GET", path)
        if result is None:
            logger.warning(f"Failed to get estimated price for {symbol}")
            return {}
        return result
    
    # Portfolio endpoints
    def get_holdings(self, *asset_codes: Optional[str]) -> Dict[str, Any]:
        """
        Get the user's crypto holdings.
        
        Args:
            *asset_codes: Asset codes (e.g., "BTC", "ETH")
            
        Returns:
            Dict[str, Any]: Holdings information or empty dict if request fails
        """
        query_params = self.get_query_params("asset_code", *asset_codes)
        path = f"/api/v1/crypto/trading/holdings/{query_params}"
        result = self.make_api_request("GET", path)
        if result is None:
            logger.warning("Failed to get holdings")
            return {}
        return result
    
    # Order endpoints
    def get_orders(self) -> Dict[str, Any]:
        """
        Get the user's crypto orders.
        
        Returns:
            Dict[str, Any]: Order information or empty dict if request fails
        """
        path = "/api/v1/crypto/trading/orders/"
        result = self.make_api_request("GET", path)
        if result is None:
            logger.warning("Failed to get orders")
            return {}
        return result

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get information about a specific order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict[str, Any]: Order information or empty dict if request fails
        """
        path = f"/api/v1/crypto/trading/orders/{order_id}/"
        result = self.make_api_request("GET", path)
        if result is None:
            logger.warning(f"Failed to get order {order_id}")
            return {}
        return result

    def place_market_order(
            self,
            symbol: str,
            side: str,
            asset_quantity: str
        ) -> Dict[str, Any]:
        """
        Place a market order.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC-USD")
            side: Order side ("buy" or "sell")
            asset_quantity: Order quantity
            
        Returns:
            Dict[str, Any]: Order information or empty dict if request fails
        
        Raises:
            ValueError: If input parameters are invalid
        """
        if side not in ["buy", "sell"]:
            raise ValueError("Side must be 'buy' or 'sell'")
            
        client_order_id = str(uuid.uuid4())
        order_config = {"asset_quantity": asset_quantity}
        
        body = {
            "client_order_id": client_order_id,
            "side": side,
            "type": "market",
            "symbol": symbol,
            "market_order_config": order_config,
        }
        
        path = "/api/v1/crypto/trading/orders/"
        result = self.make_api_request("POST", path, json.dumps(body))
        if result is None:
            logger.warning(f"Failed to place {side} market order for {symbol}")
            return {}
        return result

    def place_limit_order(
            self,
            symbol: str,
            side: str,
            asset_quantity: str,
            limit_price: str,
            time_in_force: str = "gtc"
        ) -> Dict[str, Any]:
        """
        Place a limit order.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC-USD")
            side: Order side ("buy" or "sell")
            asset_quantity: Order quantity
            limit_price: Limit price
            time_in_force: Time in force (default: "gtc")
            
        Returns:
            Dict[str, Any]: Order information or empty dict if request fails
            
        Raises:
            ValueError: If input parameters are invalid
        """
        if side not in ["buy", "sell"]:
            raise ValueError("Side must be 'buy' or 'sell'")
            
        if time_in_force not in ["gtc", "ioc", "fok"]:
            raise ValueError("Time in force must be 'gtc', 'ioc', or 'fok'")
            
        client_order_id = str(uuid.uuid4())
        order_config = {
            "asset_quantity": asset_quantity,
            "limit_price": limit_price,
            "time_in_force": time_in_force
        }
        
        body = {
            "client_order_id": client_order_id,
            "side": side,
            "type": "limit",
            "symbol": symbol,
            "limit_order_config": order_config,
        }
        
        path = "/api/v1/crypto/trading/orders/"
        result = self.make_api_request("POST", path, json.dumps(body))
        if result is None:
            logger.warning(f"Failed to place {side} limit order for {symbol}")
            return {}
        return result

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict[str, Any]: Cancellation information or empty dict if request fails
        """
        path = f"/api/v1/crypto/trading/orders/{order_id}/cancel/"
        result = self.make_api_request("POST", path)
        if result is None:
            logger.warning(f"Failed to cancel order {order_id}")
            return {}
        return result
