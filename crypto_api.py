import base64
import datetime
import json
from typing import Any, Dict, List, Optional, Union
import uuid
import requests
from nacl.signing import SigningKey

class RobinhoodCryptoAPI:
    """
    A client for interacting with the Robinhood Crypto API.
    This class handles authentication and provides methods for various API endpoints.
    """
    
    def __init__(self, api_key: str, base64_private_key: str):
        """
        Initialize the Robinhood Crypto API client.
        
        Args:
            api_key: Your Robinhood API key
            base64_private_key: Your base64-encoded private key
        """
        self.api_key = api_key
        private_key_seed = base64.b64decode(base64_private_key)
        self.private_key = SigningKey(private_key_seed)
        self.base_url = "https://trading.robinhood.com"

    @staticmethod
    def _get_current_timestamp() -> int:
        """Get the current UTC timestamp in seconds."""
        return int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp())

    @staticmethod
    def get_query_params(key: str, *args: Optional[str]) -> str:
        """
        Create query parameters for API requests.
        
        Args:
            key: The parameter key
            *args: Parameter values
            
        Returns:
            A formatted query string
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
            Authorization headers
        """
        message_to_sign = f"{self.api_key}{timestamp}{path}{method}{body}"
        signed = self.private_key.sign(message_to_sign.encode("utf-8"))

        return {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
            "x-timestamp": str(timestamp),
            "Content-Type": "application/json; charset=utf-8"
        }

    def make_api_request(self, method: str, path: str, body: str = "") -> Any:
        """
        Make an API request to the Robinhood Crypto API.
        
        Args:
            method: HTTP method
            path: API endpoint path
            body: Request body
            
        Returns:
            API response
        """
        timestamp = self._get_current_timestamp()
        headers = self.get_authorization_header(method, path, body, timestamp)
        url = self.base_url + path

        try:
            response = {}
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=json.loads(body) if body else {}, timeout=10)
            
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error making API request: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Response text: {e.response.text}")
            return None
        except Exception as e:
            # Add generic exception handling for all other exceptions
            print(f"Error making API request: {e}")
            return None

    # Account endpoints
    def get_account(self) -> Dict[str, Any]:
        """
        Get the user's Robinhood Crypto account details.
        
        Returns:
            Account information
        """
        path = "/api/v1/crypto/trading/accounts/"
        return self.make_api_request("GET", path)
    
    # Market data endpoints
    def get_trading_pairs(self, *symbols: Optional[str]) -> Dict[str, Any]:
        """
        Get information about trading pairs.
        
        Args:
            *symbols: Trading pair symbols (e.g., "BTC-USD", "ETH-USD")
            
        Returns:
            Trading pair information
        """
        query_params = self.get_query_params("symbol", *symbols)
        path = f"/api/v1/crypto/trading/trading_pairs/{query_params}"
        return self.make_api_request("GET", path)

    def get_best_bid_ask(self, *symbols: Optional[str]) -> Dict[str, Any]:
        """
        Get the best bid and ask prices for specified trading pairs.
        
        Args:
            *symbols: Trading pair symbols (e.g., "BTC-USD", "ETH-USD")
            
        Returns:
            Bid and ask price information
        """
        query_params = self.get_query_params("symbol", *symbols)
        path = f"/api/v1/crypto/marketdata/best_bid_ask/{query_params}"
        return self.make_api_request("GET", path)

    def get_estimated_price(self, symbol: str, side: str, quantity: str) -> Dict[str, Any]:
        """
        Get an estimated price for a potential order.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC-USD")
            side: Order side ("bid" or "ask")
            quantity: Order quantity
            
        Returns:
            Estimated price information
        """
        path = f"/api/v1/crypto/marketdata/estimated_price/?symbol={symbol}&side={side}&quantity={quantity}"
        return self.make_api_request("GET", path)
    
    # Portfolio endpoints
    def get_holdings(self, *asset_codes: Optional[str]) -> Dict[str, Any]:
        """
        Get the user's crypto holdings.
        
        Args:
            *asset_codes: Asset codes (e.g., "BTC", "ETH")
            
        Returns:
            Holdings information
        """
        query_params = self.get_query_params("asset_code", *asset_codes)
        path = f"/api/v1/crypto/trading/holdings/{query_params}"
        return self.make_api_request("GET", path)
    
    # Order endpoints
    def get_orders(self) -> Dict[str, Any]:
        """
        Get the user's crypto orders.
        
        Returns:
            Order information
        """
        path = "/api/v1/crypto/trading/orders/"
        return self.make_api_request("GET", path)

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get information about a specific order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order information
        """
        path = f"/api/v1/crypto/trading/orders/{order_id}/"
        return self.make_api_request("GET", path)

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
            Order information
        """
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
        return self.make_api_request("POST", path, json.dumps(body))

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
            Order information
        """
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
        return self.make_api_request("POST", path, json.dumps(body))

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Cancellation information
        """
        path = f"/api/v1/crypto/trading/orders/{order_id}/cancel/"
        return self.make_api_request("POST", path)