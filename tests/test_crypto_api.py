#!/usr/bin/env python3
"""
Tests for the Robinhood Crypto API client
"""

import unittest
import json
import os
import sys
import responses
from unittest.mock import patch, MagicMock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to test
from crypto_api import RobinhoodCryptoAPI

# Sample API responses for testing
SAMPLE_RESPONSES = {
    "account": {
        "account_number": "test_account_123",
        "status": "active",
        "buying_power": "1000.00",
        "buying_power_currency": "USD"
    },
    "trading_pairs": {
        "results": [
            {
                "symbol": "BTC-USD",
                "min_order_size": "0.000001",
                "max_order_size": "5.0",
                "status": "active"
            },
            {
                "symbol": "ETH-USD",
                "min_order_size": "0.0001",
                "max_order_size": "100.0",
                "status": "active"
            }
        ]
    },
    "best_bid_ask": {
        "results": [
            {
                "symbol": "BTC-USD",
                "bid_price": "60000.00",
                "ask_price": "60050.00",
                "bid_size": "0.5",
                "ask_size": "0.5"
            }
        ]
    },
    "holdings": {
        "results": [
            {
                "asset_code": "BTC",
                "quantity": "0.5",
                "cost_basis": "30000.00"
            },
            {
                "asset_code": "ETH",
                "quantity": "5.0",
                "cost_basis": "15000.00"
            }
        ]
    },
    "orders": {
        "results": [
            {
                "id": "order_123",
                "account_number": "test_account_123",
                "symbol": "BTC-USD",
                "client_order_id": "client_order_123",
                "side": "buy",
                "type": "market",
                "state": "filled",
                "filled_asset_quantity": "0.1",
                "average_price": "60000.00",
                "created_at": "2025-04-01T10:00:00Z",
                "updated_at": "2025-04-01T10:01:00Z"
            }
        ]
    }
}

class TestRobinhoodCryptoAPI(unittest.TestCase):
    """Test cases for the RobinhoodCryptoAPI class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test_api_key"
        self.private_key_base64 = "xQnTJVeQLmw1/Mg2YimEViSpw/SdJcgNXZ5kQkAXNPU="
        
        # Create a patcher for the nacl.signing.SigningKey
        self.signing_key_patcher = patch('nacl.signing.SigningKey')
        self.mock_signing_key = self.signing_key_patcher.start()
        
        # Mock the signature for testing
        self.mock_signing_instance = MagicMock()
        self.mock_signature = MagicMock()
        self.mock_signature.signature = b'signature'
        self.mock_signing_instance.sign.return_value = self.mock_signature
        self.mock_signing_key.return_value = self.mock_signing_instance
        
        # Create the API client with mocked dependencies
        self.api = RobinhoodCryptoAPI(self.api_key, self.private_key_base64)
        
        # Mock timestamp for deterministic testing
        self.timestamp_patcher = patch.object(RobinhoodCryptoAPI, '_get_current_timestamp')
        self.mock_timestamp = self.timestamp_patcher.start()
        self.mock_timestamp.return_value = 1617258000
    
    def tearDown(self):
        """Tear down test fixtures"""
        self.signing_key_patcher.stop()
        self.timestamp_patcher.stop()
    
    def test_initialization(self):
        """Test API client initialization"""
        self.assertEqual(self.api.api_key, self.api_key)
        self.assertEqual(self.api.base_url, "https://trading.robinhood.com")
    
    def test_get_query_params(self):
        """Test query parameter generation"""
        # Test with single parameter
        params = self.api.get_query_params("symbol", "BTC-USD")
        self.assertEqual(params, "?symbol=BTC-USD")
        
        # Test with multiple parameters
        params = self.api.get_query_params("symbol", "BTC-USD", "ETH-USD")
        self.assertEqual(params, "?symbol=BTC-USD&symbol=ETH-USD")
        
        # Test with no parameters
        params = self.api.get_query_params("symbol")
        self.assertEqual(params, "")
    
    def test_get_authorization_header(self):
        """Test authorization header generation"""
        headers = self.api.get_authorization_header("GET", "/api/endpoint", "", 1617258000)
        
        # Check all required headers are present
        self.assertIn("x-api-key", headers)
        self.assertIn("x-signature", headers)
        self.assertIn("x-timestamp", headers)
        
        # Check header values
        self.assertEqual(headers["x-api-key"], self.api_key)
        self.assertEqual(headers["x-timestamp"], "1617258000")
    
    @responses.activate
    def test_get_account(self):
        """Test get_account endpoint"""
        # Mock the API response
        responses.add(
            responses.GET,
            "https://trading.robinhood.com/api/v1/crypto/trading/accounts/",
            json=SAMPLE_RESPONSES["account"],
            status=200
        )
        
        # Call the API method
        result = self.api.get_account()
        
        # Check the result
        self.assertEqual(result, SAMPLE_RESPONSES["account"])
        self.assertEqual(result["account_number"], "test_account_123")
    
    @responses.activate
    def test_get_trading_pairs(self):
        """Test get_trading_pairs endpoint"""
        # Mock the API response
        responses.add(
            responses.GET,
            "https://trading.robinhood.com/api/v1/crypto/trading/trading_pairs/",
            json=SAMPLE_RESPONSES["trading_pairs"],
            status=200
        )
        
        # Call the API method
        result = self.api.get_trading_pairs()
        
        # Check the result
        self.assertEqual(result, SAMPLE_RESPONSES["trading_pairs"])
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["symbol"], "BTC-USD")
    
    @responses.activate
    def test_get_best_bid_ask(self):
        """Test get_best_bid_ask endpoint"""
        # Mock the API response
        responses.add(
            responses.GET,
            "https://trading.robinhood.com/api/v1/crypto/marketdata/best_bid_ask/?symbol=BTC-USD",
            json=SAMPLE_RESPONSES["best_bid_ask"],
            status=200
        )
        
        # Call the API method
        result = self.api.get_best_bid_ask("BTC-USD")
        
        # Check the result
        self.assertEqual(result, SAMPLE_RESPONSES["best_bid_ask"])
        self.assertEqual(result["results"][0]["bid_price"], "60000.00")
    
    @responses.activate
    def test_get_holdings(self):
        """Test get_holdings endpoint"""
        # Mock the API response
        responses.add(
            responses.GET,
            "https://trading.robinhood.com/api/v1/crypto/trading/holdings/",
            json=SAMPLE_RESPONSES["holdings"],
            status=200
        )
        
        # Call the API method
        result = self.api.get_holdings()
        
        # Check the result
        self.assertEqual(result, SAMPLE_RESPONSES["holdings"])
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["asset_code"], "BTC")
        
        # Test with specific asset code
        responses.add(
            responses.GET,
            "https://trading.robinhood.com/api/v1/crypto/trading/holdings/?asset_code=BTC",
            json={"results": [SAMPLE_RESPONSES["holdings"]["results"][0]]},
            status=200
        )
        
        result = self.api.get_holdings("BTC")
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["asset_code"], "BTC")
    
    @responses.activate
    def test_get_orders(self):
        """Test get_orders endpoint"""
        # Mock the API response
        responses.add(
            responses.GET,
            "https://trading.robinhood.com/api/v1/crypto/trading/orders/",
            json=SAMPLE_RESPONSES["orders"],
            status=200
        )
        
        # Call the API method
        result = self.api.get_orders()
        
        # Check the result
        self.assertEqual(result, SAMPLE_RESPONSES["orders"])
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["symbol"], "BTC-USD")
    
    @responses.activate
    def test_error_handling(self):
        """Test error handling in API requests"""
        # Mock a failed API response
        responses.add(
            responses.GET,
            "https://trading.robinhood.com/api/v1/crypto/trading/accounts/",
            json={"error": "Unauthorized"},
            status=401
        )
        
        # Call the API method and expect it to handle the error
        result = self.api.get_account()
        
        # Should return None on error
        self.assertIsNone(result)
    
    @patch('requests.get')
    def test_connection_error(self, mock_get):
        """Test handling of connection errors"""
        # Mock a connection error
        mock_get.side_effect = Exception("Connection error")
        
        # Call the API method and expect it to handle the error
        result = self.api.get_account()
        
        # Should return None on error
        self.assertIsNone(result)
    
    @responses.activate
    def test_place_market_order(self):
        """Test place_market_order endpoint"""
        # Mock the API response
        responses.add(
            responses.POST,
            "https://trading.robinhood.com/api/v1/crypto/trading/orders/",
            json={"id": "new_order_123"},
            status=201
        )
        
        # Mock UUID generation
        with patch('uuid.uuid4', return_value="test-uuid"):
            # Call the API method
            result = self.api.place_market_order("BTC-USD", "buy", "0.1")
            
            # Check the result
            self.assertEqual(result["id"], "new_order_123")

if __name__ == '__main__':
    unittest.main()
