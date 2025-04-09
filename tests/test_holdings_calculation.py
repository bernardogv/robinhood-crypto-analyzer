#!/usr/bin/env python3
"""
Tests for the portfolio holdings calculation
"""

import unittest
import json
import os
import sys
from unittest.mock import patch, MagicMock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Sample data for testing
SAMPLE_HOLDINGS = {
    "results": [
        {
            "asset_code": "BTC",
            "quantity": "0.0",  # Zero quantity to test workaround
            "cost_basis": "30000.00"
        },
        {
            "asset_code": "ETH",
            "quantity": "5.0",
            "cost_basis": "15000.00"
        },
        {
            "asset_code": "XRP",
            "quantity": "0.0",  # Zero quantity to test workaround
            "cost_basis": "5000.00"
        }
    ]
}

SAMPLE_PRICE_DATA = {
    "BTC-USD": {
        "results": [
            {
                "symbol": "BTC-USD",
                "bid_price": "60000.00",
                "ask_price": "60100.00"
            }
        ]
    },
    "ETH-USD": {
        "results": [
            {
                "symbol": "ETH-USD",
                "bid_price": "3000.00",
                "ask_price": "3020.00"
            }
        ]
    },
    "XRP-USD": {
        "results": [
            {
                "symbol": "XRP-USD",
                "bid_price": "1.80",
                "ask_price": "1.82"
            }
        ]
    }
}

SAMPLE_ORDERS = {
    "results": [
        # BTC buy orders
        {
            "symbol": "BTC-USD",
            "side": "buy",
            "state": "filled",
            "filled_asset_quantity": "0.25",
            "average_price": "55000.00",
            "updated_at": "2025-04-01T10:00:00Z"
        },
        {
            "symbol": "BTC-USD",
            "side": "buy",
            "state": "filled",
            "filled_asset_quantity": "0.1",
            "average_price": "58000.00",
            "updated_at": "2025-04-02T10:00:00Z"
        },
        # BTC sell orders
        {
            "symbol": "BTC-USD",
            "side": "sell",
            "state": "filled",
            "filled_asset_quantity": "0.05",
            "average_price": "59000.00",
            "updated_at": "2025-04-03T10:00:00Z"
        },
        # XRP buy orders
        {
            "symbol": "XRP-USD",
            "side": "buy",
            "state": "filled",
            "filled_asset_quantity": "1000.0",
            "average_price": "1.70",
            "updated_at": "2025-04-01T11:00:00Z"
        },
        {
            "symbol": "XRP-USD",
            "side": "buy",
            "state": "filled",
            "filled_asset_quantity": "500.0",
            "average_price": "1.75",
            "updated_at": "2025-04-02T11:00:00Z"
        },
        # XRP sell orders
        {
            "symbol": "XRP-USD",
            "side": "sell",
            "state": "filled",
            "filled_asset_quantity": "200.0",
            "average_price": "1.78",
            "updated_at": "2025-04-03T11:00:00Z"
        },
        # Order with different state (should be ignored)
        {
            "symbol": "BTC-USD",
            "side": "buy",
            "state": "canceled",
            "filled_asset_quantity": "0.1",
            "average_price": "57000.00",
            "updated_at": "2025-04-04T10:00:00Z"
        }
    ]
}

class TestHoldingsCalculation(unittest.TestCase):
    """Test cases for portfolio holdings calculation"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock the API client
        self.api_patcher = patch('crypto_api.RobinhoodCryptoAPI')
        self.mock_api_class = self.api_patcher.start()
        self.mock_api = self.mock_api_class.return_value
        
        # Configure the mock API to return our sample data
        self.mock_api.get_holdings.return_value = SAMPLE_HOLDINGS
        self.mock_api.get_orders.return_value = SAMPLE_ORDERS
        
        # Configure the get_best_bid_ask method to return appropriate price data
        def mock_get_best_bid_ask(symbol):
            return SAMPLE_PRICE_DATA.get(symbol, {"results": []})
        
        self.mock_api.get_best_bid_ask.side_effect = mock_get_best_bid_ask
    
    def tearDown(self):
        """Tear down test fixtures"""
        self.api_patcher.stop()
    
    def test_calculate_holdings_from_api(self):
        """Test holdings calculation from direct API data"""
        # Import the function here to avoid circular imports
        from examples.portfolio_analysis import main
        
        # Mock to capture the output
        with patch('builtins.print') as mock_print:
            # Run the analysis
            with patch('sys.exit') as mock_exit:  # Prevent actual exit
                with patch('matplotlib.pyplot.savefig'):  # Prevent saving figures
                    with patch('matplotlib.pyplot.close'):  # Prevent closing figures
                        main()
            
            # Check if ETH quantity was processed correctly from direct API
            found_eth_output = False
            for call in mock_print.call_args_list:
                if isinstance(call.args[0], str) and 'ETH: 5.0 @' in call.args[0]:
                    found_eth_output = True
                    break
            self.assertTrue(found_eth_output, "ETH holdings not processed correctly")
    
    def test_calculate_holdings_from_orders(self):
        """Test holdings calculation from orders when API returns zero"""
        # Run a manual calculation to verify our expected values
        expected_btc_quantity = 0.25 + 0.1 - 0.05  # 0.3 BTC
        expected_xrp_quantity = 1000.0 + 500.0 - 200.0  # 1300 XRP
        
        # Import the function here to avoid circular imports
        from examples.portfolio_analysis import main
        
        # Mock to capture the output
        with patch('builtins.print') as mock_print:
            # Run the analysis
            with patch('sys.exit') as mock_exit:  # Prevent actual exit
                with patch('matplotlib.pyplot.savefig'):  # Prevent saving figures
                    with patch('matplotlib.pyplot.close'):  # Prevent closing figures
                        main()
            
            # Check if BTC quantity was calculated correctly from orders
            found_btc_expected = False
            found_xrp_expected = False
            
            for call in mock_print.call_args_list:
                if isinstance(call.args[0], str):
                    if 'BTC: ' in call.args[0] and '0.3 @' in call.args[0]:
                        found_btc_expected = True
                    elif 'XRP: ' in call.args[0] and '1300.0 @' in call.args[0]:
                        found_xrp_expected = True
            
            self.assertTrue(found_btc_expected, "BTC holdings not calculated correctly from orders")
            self.assertTrue(found_xrp_expected, "XRP holdings not calculated correctly from orders")

    def test_portfolio_value_calculation(self):
        """Test total portfolio value calculation"""
        # Run a manual calculation to verify our expected values
        expected_btc_quantity = 0.3  # From orders
        expected_eth_quantity = 5.0  # From API
        expected_xrp_quantity = 1300.0  # From orders
        
        expected_btc_price = (60000.00 + 60100.00) / 2  # Average of bid/ask
        expected_eth_price = (3000.00 + 3020.00) / 2
        expected_xrp_price = (1.80 + 1.82) / 2
        
        expected_btc_value = expected_btc_quantity * expected_btc_price
        expected_eth_value = expected_eth_quantity * expected_eth_price
        expected_xrp_value = expected_xrp_quantity * expected_xrp_price
        
        expected_total_value = expected_btc_value + expected_eth_value + expected_xrp_value
        
        # Import the function here to avoid circular imports
        from examples.portfolio_analysis import main
        
        # Mock to capture the output
        with patch('builtins.print') as mock_print:
            # Run the analysis
            with patch('sys.exit') as mock_exit:  # Prevent actual exit
                with patch('matplotlib.pyplot.savefig'):  # Prevent saving figures
                    with patch('matplotlib.pyplot.close'):  # Prevent closing figures
                        main()
            
            # Check if the total portfolio value is calculated correctly
            found_total_value = False
            
            for call in mock_print.call_args_list:
                if isinstance(call.args[0], str) and 'Total Portfolio Value:' in call.args[0]:
                    value_str = call.args[0].split('$')[1].strip()
                    try:
                        value = float(value_str)
                        # Allow for small float rounding differences
                        if abs(value - expected_total_value) < 0.01:
                            found_total_value = True
                    except ValueError:
                        pass
            
            self.assertTrue(found_total_value, "Total portfolio value not calculated correctly")

    def test_portfolio_allocation_calculation(self):
        """Test portfolio allocation percentage calculation"""
        # Calculate expected allocation percentages
        expected_btc_quantity = 0.3
        expected_eth_quantity = 5.0
        expected_xrp_quantity = 1300.0
        
        expected_btc_price = (60000.00 + 60100.00) / 2
        expected_eth_price = (3000.00 + 3020.00) / 2
        expected_xrp_price = (1.80 + 1.82) / 2
        
        expected_btc_value = expected_btc_quantity * expected_btc_price
        expected_eth_value = expected_eth_quantity * expected_eth_price
        expected_xrp_value = expected_xrp_quantity * expected_xrp_price
        
        expected_total_value = expected_btc_value + expected_eth_value + expected_xrp_value
        
        expected_btc_allocation = (expected_btc_value / expected_total_value) * 100
        expected_eth_allocation = (expected_eth_value / expected_total_value) * 100
        expected_xrp_allocation = (expected_xrp_value / expected_total_value) * 100
        
        # Import the function here to avoid circular imports
        from examples.portfolio_analysis import main
        
        # Mock to capture the output
        with patch('builtins.print') as mock_print:
            # Run the analysis
            with patch('sys.exit') as mock_exit:  # Prevent actual exit
                with patch('matplotlib.pyplot.savefig'):  # Prevent saving figures
                    with patch('matplotlib.pyplot.close'):  # Prevent closing figures
                        main()
            
            # Check if allocations were calculated correctly
            found_btc_allocation = False
            found_eth_allocation = False
            found_xrp_allocation = False
            
            for call in mock_print.call_args_list:
                if isinstance(call.args[0], str):
                    # Check BTC allocation
                    if 'BTC: $' in call.args[0] and '%' in call.args[0]:
                        allocation_str = call.args[0].split('(')[1].split('%')[0].strip()
                        try:
                            allocation = float(allocation_str)
                            # Allow for small float rounding differences
                            if abs(allocation - expected_btc_allocation) < 0.1:
                                found_btc_allocation = True
                        except ValueError:
                            pass
                    
                    # Check ETH allocation
                    elif 'ETH: $' in call.args[0] and '%' in call.args[0]:
                        allocation_str = call.args[0].split('(')[1].split('%')[0].strip()
                        try:
                            allocation = float(allocation_str)
                            if abs(allocation - expected_eth_allocation) < 0.1:
                                found_eth_allocation = True
                        except ValueError:
                            pass
                    
                    # Check XRP allocation
                    elif 'XRP: $' in call.args[0] and '%' in call.args[0]:
                        allocation_str = call.args[0].split('(')[1].split('%')[0].strip()
                        try:
                            allocation = float(allocation_str)
                            if abs(allocation - expected_xrp_allocation) < 0.1:
                                found_xrp_allocation = True
                        except ValueError:
                            pass
            
            self.assertTrue(found_btc_allocation, "BTC allocation not calculated correctly")
            self.assertTrue(found_eth_allocation, "ETH allocation not calculated correctly")
            self.assertTrue(found_xrp_allocation, "XRP allocation not calculated correctly")

    def test_handle_api_errors(self):
        """Test error handling when API calls fail"""
        # Configure the mock API to raise exceptions
        self.mock_api.get_holdings.side_effect = Exception("API Error: Holdings")
        
        # Import the function here to avoid circular imports
        from examples.portfolio_analysis import main
        
        # Mock to capture the output
        with patch('builtins.print') as mock_print:
            # Run the analysis
            with patch('sys.exit') as mock_exit:  # Prevent actual exit
                main()
            
            # Check if the error was reported
            found_error_message = False
            for call in mock_print.call_args_list:
                if isinstance(call.args[0], str) and 'Error analyzing portfolio:' in call.args[0]:
                    found_error_message = True
                    break
            
            self.assertTrue(found_error_message, "API error not handled correctly")


if __name__ == '__main__':
    unittest.main()
