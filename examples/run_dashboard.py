#!/usr/bin/env python3
"""
Dashboard Launcher for Robinhood Crypto Analyzer

This script launches the interactive dashboard for analyzing cryptocurrency data.
"""

import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our modules
from crypto_api import RobinhoodCryptoAPI
from visualizers.dashboard import CryptoDashboard

# Try to import the API credentials from config.py
try:
    from config import API_KEY, BASE64_PRIVATE_KEY
except ImportError:
    print("Error: API credentials not found.")
    print("Please create a config.py file with your API credentials.")
    sys.exit(1)

def main():
    """Main function."""
    print("Launching Robinhood Crypto Analyzer Dashboard...")
    print("===============================================")
    
    # Initialize the API client
    print("\nInitializing API client...")
    api = RobinhoodCryptoAPI(API_KEY, BASE64_PRIVATE_KEY)
    
    # Create and run the dashboard
    print("\nStarting dashboard server...")
    print("Access the dashboard at http://localhost:8050")
    print("Press Ctrl+C to stop the server")
    
    dashboard = CryptoDashboard(api)
    dashboard.run_server(debug=True, port=8050)

if __name__ == "__main__":
    main()
