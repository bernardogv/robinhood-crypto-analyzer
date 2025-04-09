# Tests for Robinhood Crypto Analyzer

This directory contains test suites for the Robinhood Crypto Analyzer project.

## Running Tests

You can run all tests with pytest:

```bash
# From the project root directory
python -m pytest

# To run with verbose output
python -m pytest -v

# To run a specific test file
python -m pytest tests/test_crypto_api.py
```

## Test Suites

### API Client Tests (`test_crypto_api.py`)

Tests for the Robinhood Crypto API client class, including:

- Authentication and header generation
- Query parameter formatting
- API endpoint calls
- Error handling

### Holdings Calculation Tests (`test_holdings_calculation.py`)

Tests for portfolio holdings calculation, including:

- Direct API data processing
- Fallback to order history for calculating positions
- Portfolio value calculation
- Portfolio allocation percentage calculation
- Error handling

### Trading Strategy Tests (`test_strategies.py`)

Tests for trading strategies, including:

- Moving Average Crossover strategy
- Relative Strength Index (RSI) strategy
- Spread Trading strategy
- Strategy performance metrics
- Backtest result validation

## Mock Data

The tests use mock data to simulate API responses without requiring actual API credentials. This allows for testing all functionality without making real API calls.

## Adding New Tests

When adding new functionality to the project, please also add corresponding tests. Follow these guidelines:

1. Create a new test class by subclassing `unittest.TestCase`
2. Use the `setUp` method to initialize test data and dependencies
3. Use descriptive method names that begin with `test_`
4. Mock external dependencies to avoid real API calls
5. Add assertions to validate expected behavior

## Debugging Tests

If tests are failing, you can get more detailed output by running:

```bash
python -m pytest -vv --no-header --showlocals
```

## Test Coverage

To measure test coverage, install pytest-cov and run:

```bash
# Install pytest-cov
pip install pytest-cov

# Run tests with coverage report
python -m pytest --cov=. --cov-report=term-missing
```
