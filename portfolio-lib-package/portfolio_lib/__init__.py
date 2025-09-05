"""
Portfolio-lib - Lightweight Python Backtesting Library
======================================================

A comprehensive backtesting framework for algorithmic trading strategies.

Authors: Rahul Ashok, Pritham Devaprasad, Siddarth S, and Anish R

This library provides a lightweight, high-performance backtesting engine
for developing and testing quantitative trading strategies.

Key Features:
- Ultra-lightweight architecture (< 500KB)
- 81+ technical indicators
- Event-driven backtesting engine
- Multi-asset portfolio simulation
- Comprehensive performance analytics
- Risk management tools
- yfinance data integration

Basic Usage:
    >>> from portfolio_lib import BaseStrategy, Backtest, YFinanceDataFeed
    >>> 
    >>> class MyStrategy(BaseStrategy):
    ...     def next(self):
    ...         # Your strategy logic here
    ...         pass
    >>> 
    >>> strategy = MyStrategy()
    >>> backtest = Backtest(strategy, initial_cash=100000)
    >>> data_feed = YFinanceDataFeed(['AAPL'])
    >>> backtest.add_data_source(data_feed)
    >>> results = backtest.run('2020-01-01', '2023-12-31')
    >>> print(results.summary())
"""

__version__ = "1.0.0"
__author__ = "Rahul Ashok, Pritham Devaprasad, Siddarth S, and Anish R"
__email__ = "contact@portfolio-lib.com"
__description__ = "Lightweight Python backtesting library for algorithmic trading"

# Import core backtesting framework
from .core import (
    # Core classes
    Position,
    Trade,
    Portfolio,
    
    # Data handling
    DataFeed,
    YFinanceDataFeed,
    
    # Strategy framework
    BaseStrategy,
    
    # Backtesting engine
    Backtest,
    BacktestResults,
    
    # Performance analysis
    PerformanceMetrics,
    
    # Technical indicators and helpers
    TechnicalIndicators,
)

# Import additional technical indicators
from .indicators import TechnicalIndicators as AdditionalIndicators

# Import advanced portfolio analytics and risk management
from .portfolio import (
    RiskMetrics,
    AdvancedPortfolioAnalytics,
    PositionSizing,
    PerformanceAttribution
)

__all__ = [
    # Core classes
    'Position', 'Trade', 'Portfolio',
    
    # Data handling
    'DataFeed', 'YFinanceDataFeed',
    
    # Strategy framework
    'BaseStrategy',
    
    # Backtesting engine
    'Backtest', 'BacktestResults',
    
    # Performance analysis
    'PerformanceMetrics',
    
    # Technical indicators
    'TechnicalIndicators', 'AdditionalIndicators',
    
    # Advanced analytics and risk management
    'RiskMetrics', 'AdvancedPortfolioAnalytics', 
    'PositionSizing', 'PerformanceAttribution'
]