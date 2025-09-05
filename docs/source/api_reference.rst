API Reference
=============

Complete reference for all portfolio-lib modules, classes, and functions.

Core Modules
-----------

.. toctree::
   :maxdepth: 2

   portfolio_lib.indicators
   portfolio_lib.core  
   portfolio_lib.portfolio

Module Overview
--------------

portfolio_lib.indicators
~~~~~~~~~~~~~~~~~~~~~~~

Technical analysis indicators and oscillators.

Key classes and functions:

* :class:`~portfolio_lib.indicators.TechnicalIndicators` - Main indicators class
* :func:`~portfolio_lib.indicators.TechnicalIndicators.sma` - Simple Moving Average
* :func:`~portfolio_lib.indicators.TechnicalIndicators.ema` - Exponential Moving Average  
* :func:`~portfolio_lib.indicators.TechnicalIndicators.rsi` - Relative Strength Index
* :func:`~portfolio_lib.indicators.TechnicalIndicators.macd` - MACD Indicator
* :func:`~portfolio_lib.indicators.TechnicalIndicators.bollinger_bands` - Bollinger Bands

portfolio_lib.core
~~~~~~~~~~~~~~~~~

Core portfolio management classes.

Key classes:

* :class:`~portfolio_lib.core.Portfolio` - Main portfolio container
* :class:`~portfolio_lib.core.Position` - Individual position tracking
* :class:`~portfolio_lib.core.Trade` - Trade execution and tracking
* :class:`~portfolio_lib.core.DataFeed` - Market data management

portfolio_lib.portfolio
~~~~~~~~~~~~~~~~~~~~~~

Advanced portfolio analytics and risk management.

Key classes:

* :class:`~portfolio_lib.portfolio.RiskMetrics` - Basic risk calculations
* :class:`~portfolio_lib.portfolio.AdvancedPortfolioAnalytics` - Advanced analytics
* :class:`~portfolio_lib.portfolio.PositionSizing` - Position sizing algorithms
* :class:`~portfolio_lib.portfolio.PerformanceAttribution` - Performance analysis

Function Categories
------------------

Technical Indicators
~~~~~~~~~~~~~~~~~~~

**Trend Following**
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- MACD (Moving Average Convergence Divergence)

**Momentum Oscillators**  
- Relative Strength Index (RSI)
- Stochastic Oscillator
- Williams %R
- Commodity Channel Index (CCI)

**Volatility Indicators**
- Bollinger Bands  
- Average True Range (ATR)
- Average Directional Index (ADX)

**Volume Indicators**
- On Balance Volume (OBV)
- Money Flow Index (MFI)

**Specialized Indicators**
- Ichimoku Cloud
- Parabolic SAR
- Momentum

Risk Metrics
~~~~~~~~~~~

**Basic Risk Measures**
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Maximum Drawdown
- Volatility

**Advanced Risk Measures**
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Burke Ratio
- Ulcer Index

**Market Risk**
- Beta (Systematic Risk)
- Alpha (Risk-adjusted Return)  
- Tracking Error
- Information Ratio

Position Sizing
~~~~~~~~~~~~~~

**Risk-Based Sizing**
- Fixed Fractional
- Kelly Criterion
- Volatility-based Sizing

**Portfolio Construction**
- Risk Parity
- Equal Weight
- Market Cap Weighted

Performance Analytics
~~~~~~~~~~~~~~~~~~~

**Return Analysis**
- Total Return
- Annualized Return
- Risk-adjusted Returns

**Attribution Analysis** 
- Sector Attribution
- Factor Attribution
- Security Selection vs Allocation

Quick Reference
--------------

Common Usage Patterns
~~~~~~~~~~~~~~~~~~~~

**Calculate Technical Indicators**

.. code-block:: python

   from portfolio_lib.indicators import TechnicalIndicators
   
   # Moving averages
   sma = TechnicalIndicators.sma(prices, 20)
   ema = TechnicalIndicators.ema(prices, 20)
   
   # Oscillators
   rsi = TechnicalIndicators.rsi(prices, 14)
   macd_line, signal, histogram = TechnicalIndicators.macd(prices)

**Risk Analysis**

.. code-block:: python

   from portfolio_lib.portfolio import RiskMetrics
   
   metrics = RiskMetrics(returns)
   var_95 = metrics.var_95(returns)
   max_dd = metrics.maximum_drawdown(equity_curve)

**Portfolio Management**

.. code-block:: python

   from portfolio_lib.core import Portfolio, Position
   
   portfolio = Portfolio(initial_cash=100000)
   position = Position("AAPL", 100, 150.0, datetime.now())
   portfolio.add_position(position)

Data Types and Formats
---------------------

**Input Data Formats**

All functions accept:
- Python lists: ``[100, 101, 102, ...]``
- NumPy arrays: ``np.array([100, 101, 102, ...])``
- Pandas Series: ``pd.Series([100, 101, 102, ...])``

**Return Types**

- Single values: ``float``
- Arrays: ``np.ndarray`` 
- Tuples: ``(upper, middle, lower)`` for Bollinger Bands
- Objects: Custom classes for complex results

**Date/Time Handling**

- Use ``pandas.Timestamp`` or ``datetime.datetime``
- Support for time zones and business day calendars
- Automatic alignment of time series data

Error Handling
-------------

**Common Exceptions**

- ``ValueError``: Invalid parameters or insufficient data
- ``TypeError``: Wrong data types passed to functions  
- ``IndexError``: Array index out of bounds
- ``ZeroDivisionError``: Division by zero in calculations

**Best Practices**

- Always validate input data lengths
- Check for NaN values in results
- Handle insufficient data gracefully
- Use appropriate error handling in production code

Version Compatibility
--------------------

**Supported Python Versions**
- Python 3.8+
- Python 3.9+ (recommended)
- Python 3.10+
- Python 3.11+

**Dependencies**
- NumPy >= 1.20.0
- Pandas >= 1.3.0  
- Matplotlib >= 3.4.0
- SciPy >= 1.7.0

**Breaking Changes**
- v1.0.0: Initial stable release
- See CHANGELOG.md for version-specific changes
