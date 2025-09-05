.. portfolio-lib documentation master file, created by
   sphinx-quickstart on Fri Sep  5 15:21:16 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

portfolio-lib documentation
===========================


Welcome to portfolio-lib's documentation!
==========================================

portfolio-lib is a comprehensive Python library for quantitative finance and portfolio management. It provides advanced technical indicators, portfolio analytics, risk metrics, and trading tools for both individual investors and institutional users.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   installation
   getting_started

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   examples
   advanced_usage
   visual_guide

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api_reference

.. toctree::
   :maxdepth: 2
   :caption: Help & Support:

   troubleshooting

What is portfolio-lib?
=====================

portfolio-lib is designed for:

* **Technical Analysis**: 17+ technical indicators with mathematical formulas
* **Portfolio Management**: Advanced risk analytics and position sizing
* **Risk Assessment**: VaR, CVaR, drawdown analysis, and more
* **Performance Attribution**: Sector and factor-based analysis
* **Trading Strategies**: Backtesting and strategy development tools

Key Features
============

✨ **Technical Indicators**
   Complete set of indicators including SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, Williams %R, ATR, ADX, CCI, OBV, MFI, Ichimoku, Parabolic SAR, and more.

📊 **Portfolio Analytics**
   Advanced portfolio analytics including risk metrics, performance attribution, position sizing algorithms, and comprehensive risk management tools.

🎯 **Risk Management**
   Value at Risk (VaR), Conditional VaR, maximum drawdown, Sharpe ratio, alpha, beta, tracking error, and many other risk measures.

📈 **Visualization**
   Built-in plotting capabilities with matplotlib integration for creating professional charts and analysis reports.

🔧 **Easy to Use**
   Clean, intuitive API design with comprehensive documentation and examples.

Quick Start Example
==================

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   from portfolio_lib.portfolio import RiskMetrics
   
   # Sample price data
   prices = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
   
   # Calculate technical indicators
   sma = TechnicalIndicators.sma(prices, 5)
   rsi = TechnicalIndicators.rsi(prices, 14)
   
   # Calculate risk metrics
   returns = np.diff(prices) / prices[:-1]
   metrics = RiskMetrics(returns)
   
   print(f"Latest SMA: {sma[-1]:.2f}")
   print(f"Latest RSI: {rsi[-1]:.1f}")
   print(f"Portfolio Volatility: {metrics.var_95(returns):.2%}")

Installation
============

Install portfolio-lib using pip:

.. code-block:: bash

   pip install portfolio-lib

For development installation:

.. code-block:: bash

   git clone https://github.com/your-username/portfolio-lib.git
   cd portfolio-lib
   pip install -e .

Support
=======

* **Documentation**: Complete guides and API reference
* **Examples**: 30+ practical examples and use cases  
* **Community**: GitHub discussions and issue tracking
* **Professional Support**: Available for enterprise users

License
=======

portfolio-lib is released under the MIT License. See the LICENSE file for details.

