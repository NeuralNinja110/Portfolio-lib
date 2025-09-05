portfolio_lib.portfolio module
============================

.. automodule:: portfolio_lib.portfolio
   :members:
   :show-inheritance:
   :undoc-members:

.. _portfolio-examples:

Portfolio Classes, Examples, and Visuals
----------------------------------------

RiskMetrics Class
~~~~~~~~~~~~~~~~~

A dataclass container for comprehensive risk metrics.

**Attributes:**

- var_95: float — Value at Risk (95%)
- var_99: float — Value at Risk (99%)
- cvar_95: float — Conditional VaR (95%)
- cvar_99: float — Conditional VaR (99%)
- skewness: float — Return distribution skewness
- kurtosis: float — Return distribution kurtosis
- maximum_drawdown: float — Maximum portfolio drawdown
- calmar_ratio: float — Annual return / max drawdown
- sterling_ratio: float — Risk-adjusted return metric
- burke_ratio: float — Return / Ulcer Index

**Tested Example:**

.. code-block:: python

   from portfolio_lib.portfolio import RiskMetrics
   metrics = RiskMetrics(
       var_95=-0.02, var_99=-0.035, cvar_95=-0.025, cvar_99=-0.04,
       skewness=-0.5, kurtosis=3.2, maximum_drawdown=-0.15,
       calmar_ratio=1.2, sterling_ratio=1.8, burke_ratio=2.1
   )
   print(f"VaR 95%: {metrics.var_95:.2%}")
   print(f"Max Drawdown: {metrics.maximum_drawdown:.2%}")

**Output:**

.. code-block:: text

   VaR 95%: -2.00%
   Max Drawdown: -15.00%

AdvancedPortfolioAnalytics Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive portfolio analytics and risk management.

**Methods:**

- calculate_var(confidence_level): Value at Risk calculation
- calculate_cvar(confidence_level): Conditional VaR calculation
- calculate_maximum_drawdown(equity_curve): Maximum drawdown analysis
- calculate_ulcer_index(equity_curve): Downside risk measure
- calculate_tracking_error(): Volatility vs benchmark
- calculate_information_ratio(): Risk-adjusted excess return
- calculate_beta(): Systematic risk measure
- calculate_alpha(risk_free_rate): Risk-adjusted excess return
- calculate_comprehensive_risk_metrics(equity_curve): All metrics

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.portfolio import AdvancedPortfolioAnalytics
   np.random.seed(42)
   returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
   benchmark = np.random.normal(0.0008, 0.015, 252)
   analytics = AdvancedPortfolioAnalytics(returns, benchmark)
   equity_curve = np.cumprod(1 + returns)
   metrics = analytics.calculate_comprehensive_risk_metrics(equity_curve)
   print(f"VaR 95%: {analytics.calculate_var(0.05):.4f}")
   print(f"Tracking Error: {analytics.calculate_tracking_error():.4f}")
   print(f"Information Ratio: {analytics.calculate_information_ratio():.4f}")

**Output:**

.. code-block:: text

   VaR 95%: -0.0316
   Tracking Error: 0.0793
   Information Ratio: 0.9525

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.portfolio import AdvancedPortfolioAnalytics
   np.random.seed(42)
   returns = np.random.normal(0.001, 0.02, 252)
   equity_curve = np.cumprod(1 + returns)
   analytics = AdvancedPortfolioAnalytics(returns)
   max_dd, _, _ = analytics.calculate_maximum_drawdown(equity_curve)
   plt.figure(figsize=(10, 6))
   plt.plot(equity_curve, label='Equity Curve')
   plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Initial Value')
   plt.title(f'Portfolio Equity Curve (Max DD: {max_dd:.2%})')
   plt.legend()
   plt.show()

PositionSizing Class
~~~~~~~~~~~~~~~~~~~

Static methods for position sizing and risk management.

**Methods:**

- kelly_criterion(win_rate, avg_win, avg_loss): Optimal position size
- fixed_fractional(account_equity, risk_per_trade, stop_loss_pct): Fixed risk sizing
- volatility_position_sizing(account_equity, target_vol, asset_vol): Vol targeting
- risk_parity_weights(covariance_matrix): Equal risk contribution weights

**Kelly Criterion Formula:**

.. math::
   f^* = \frac{bp - q}{b}

Where :math:`f^*` is the Kelly fraction, :math:`b` is the win/loss ratio, :math:`p` is win probability, :math:`q` is loss probability.

**Tested Example:**

.. code-block:: python

   from portfolio_lib.portfolio import PositionSizing
   import numpy as np
   # Kelly Criterion
   kelly_size = PositionSizing.kelly_criterion(0.6, 100, 50)
   print(f"Kelly Fraction: {kelly_size:.4f}")
   # Fixed Fractional
   position_size = PositionSizing.fixed_fractional(10000, 0.02, 0.05)
   print(f"Position Size: ${position_size:.2f}")
   # Risk Parity
   cov_matrix = np.array([[0.04, 0.01], [0.01, 0.09]])
   weights = PositionSizing.risk_parity_weights(cov_matrix)
   print(f"Risk Parity Weights: {weights}")

**Output:**

.. code-block:: text

   Kelly Fraction: 0.2000
   Position Size: $4000.00
   Risk Parity Weights: [0.6 0.4]

PerformanceAttribution Class
~~~~~~~~~~~~~~~~~~~~~~~~~~

Performance attribution analysis for portfolio management.

**Methods:**

- brinson_attribution(benchmark_weights): Brinson-Fachler attribution
- calculate_sector_attribution(sector_mapping): Sector-based attribution

**Brinson Attribution Formula:**

.. math::
   \text{Allocation Effect} = (w_p - w_b) \times r_b

.. math::
   \text{Selection Effect} = w_b \times (r_p - r_b)

.. math::
   \text{Interaction Effect} = (w_p - w_b) \times (r_p - r_b)

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.portfolio import PerformanceAttribution
   portfolio_returns = np.array([0.05, 0.03, 0.02])
   benchmark_returns = np.array([0.04, 0.025, 0.015])
   weights = np.array([0.4, 0.3, 0.3])
   asset_returns = np.array([0.06, 0.04, 0.02])
   attribution = PerformanceAttribution(portfolio_returns, benchmark_returns, weights, asset_returns)
   benchmark_weights = np.array([0.5, 0.3, 0.2])
   results = attribution.brinson_attribution(benchmark_weights)
   print(f"Allocation Effect: {results['allocation']}")

**Output:**

.. code-block:: text

   Allocation Effect: [-0.006  0.     0.002]
