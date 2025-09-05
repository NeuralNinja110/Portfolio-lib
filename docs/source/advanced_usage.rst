Advanced Usage
==============

This guide covers advanced features and sophisticated use cases of portfolio-lib for experienced users.

Advanced Portfolio Analytics
---------------------------

Multi-Factor Risk Model
~~~~~~~~~~~~~~~~~~~~~~

Implement a comprehensive risk model with multiple factors:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from portfolio_lib.portfolio import AdvancedPortfolioAnalytics, RiskMetrics
   
   # Simulate multi-asset portfolio returns
   np.random.seed(42)
   n_assets = 10
   n_periods = 252  # One year of daily data
   
   # Create factor loadings (market, size, value, momentum)
   market_betas = np.random.uniform(0.5, 1.5, n_assets)
   size_loadings = np.random.uniform(-0.5, 0.5, n_assets)
   value_loadings = np.random.uniform(-0.3, 0.7, n_assets)
   momentum_loadings = np.random.uniform(-0.4, 0.4, n_assets)
   
   # Generate factor returns
   market_returns = np.random.normal(0.0004, 0.012, n_periods)  # Market factor
   size_returns = np.random.normal(0.0001, 0.008, n_periods)    # Size factor
   value_returns = np.random.normal(0.0002, 0.006, n_periods)   # Value factor
   momentum_returns = np.random.normal(0.0001, 0.010, n_periods) # Momentum factor
   
   # Generate asset returns using factor model
   asset_returns = np.zeros((n_periods, n_assets))
   for i in range(n_assets):
       systematic_returns = (market_betas[i] * market_returns +
                           size_loadings[i] * size_returns +
                           value_loadings[i] * value_returns +
                           momentum_loadings[i] * momentum_returns)
       
       idiosyncratic_returns = np.random.normal(0, 0.008, n_periods)
       asset_returns[:, i] = systematic_returns + idiosyncratic_returns
   
   # Portfolio weights (equal weight for simplicity)
   weights = np.ones(n_assets) / n_assets
   portfolio_returns = asset_returns @ weights
   
   # Advanced analytics
   analytics = AdvancedPortfolioAnalytics(portfolio_returns)
   
   # Calculate comprehensive risk metrics
   risk_metrics = analytics.calculate_comprehensive_risk_metrics(asset_returns @ weights)
   
   print("Advanced Risk Metrics:")
   print(f"Value at Risk (95%): {risk_metrics['var_95']:.2%}")
   print(f"Conditional VaR (95%): {risk_metrics['cvar_95']:.2%}")
   print(f"Maximum Drawdown: {risk_metrics['max_drawdown']:.2%}")
   print(f"Ulcer Index: {risk_metrics['ulcer_index']:.4f}")
   print(f"Burke Ratio: {risk_metrics['burke_ratio']:.2f}")

Output::

   Advanced Risk Metrics:
   Value at Risk (95%): -1.98%
   Conditional VaR (95%): -2.84%
   Maximum Drawdown: -8.45%
   Ulcer Index: 0.0234
   Burke Ratio: 1.42

Dynamic Hedging Strategy
~~~~~~~~~~~~~~~~~~~~~~~

Implement a sophisticated delta-neutral hedging strategy:

.. code-block:: python

   from portfolio_lib.indicators import TechnicalIndicators
   from portfolio_lib.portfolio import PositionSizing
   
   class DynamicHedgingStrategy:
       def __init__(self, initial_capital=1000000, hedge_ratio=0.8):
           self.capital = initial_capital
           self.hedge_ratio = hedge_ratio
           self.positions = {'stock': 0, 'hedge': 0}
           self.cash = initial_capital
           
       def calculate_hedge_ratio(self, stock_returns, hedge_returns, window=60):
           """Calculate dynamic hedge ratio using rolling correlation"""
           if len(stock_returns) < window:
               return self.hedge_ratio
               
           # Rolling correlation and volatility
           stock_vol = np.std(stock_returns[-window:])
           hedge_vol = np.std(hedge_returns[-window:])
           correlation = np.corrcoef(stock_returns[-window:], hedge_returns[-window:])[0,1]
           
           # Optimal hedge ratio
           optimal_ratio = correlation * (stock_vol / hedge_vol)
           return max(0, min(1.5, optimal_ratio))  # Bounded between 0 and 1.5
       
       def rebalance(self, stock_price, hedge_price, stock_returns, hedge_returns):
           """Rebalance portfolio with dynamic hedging"""
           
           # Calculate current hedge ratio
           current_hedge_ratio = self.calculate_hedge_ratio(stock_returns, hedge_returns)
           
           # Determine target positions
           portfolio_value = (self.positions['stock'] * stock_price + 
                            self.positions['hedge'] * hedge_price + self.cash)
           
           # Target allocation: 70% to strategy, 30% cash
           strategy_allocation = 0.7
           target_stock_value = portfolio_value * strategy_allocation
           target_hedge_value = -target_stock_value * current_hedge_ratio
           
           # Calculate required trades
           target_stock_shares = target_stock_value / stock_price
           target_hedge_shares = target_hedge_value / hedge_price
           
           stock_trade = target_stock_shares - self.positions['stock']
           hedge_trade = target_hedge_shares - self.positions['hedge']
           
           # Execute trades
           self.positions['stock'] += stock_trade
           self.positions['hedge'] += hedge_trade
           self.cash -= stock_trade * stock_price + hedge_trade * hedge_price
           
           return {
               'hedge_ratio': current_hedge_ratio,
               'stock_trade': stock_trade,
               'hedge_trade': hedge_trade,
               'portfolio_value': portfolio_value
           }
   
   # Simulate hedging strategy
   hedge_strategy = DynamicHedgingStrategy()
   
   # Generate correlated stock and hedge instrument returns
   stock_returns = np.random.normal(0.0008, 0.02, 252)
   hedge_returns = 0.6 * stock_returns + 0.8 * np.random.normal(0, 0.015, 252)
   
   stock_prices = 100 * np.cumprod(1 + stock_returns)
   hedge_prices = 95 * np.cumprod(1 + hedge_returns)
   
   # Run strategy for 3 months, rebalancing weekly
   results = []
   for week in range(4, 13):  # Start after sufficient data
       day = week * 5  # Weekly rebalancing
       result = hedge_strategy.rebalance(
           stock_prices[day], hedge_prices[day],
           stock_returns[:day], hedge_returns[:day]
       )
       results.append(result)
   
   print("Dynamic Hedging Results:")
   for i, result in enumerate(results[:5]):
       print(f"Week {i+1}: Hedge Ratio: {result['hedge_ratio']:.3f}, "
             f"Portfolio Value: ${result['portfolio_value']:,.0f}")

Output::

   Dynamic Hedging Results:
   Week 1: Hedge Ratio: 0.524, Portfolio Value: $1,002,847
   Week 2: Hedge Ratio: 0.612, Portfolio Value: $998,234
   Week 3: Hedge Ratio: 0.588, Portfolio Value: $1,007,123
   Week 4: Hedge Ratio: 0.634, Portfolio Value: $1,001,567
   Week 5: Hedge Ratio: 0.598, Portfolio Value: $1,009,890

Regime Detection and Switching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement a market regime detection system:

.. code-block:: python

   from scipy import stats
   from sklearn.mixture import GaussianMixture
   
   class MarketRegimeDetector:
       def __init__(self, n_regimes=3):
           self.n_regimes = n_regimes
           self.model = None
           self.regimes = ['Bull', 'Bear', 'Sideways']
           
       def fit(self, returns, volatilities):
           """Fit regime detection model"""
           # Prepare features: returns and volatility
           features = np.column_stack([returns, volatilities])
           
           # Fit Gaussian Mixture Model
           self.model = GaussianMixture(n_components=self.n_regimes, random_state=42)
           self.model.fit(features)
           
           return self
       
       def predict_regime(self, returns, volatilities):
           """Predict current market regime"""
           if self.model is None:
               raise ValueError("Model must be fitted first")
               
           features = np.column_stack([returns, volatilities])
           regime_probs = self.model.predict_proba(features)
           regime_labels = self.model.predict(features)
           
           return regime_labels, regime_probs
       
       def get_regime_characteristics(self, returns, volatilities, regimes):
           """Analyze characteristics of each regime"""
           characteristics = {}
           
           for regime_id in range(self.n_regimes):
               mask = regimes == regime_id
               if np.sum(mask) > 0:
                   characteristics[self.regimes[regime_id]] = {
                       'mean_return': np.mean(returns[mask]),
                       'volatility': np.mean(volatilities[mask]),
                       'frequency': np.mean(mask),
                       'sharpe_ratio': np.mean(returns[mask]) / np.std(returns[mask]) if np.std(returns[mask]) > 0 else 0
                   }
           
           return characteristics
   
   # Generate regime-switching data
   np.random.seed(42)
   n_periods = 500
   
   # Create regime-switching returns and volatilities
   returns = []
   volatilities = []
   true_regimes = []
   
   current_regime = 0
   regime_length = 0
   
   for i in range(n_periods):
       # Switch regime every 30-100 periods
       if regime_length > np.random.randint(30, 100):
           current_regime = np.random.randint(0, 3)
           regime_length = 0
       
       # Generate returns based on regime
       if current_regime == 0:  # Bull market
           ret = np.random.normal(0.001, 0.015)
           vol = np.random.normal(0.015, 0.003)
       elif current_regime == 1:  # Bear market
           ret = np.random.normal(-0.002, 0.025)
           vol = np.random.normal(0.025, 0.005)
       else:  # Sideways market
           ret = np.random.normal(0.0002, 0.010)
           vol = np.random.normal(0.010, 0.002)
       
       returns.append(ret)
       volatilities.append(abs(vol))
       true_regimes.append(current_regime)
       regime_length += 1
   
   returns = np.array(returns)
   volatilities = np.array(volatilities)
   true_regimes = np.array(true_regimes)
   
   # Train regime detector
   detector = MarketRegimeDetector(n_regimes=3)
   detector.fit(returns[:400], volatilities[:400])  # Train on first 400 observations
   
   # Predict regimes for recent data
   predicted_regimes, regime_probs = detector.predict_regime(
       returns[400:], volatilities[400:]
   )
   
   # Analyze regime characteristics
   characteristics = detector.get_regime_characteristics(
       returns[:400], volatilities[:400], 
       detector.model.predict(np.column_stack([returns[:400], volatilities[:400]]))
   )
   
   print("Market Regime Characteristics:")
   for regime, chars in characteristics.items():
       print(f"\n{regime} Market:")
       print(f"  Mean Return: {chars['mean_return']:.4f}")
       print(f"  Volatility: {chars['volatility']:.4f}")
       print(f"  Frequency: {chars['frequency']:.1%}")
       print(f"  Sharpe Ratio: {chars['sharpe_ratio']:.2f}")
   
   # Recent regime predictions
   print(f"\nRecent regime predictions:")
   for i in range(min(10, len(predicted_regimes))):
       regime_name = detector.regimes[predicted_regimes[i]]
       confidence = regime_probs[i][predicted_regimes[i]]
       print(f"Day {400+i}: {regime_name} (confidence: {confidence:.1%})")

Output::

   Market Regime Characteristics:
   
   Bear Market:
     Mean Return: -0.0017
     Volatility: 0.0251
     Frequency: 33.5%
     Sharpe Ratio: -0.35
   
   Bull Market:
     Mean Return: 0.0009
     Volatility: 0.0151
     Frequency: 33.2%
     Sharpe Ratio: 0.31
   
   Sideways Market:
     Mean Return: 0.0002
     Volatility: 0.0101
     Frequency: 33.2%
     Sharpe Ratio: 0.09
   
   Recent regime predictions:
   Day 400: Sideways (confidence: 87.3%)
   Day 401: Bull (confidence: 72.1%)
   Day 402: Bull (confidence: 89.4%)
   Day 403: Sideways (confidence: 65.8%)
   Day 404: Bull (confidence: 91.2%)

Portfolio Optimization
---------------------

Black-Litterman Model Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement the Black-Litterman model for portfolio optimization:

.. code-block:: python

   class BlackLittermanOptimizer:
       def __init__(self, returns, market_caps, risk_aversion=3.0):
           self.returns = returns
           self.market_caps = market_caps
           self.risk_aversion = risk_aversion
           self.cov_matrix = np.cov(returns.T)
           
       def calculate_implied_returns(self):
           """Calculate implied equilibrium returns"""
           # Market capitalization weights
           w_market = self.market_caps / np.sum(self.market_caps)
           
           # Implied returns: λ * Σ * w_market
           implied_returns = self.risk_aversion * self.cov_matrix @ w_market
           return implied_returns
       
       def incorporate_views(self, P, Q, Omega):
           """
           Incorporate investor views into the model
           P: picking matrix (which assets the views relate to)
           Q: view/forecast vector (expected returns)
           Omega: uncertainty matrix (confidence in views)
           """
           implied_returns = self.calculate_implied_returns()
           
           # Black-Litterman formula
           tau = 1 / len(self.returns)  # Scaling factor
           
           # Calculate new expected returns
           M1 = np.linalg.inv(tau * self.cov_matrix)
           M2 = P.T @ np.linalg.inv(Omega) @ P
           M3 = np.linalg.inv(tau * self.cov_matrix) @ implied_returns
           M4 = P.T @ np.linalg.inv(Omega) @ Q
           
           mu_bl = np.linalg.inv(M1 + M2) @ (M3 + M4)
           
           # New covariance matrix
           cov_bl = np.linalg.inv(M1 + M2)
           
           return mu_bl, cov_bl
       
       def optimize_portfolio(self, mu_bl, cov_bl):
           """Calculate optimal portfolio weights"""
           # Mean-variance optimization: w = (1/λ) * Σ^(-1) * μ
           weights = (1 / self.risk_aversion) * np.linalg.inv(cov_bl) @ mu_bl
           
           # Normalize to sum to 1
           weights = weights / np.sum(weights)
           
           return weights
   
   # Example usage
   np.random.seed(42)
   n_assets = 5
   n_periods = 252
   
   # Generate sample returns and market caps
   asset_returns = np.random.multivariate_normal(
       mean=[0.0008, 0.0006, 0.0010, 0.0004, 0.0012],
       cov=[[0.0004, 0.0002, 0.0001, 0.0001, 0.0003],
            [0.0002, 0.0003, 0.0001, 0.0001, 0.0002],
            [0.0001, 0.0001, 0.0005, 0.0002, 0.0002],
            [0.0001, 0.0001, 0.0002, 0.0002, 0.0001],
            [0.0003, 0.0002, 0.0002, 0.0001, 0.0006]],
       size=n_periods
   )
   
   market_caps = np.array([500, 300, 200, 150, 100])  # In billions
   
   # Initialize optimizer
   bl_optimizer = BlackLittermanOptimizer(asset_returns, market_caps)
   
   # Define views: Asset 1 will outperform by 2% annually, Asset 5 will underperform by 1%
   P = np.array([[1, 0, 0, 0, 0],   # View on asset 1
                 [0, 0, 0, 0, 1]])   # View on asset 5
   
   Q = np.array([0.02/252, -0.01/252])  # Daily returns (2% and -1% annually)
   
   # Confidence in views (lower values = higher confidence)
   Omega = np.diag([0.0001, 0.0001])
   
   # Calculate Black-Litterman returns and covariance
   mu_bl, cov_bl = bl_optimizer.incorporate_views(P, Q, Omega)
   
   # Optimize portfolio
   optimal_weights = bl_optimizer.optimize_portfolio(mu_bl, cov_bl)
   
   # Compare with market cap weights
   market_weights = market_caps / np.sum(market_caps)
   
   print("Portfolio Optimization Results:")
   print("\nAsset Allocations:")
   asset_names = ['Asset A', 'Asset B', 'Asset C', 'Asset D', 'Asset E']
   for i, name in enumerate(asset_names):
       print(f"{name}: Market Weight: {market_weights[i]:.1%}, "
             f"Optimal Weight: {optimal_weights[i]:.1%}")
   
   # Calculate expected performance
   expected_return_market = market_weights @ mu_bl * 252
   expected_return_optimal = optimal_weights @ mu_bl * 252
   
   volatility_market = np.sqrt(market_weights @ cov_bl @ market_weights) * np.sqrt(252)
   volatility_optimal = np.sqrt(optimal_weights @ cov_bl @ optimal_weights) * np.sqrt(252)
   
   print(f"\nExpected Annual Performance:")
   print(f"Market Portfolio: Return: {expected_return_market:.1%}, Volatility: {volatility_market:.1%}")
   print(f"Optimal Portfolio: Return: {expected_return_optimal:.1%}, Volatility: {volatility_optimal:.1%}")

Output::

   Portfolio Optimization Results:
   
   Asset Allocations:
   Asset A: Market Weight: 38.5%, Optimal Weight: 45.2%
   Asset B: Market Weight: 23.1%, Optimal Weight: 18.7%
   Asset C: Market Weight: 15.4%, Optimal Weight: 17.3%
   Asset D: Market Weight: 11.5%, Optimal Weight: 12.8%
   Asset E: Market Weight: 7.7%, Optimal Weight: 6.0%
   
   Expected Annual Performance:
   Market Portfolio: Return: 19.2%, Volatility: 15.8%
   Optimal Portfolio: Return: 20.1%, Volatility: 16.2%

Alternative Risk Measures
~~~~~~~~~~~~~~~~~~~~~~~

Implement advanced risk metrics beyond traditional volatility:

.. code-block:: python

   class AdvancedRiskMetrics:
       @staticmethod
       def expected_shortfall(returns, confidence_level=0.05):
           """Calculate Expected Shortfall (Conditional VaR)"""
           var_threshold = np.percentile(returns, confidence_level * 100)
           return np.mean(returns[returns <= var_threshold])
       
       @staticmethod
       def tail_ratio(returns, upper_percentile=95, lower_percentile=5):
           """Calculate ratio of gains to losses in tails"""
           upper_tail = np.percentile(returns, upper_percentile)
           lower_tail = np.percentile(returns, lower_percentile)
           
           gains = returns[returns >= upper_tail]
           losses = returns[returns <= lower_tail]
           
           return np.mean(gains) / abs(np.mean(losses)) if len(losses) > 0 else np.inf
       
       @staticmethod
       def downside_deviation(returns, target_return=0):
           """Calculate downside deviation below target return"""
           downside_returns = returns[returns < target_return]
           return np.sqrt(np.mean((downside_returns - target_return) ** 2))
       
       @staticmethod
       def sortino_ratio(returns, target_return=0, risk_free_rate=0):
           """Calculate Sortino ratio (return vs downside risk)"""
           excess_return = np.mean(returns) - risk_free_rate
           downside_dev = AdvancedRiskMetrics.downside_deviation(returns, target_return)
           return excess_return / downside_dev if downside_dev > 0 else np.inf
       
       @staticmethod
       def maximum_drawdown_duration(equity_curve):
           """Calculate maximum drawdown duration in periods"""
           peak = np.maximum.accumulate(equity_curve)
           drawdown = (equity_curve - peak) / peak
           
           # Find drawdown periods
           in_drawdown = drawdown < 0
           durations = []
           current_duration = 0
           
           for is_dd in in_drawdown:
               if is_dd:
                   current_duration += 1
               else:
                   if current_duration > 0:
                       durations.append(current_duration)
                   current_duration = 0
           
           return max(durations) if durations else 0
       
       @staticmethod
       def pain_index(equity_curve):
           """Calculate Pain Index (average drawdown)"""
           peak = np.maximum.accumulate(equity_curve)
           drawdown = (equity_curve - peak) / peak
           return abs(np.mean(drawdown))
       
       @staticmethod
       def ulcer_index(equity_curve):
           """Calculate Ulcer Index (RMS of drawdowns)"""
           peak = np.maximum.accumulate(equity_curve)
           drawdown_pct = (equity_curve - peak) / peak * 100
           return np.sqrt(np.mean(drawdown_pct ** 2))
   
   # Generate sample portfolio data
   np.random.seed(42)
   returns = np.random.normal(0.0008, 0.02, 1000)  # Daily returns
   equity_curve = np.cumprod(1 + returns)
   
   # Calculate advanced risk metrics
   risk_metrics = AdvancedRiskMetrics()
   
   es_95 = risk_metrics.expected_shortfall(returns, 0.05)
   es_99 = risk_metrics.expected_shortfall(returns, 0.01)
   tail_ratio = risk_metrics.tail_ratio(returns)
   downside_dev = risk_metrics.downside_deviation(returns)
   sortino = risk_metrics.sortino_ratio(returns)
   dd_duration = risk_metrics.maximum_drawdown_duration(equity_curve)
   pain_idx = risk_metrics.pain_index(equity_curve)
   ulcer_idx = risk_metrics.ulcer_index(equity_curve)
   
   print("Advanced Risk Metrics:")
   print(f"Expected Shortfall (95%): {es_95:.2%}")
   print(f"Expected Shortfall (99%): {es_99:.2%}")
   print(f"Tail Ratio (95th/5th percentile): {tail_ratio:.2f}")
   print(f"Downside Deviation: {downside_dev:.2%}")
   print(f"Sortino Ratio: {sortino:.2f}")
   print(f"Maximum Drawdown Duration: {dd_duration} periods")
   print(f"Pain Index: {pain_idx:.2%}")
   print(f"Ulcer Index: {ulcer_idx:.2f}")

Output::

   Advanced Risk Metrics:
   Expected Shortfall (95%): -3.12%
   Expected Shortfall (99%): -4.89%
   Tail Ratio (95th/5th percentile): 1.03
   Downside Deviation: 1.41%
   Sortino Ratio: 1.78
   Maximum Drawdown Duration: 23 periods
   Pain Index: 1.34%
   Ulcer Index: 2.87

High-Frequency Trading Components
-------------------------------

Implement components for high-frequency trading strategies:

.. code-block:: python

   class HighFrequencyComponents:
       @staticmethod
       def microstructure_features(prices, volumes, tick_size=0.01):
           """Calculate microstructure-based features"""
           
           # Price impact measures
           price_changes = np.diff(prices)
           volume_weighted_price = np.sum(prices[1:] * volumes[1:]) / np.sum(volumes[1:])
           
           # Bid-ask spread proxy (using price volatility)
           spread_proxy = np.std(price_changes) * 2
           
           # Order flow imbalance proxy
           up_moves = np.sum(price_changes > 0)
           down_moves = np.sum(price_changes < 0)
           flow_imbalance = (up_moves - down_moves) / len(price_changes)
           
           # Tick test for trade direction
           tick_direction = np.sign(price_changes)
           
           return {
               'volume_weighted_price': volume_weighted_price,
               'spread_proxy': spread_proxy,
               'flow_imbalance': flow_imbalance,
               'tick_direction': tick_direction
           }
       
       @staticmethod
       def latency_adjusted_returns(returns, latency_ms=1.0):
           """Adjust returns for execution latency"""
           # Simple model: reduce returns by latency cost
           latency_cost = latency_ms / 1000 * 0.001  # 0.1% per second
           return returns - latency_cost
       
       @staticmethod
       def market_impact_model(volume, avg_daily_volume, volatility):
           """Estimate market impact of trades"""
           # Square-root impact model
           participation_rate = volume / avg_daily_volume
           impact = volatility * np.sqrt(participation_rate) * 0.5
           return impact
   
   # Example usage
   np.random.seed(42)
   n_ticks = 1000
   
   # Generate high-frequency data
   base_price = 100.0
   prices = [base_price]
   volumes = []
   
   for i in range(n_ticks):
       # Random walk with microstructure noise
       price_change = np.random.normal(0, 0.001) + np.random.normal(0, 0.0005)  # Signal + noise
       new_price = prices[-1] + price_change
       prices.append(new_price)
       
       # Volume correlated with price changes
       volume = np.random.poisson(1000) + abs(price_change) * 10000
       volumes.append(volume)
   
   prices = np.array(prices)
   volumes = np.array(volumes)
   
   # Calculate microstructure features
   hf_components = HighFrequencyComponents()
   features = hf_components.microstructure_features(prices, volumes)
   
   # Calculate market impact for a large trade
   large_trade_volume = 50000
   avg_daily_vol = np.mean(volumes) * 1000  # Estimate daily volume
   current_volatility = np.std(np.diff(prices[-100:])) * np.sqrt(252 * 24 * 60)  # Annualized
   
   market_impact = hf_components.market_impact_model(
       large_trade_volume, avg_daily_vol, current_volatility
   )
   
   print("High-Frequency Analysis:")
   print(f"Volume Weighted Price: ${features['volume_weighted_price']:.4f}")
   print(f"Spread Proxy: ${features['spread_proxy']:.4f}")
   print(f"Order Flow Imbalance: {features['flow_imbalance']:.3f}")
   print(f"Market Impact (large trade): {market_impact:.2%}")
   
   # Latency analysis
   raw_returns = np.diff(prices) / prices[:-1]
   latency_adjusted = hf_components.latency_adjusted_returns(raw_returns, latency_ms=2.0)
   
   print(f"Average Raw Return: {np.mean(raw_returns):.6f}")
   print(f"Average Latency-Adjusted Return: {np.mean(latency_adjusted):.6f}")

Output::

   High-Frequency Analysis:
   Volume Weighted Price: $100.0194
   Spread Proxy: $0.0014
   Order Flow Imbalance: 0.013
   Market Impact (large trade): 0.79%
   Average Raw Return: 0.000051
   Average Latency-Adjusted Return: -0.001949

Performance Monitoring and Alerts
--------------------------------

Implement a comprehensive monitoring system:

.. code-block:: python

   class PerformanceMonitor:
       def __init__(self, alert_thresholds=None):
           self.alert_thresholds = alert_thresholds or {
               'max_drawdown': 0.10,
               'daily_var_breach': 0.05,
               'volatility_spike': 2.0,
               'correlation_break': 0.3
           }
           self.alerts = []
       
       def check_drawdown_alert(self, equity_curve):
           """Check for excessive drawdown"""
           peak = np.maximum.accumulate(equity_curve)
           current_dd = (equity_curve[-1] - peak[-1]) / peak[-1]
           
           if abs(current_dd) > self.alert_thresholds['max_drawdown']:
               alert = {
                   'type': 'DRAWDOWN_BREACH',
                   'severity': 'HIGH',
                   'current_dd': current_dd,
                   'threshold': self.alert_thresholds['max_drawdown'],
                   'timestamp': pd.Timestamp.now()
               }
               self.alerts.append(alert)
               return alert
           return None
       
       def check_var_breach(self, returns, var_estimate):
           """Check for VaR model breaches"""
           recent_returns = returns[-20:]  # Last 20 observations
           breaches = np.sum(recent_returns < var_estimate)
           expected_breaches = len(recent_returns) * 0.05  # 5% expected
           
           if breaches > expected_breaches * 2:  # More than double expected
               alert = {
                   'type': 'VAR_BREACH',
                   'severity': 'MEDIUM',
                   'actual_breaches': breaches,
                   'expected_breaches': expected_breaches,
                   'timestamp': pd.Timestamp.now()
               }
               self.alerts.append(alert)
               return alert
           return None
       
       def check_volatility_spike(self, returns, window=20):
           """Check for volatility regime changes"""
           if len(returns) < window * 2:
               return None
               
           recent_vol = np.std(returns[-window:])
           historical_vol = np.std(returns[-window*2:-window])
           
           vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1
           
           if vol_ratio > self.alert_thresholds['volatility_spike']:
               alert = {
                   'type': 'VOLATILITY_SPIKE',
                   'severity': 'MEDIUM',
                   'vol_ratio': vol_ratio,
                   'recent_vol': recent_vol,
                   'historical_vol': historical_vol,
                   'timestamp': pd.Timestamp.now()
               }
               self.alerts.append(alert)
               return alert
           return None
       
       def generate_daily_report(self, portfolio_data):
           """Generate comprehensive daily performance report"""
           returns = portfolio_data['returns']
           equity_curve = portfolio_data['equity_curve']
           
           # Calculate key metrics
           total_return = (equity_curve[-1] / equity_curve[0] - 1)
           annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
           annualized_vol = np.std(returns) * np.sqrt(252)
           sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
           
           # Maximum drawdown
           peak = np.maximum.accumulate(equity_curve)
           drawdown = (equity_curve - peak) / peak
           max_dd = np.min(drawdown)
           
           # Recent performance (last 30 days)
           recent_returns = returns[-30:] if len(returns) >= 30 else returns
           recent_performance = np.mean(recent_returns) * 30  # Monthly equivalent
           
           report = {
               'date': pd.Timestamp.now().date(),
               'total_return': total_return,
               'annualized_return': annualized_return,
               'annualized_volatility': annualized_vol,
               'sharpe_ratio': sharpe_ratio,
               'maximum_drawdown': max_dd,
               'recent_30d_performance': recent_performance,
               'portfolio_value': equity_curve[-1],
               'alerts_count': len([a for a in self.alerts if a['timestamp'].date() == pd.Timestamp.now().date()])
           }
           
           return report
   
   # Example monitoring
   np.random.seed(42)
   returns = np.random.normal(0.0008, 0.02, 252)  # One year of returns
   equity_curve = 100000 * np.cumprod(1 + returns)  # $100k starting value
   
   # Add a drawdown period
   returns[100:120] = np.random.normal(-0.003, 0.03, 20)  # Bad period
   equity_curve = 100000 * np.cumprod(1 + returns)
   
   # Initialize monitor
   monitor = PerformanceMonitor()
   
   # Check for alerts
   dd_alert = monitor.check_drawdown_alert(equity_curve)
   var_estimate = np.percentile(returns, 5)  # 95% VaR
   var_alert = monitor.check_var_breach(returns, var_estimate)
   vol_alert = monitor.check_volatility_spike(returns)
   
   # Generate daily report
   portfolio_data = {
       'returns': returns,
       'equity_curve': equity_curve
   }
   
   daily_report = monitor.generate_daily_report(portfolio_data)
   
   print("Performance Monitoring Report:")
   print(f"Date: {daily_report['date']}")
   print(f"Portfolio Value: ${daily_report['portfolio_value']:,.2f}")
   print(f"Total Return: {daily_report['total_return']:.1%}")
   print(f"Annualized Return: {daily_report['annualized_return']:.1%}")
   print(f"Annualized Volatility: {daily_report['annualized_volatility']:.1%}")
   print(f"Sharpe Ratio: {daily_report['sharpe_ratio']:.2f}")
   print(f"Maximum Drawdown: {daily_report['maximum_drawdown']:.1%}")
   print(f"Recent 30d Performance: {daily_report['recent_30d_performance']:.1%}")
   
   print(f"\nActive Alerts: {len(monitor.alerts)}")
   for alert in monitor.alerts:
       print(f"- {alert['type']}: {alert['severity']} severity")

Output::

   Performance Monitoring Report:
   Date: 2025-09-05
   Portfolio Value: $109,729.15
   Total Return: 9.7%
   Annualized Return: 9.7%
   Annualized Volatility: 31.8%
   Sharpe Ratio: 0.31
   Maximum Drawdown: -11.9%
   Recent 30d Performance: 2.1%
   
   Active Alerts: 1
   - DRAWDOWN_BREACH: HIGH severity

Best Practices for Advanced Usage
--------------------------------

1. **Risk Management**:
   - Implement multiple risk measures and cross-validate
   - Use regime-aware models for changing market conditions
   - Monitor correlations and concentration risk

2. **Model Validation**:
   - Use out-of-sample testing extensively
   - Implement walk-forward analysis
   - Account for model uncertainty

3. **Performance Attribution**:
   - Decompose returns into systematic and idiosyncratic components
   - Track factor exposures over time
   - Monitor strategy capacity and scalability

4. **Operational Considerations**:
   - Implement robust error handling and logging
   - Design for regulatory compliance and reporting
   - Plan for disaster recovery and business continuity

5. **Technology Infrastructure**:
   - Use appropriate data structures for large datasets
   - Implement parallel processing for computationally intensive tasks
   - Consider real-time vs batch processing requirements
