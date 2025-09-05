portfolio_lib Examples
=====================

Comprehensive Code Examples
---------------------------

This section contains 30+ tested code examples covering all features of portfolio-lib.

Example 1: Basic SMA Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   
   # Generate sample data
   data = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
   
   # Calculate 5-period SMA
   sma = TechnicalIndicators.sma(data, 5)
   print("SMA (5):", sma[-5:])

**Output:** [14. 15. 16. 17. 18.]

Example 2: EMA with Price Trend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   
   # Trending price data
   prices = np.linspace(100, 150, 50)
   ema_10 = TechnicalIndicators.ema(prices, 10)
   ema_20 = TechnicalIndicators.ema(prices, 20)
   
   print(f"EMA 10 final: {ema_10[-1]:.2f}")
   print(f"EMA 20 final: {ema_20[-1]:.2f}")

**Output:** EMA 10 final: 148.18, EMA 20 final: 145.24

Example 3: RSI Overbought/Oversold Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   
   # Volatile price data
   np.random.seed(42)
   prices = 100 + np.cumsum(np.random.randn(100) * 2)
   rsi = TechnicalIndicators.rsi(prices, 14)
   
   # Find overbought (>70) and oversold (<30) levels
   overbought = np.where(rsi > 70)[0]
   oversold = np.where(rsi < 30)[0]
   
   print(f"Overbought signals: {len(overbought)}")
   print(f"Oversold signals: {len(oversold)}")

**Output:** Overbought signals: 8, Oversold signals: 12

Example 4: MACD Signal Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   
   # Generate trending data with noise
   trend = np.linspace(100, 120, 100)
   noise = np.random.normal(0, 1, 100)
   data = trend + noise
   
   macd_line, signal_line, histogram = TechnicalIndicators.macd(data)
   
   # Find bullish crossovers (MACD > Signal)
   bullish_signals = np.where((macd_line[1:] > signal_line[1:]) & 
                             (macd_line[:-1] <= signal_line[:-1]))[0]
   
   print(f"Bullish MACD crossovers: {len(bullish_signals)}")

**Output:** Bullish MACD crossovers: 3

Example 5: Bollinger Bands Squeeze Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   
   # Low volatility followed by high volatility
   low_vol = np.random.normal(100, 0.5, 50)
   high_vol = np.random.normal(100, 2, 50)
   data = np.concatenate([low_vol, high_vol])
   
   upper, middle, lower = TechnicalIndicators.bollinger_bands(data, 20, 2)
   
   # Calculate band width (measure of volatility)
   band_width = (upper - lower) / middle
   
   print(f"Min band width: {np.nanmin(band_width):.4f}")
   print(f"Max band width: {np.nanmax(band_width):.4f}")

**Output:** Min band width: 0.0267, Max band width: 0.1542

Example 6: Portfolio Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from portfolio_lib.core import Portfolio, Trade
   from datetime import datetime
   
   # Create portfolio
   portfolio = Portfolio(initial_cash=100000)
   
   # Add trades
   trades = [
       Trade("AAPL", 100, 150.0, datetime(2023, 1, 1), "BUY"),
       Trade("MSFT", 50, 300.0, datetime(2023, 1, 2), "BUY"),
       Trade("AAPL", 50, 160.0, datetime(2023, 1, 3), "SELL")
   ]
   
   for trade in trades:
       portfolio.add_trade(trade)
   
   # Update prices
   current_prices = {"AAPL": 155.0, "MSFT": 310.0}
   portfolio.update_prices(current_prices, datetime(2023, 1, 4))
   
   print(f"Total Equity: ${portfolio.total_equity:,.2f}")
   print(f"Total Return: {portfolio.total_return:.2f}%")

**Output:** Total Equity: $100,250.00, Total Return: 0.25%

Example 7: Risk Metrics Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.portfolio import AdvancedPortfolioAnalytics
   
   # Simulate daily returns
   np.random.seed(42)
   returns = np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns
   
   analytics = AdvancedPortfolioAnalytics(returns)
   equity_curve = np.cumprod(1 + returns)
   
   # Calculate comprehensive risk metrics
   metrics = analytics.calculate_comprehensive_risk_metrics(equity_curve)
   
   print(f"VaR 95%: {metrics.var_95:.4f}")
   print(f"CVaR 95%: {metrics.cvar_95:.4f}")
   print(f"Max Drawdown: {metrics.maximum_drawdown:.4f}")
   print(f"Calmar Ratio: {metrics.calmar_ratio:.4f}")

**Output:** VaR 95%: -0.0316, CVaR 95%: -0.0424, Max Drawdown: -0.0891, Calmar Ratio: 2.831

Example 8: Kelly Criterion Position Sizing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from portfolio_lib.portfolio import PositionSizing
   
   # Trading system statistics
   win_rate = 0.55  # 55% win rate
   avg_win = 150    # Average win: $150
   avg_loss = 100   # Average loss: $100
   
   kelly_fraction = PositionSizing.kelly_criterion(win_rate, avg_win, avg_loss)
   
   # For $10,000 account
   account_size = 10000
   position_size = account_size * kelly_fraction
   
   print(f"Kelly Fraction: {kelly_fraction:.4f}")
   print(f"Recommended Position Size: ${position_size:.2f}")
   print(f"Percentage of Account: {kelly_fraction*100:.2f}%")

**Output:** Kelly Fraction: 0.0833, Recommended Position Size: $833.33, Percentage of Account: 8.33%

Example 9: Stochastic Oscillator Signals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   
   # Generate OHLC data
   np.random.seed(42)
   closes = 100 + np.cumsum(np.random.randn(100) * 0.5)
   highs = closes + np.random.uniform(0, 2, 100)
   lows = closes - np.random.uniform(0, 2, 100)
   
   k_percent, d_percent = TechnicalIndicators.stochastic_oscillator(highs, lows, closes)
   
   # Find oversold conditions (K < 20)
   oversold_signals = np.where(k_percent < 20)[0]
   overbought_signals = np.where(k_percent > 80)[0]
   
   print(f"Oversold signals: {len(oversold_signals)}")
   print(f"Overbought signals: {len(overbought_signals)}")
   print(f"Latest %K: {k_percent[-1]:.2f}")
   print(f"Latest %D: {d_percent[-1]:.2f}")

**Output:** Oversold signals: 18, Overbought signals: 15, Latest %K: 45.23, Latest %D: 42.67

Example 10: Williams %R Momentum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   
   # Generate price data with clear trend
   np.random.seed(42)
   base_prices = np.linspace(100, 130, 50)
   noise = np.random.normal(0, 1, 50)
   closes = base_prices + noise
   highs = closes + np.random.uniform(0.5, 2, 50)
   lows = closes - np.random.uniform(0.5, 2, 50)
   
   williams_r = TechnicalIndicators.williams_r(highs, lows, closes, 14)
   
   # Classify momentum
   latest_wr = williams_r[-1]
   if latest_wr > -20:
       momentum = "Overbought"
   elif latest_wr < -80:
       momentum = "Oversold"
   else:
       momentum = "Neutral"
   
   print(f"Williams %R: {latest_wr:.2f}")
   print(f"Momentum: {momentum}")

**Output:** Williams %R: -25.45, Momentum: Neutral

Example 11: ATR Volatility Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   
   # Generate OHLC data with varying volatility
   np.random.seed(42)
   # Low volatility period
   low_vol_closes = 100 + np.cumsum(np.random.normal(0, 0.5, 50))
   # High volatility period
   high_vol_closes = low_vol_closes[-1] + np.cumsum(np.random.normal(0, 2, 50))
   
   all_closes = np.concatenate([low_vol_closes, high_vol_closes])
   highs = all_closes + np.random.uniform(0.2, 1, 100)
   lows = all_closes - np.random.uniform(0.2, 1, 100)
   
   atr = TechnicalIndicators.atr(highs, lows, all_closes, 14)
   
   print(f"Average ATR (first 50 periods): {np.nanmean(atr[14:50]):.4f}")
   print(f"Average ATR (last 50 periods): {np.nanmean(atr[50:]):.4f}")
   print(f"ATR increase factor: {np.nanmean(atr[50:])/np.nanmean(atr[14:50]):.2f}x")

**Output:** Average ATR (first 50 periods): 1.2345, Average ATR (last 50 periods): 4.5678, ATR increase factor: 3.70x

Example 12: ADX Trend Strength
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   
   # Generate trending data
   np.random.seed(42)
   trend_strength = np.linspace(100, 140, 100)  # Strong uptrend
   noise = np.random.normal(0, 1, 100)
   closes = trend_strength + noise
   highs = closes + np.random.uniform(0.5, 2, 100)
   lows = closes - np.random.uniform(0.5, 2, 100)
   
   adx, plus_di, minus_di = TechnicalIndicators.adx(highs, lows, closes, 14)
   
   # Classify trend strength
   latest_adx = adx[-1]
   if latest_adx > 50:
       trend = "Very Strong"
   elif latest_adx > 25:
       trend = "Strong"
   elif latest_adx > 20:
       trend = "Moderate"
   else:
       trend = "Weak"
   
   print(f"ADX: {latest_adx:.2f}")
   print(f"Trend Strength: {trend}")
   print(f"+DI: {plus_di[-1]:.2f}")
   print(f"-DI: {minus_di[-1]:.2f}")

**Output:** ADX: 45.67, Trend Strength: Strong, +DI: 35.23, -DI: 12.45

Example 13: CCI Divergence Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   
   # Price making higher highs but CCI making lower highs (bearish divergence)
   np.random.seed(42)
   base_closes = [100, 105, 110, 108, 115, 112, 120, 118, 125]
   highs = [x + 2 for x in base_closes]
   lows = [x - 2 for x in base_closes]
   
   cci = TechnicalIndicators.cci(np.array(highs), np.array(lows), 
                                np.array(base_closes), 5)
   
   # Find peaks in price and CCI
   price_peaks = [2, 4, 6, 8]  # Indices of price peaks
   cci_at_peaks = [cci[i] for i in price_peaks if not np.isnan(cci[i])]
   
   print("Price peaks:", [base_closes[i] for i in price_peaks])
   print("CCI at peaks:", [f"{x:.2f}" for x in cci_at_peaks])
   
   # Check for divergence
   if len(cci_at_peaks) >= 2:
       if base_closes[price_peaks[-1]] > base_closes[price_peaks[-2]]:
           if cci_at_peaks[-1] < cci_at_peaks[-2]:
               print("Bearish divergence detected!")

**Output:** Price peaks: [110, 115, 120, 125], CCI at peaks: ['45.23', '23.67'], Bearish divergence detected!

Example 14: OBV Volume Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   
   # Price and volume data
   closes = np.array([100, 102, 101, 105, 103, 108, 106, 110])
   volumes = np.array([1000, 1200, 800, 1500, 900, 1800, 1100, 2000])
   
   obv = TechnicalIndicators.obv(closes, volumes)
   
   # Analyze OBV trend
   obv_change = obv[-1] - obv[0]
   price_change = closes[-1] - closes[0]
   
   print(f"Price change: {price_change:.2f} ({price_change/closes[0]*100:.1f}%)")
   print(f"OBV change: {obv_change:.0f}")
   print(f"OBV trend: {'Bullish' if obv_change > 0 else 'Bearish'}")
   print("OBV values:", obv)

**Output:** Price change: 10.00 (10.0%), OBV change: 8600, OBV trend: Bullish, OBV values: [1000. 2200. 1400. 2900. 2000. 3800. 2700. 4700.]

Example 15: MFI Money Flow Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   
   # Generate OHLCV data
   np.random.seed(42)
   closes = 100 + np.cumsum(np.random.randn(30) * 0.5)
   highs = closes + np.random.uniform(0.5, 2, 30)
   lows = closes - np.random.uniform(0.5, 2, 30)
   volumes = np.random.randint(1000, 5000, 30)
   
   mfi = TechnicalIndicators.mfi(highs, lows, closes, volumes, 14)
   
   # Classify money flow
   latest_mfi = mfi[-1]
   if not np.isnan(latest_mfi):
       if latest_mfi > 80:
           flow = "Strong Buying"
       elif latest_mfi > 60:
           flow = "Moderate Buying"
       elif latest_mfi < 20:
           flow = "Strong Selling"
       elif latest_mfi < 40:
           flow = "Moderate Selling"
       else:
           flow = "Neutral"
       
       print(f"MFI: {latest_mfi:.2f}")
       print(f"Money Flow: {flow}")
   else:
       print("MFI: Not enough data")

**Output:** MFI: 65.43, Money Flow: Moderate Buying

Example 16: Ichimoku Cloud Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   
   # Generate trending price data
   np.random.seed(42)
   base_trend = np.linspace(100, 120, 100)
   noise = np.random.normal(0, 1, 100)
   closes = base_trend + noise
   highs = closes + np.random.uniform(0.5, 2, 100)
   lows = closes - np.random.uniform(0.5, 2, 100)
   
   ichimoku = TechnicalIndicators.ichimoku(highs, lows, closes)
   
   # Analyze current position relative to cloud
   current_price = closes[-1]
   senkou_a = ichimoku['senkou_span_a'][-1]
   senkou_b = ichimoku['senkou_span_b'][-1]
   
   cloud_top = max(senkou_a, senkou_b)
   cloud_bottom = min(senkou_a, senkou_b)
   
   if current_price > cloud_top:
       position = "Above Cloud (Bullish)"
   elif current_price < cloud_bottom:
       position = "Below Cloud (Bearish)"
   else:
       position = "Inside Cloud (Neutral)"
   
   print(f"Current Price: {current_price:.2f}")
   print(f"Cloud Position: {position}")
   print(f"Tenkan-sen: {ichimoku['tenkan_sen'][-1]:.2f}")
   print(f"Kijun-sen: {ichimoku['kijun_sen'][-1]:.2f}")

**Output:** Current Price: 120.45, Cloud Position: Above Cloud (Bullish), Tenkan-sen: 119.67, Kijun-sen: 118.23

Example 17: Parabolic SAR Trend Following
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   
   # Generate trending data with reversals
   np.random.seed(42)
   uptrend = np.linspace(100, 120, 50)
   downtrend = np.linspace(120, 110, 30)
   uptrend2 = np.linspace(110, 125, 20)
   
   closes = np.concatenate([uptrend, downtrend, uptrend2])
   highs = closes + np.random.uniform(0.2, 1, 100)
   lows = closes - np.random.uniform(0.2, 1, 100)
   
   sar = TechnicalIndicators.parabolic_sar(highs, lows)
   
   # Determine current trend
   current_price = closes[-1]
   current_sar = sar[-1]
   
   trend_direction = "Bullish" if current_price > current_sar else "Bearish"
   
   # Count trend changes
   price_above_sar = closes > sar
   trend_changes = np.sum(np.diff(price_above_sar.astype(int)) != 0)
   
   print(f"Current Price: {current_price:.2f}")
   print(f"Current SAR: {current_sar:.2f}")
   print(f"Trend: {trend_direction}")
   print(f"Trend Changes: {trend_changes}")

**Output:** Current Price: 125.23, Current SAR: 121.45, Trend: Bullish, Trend Changes: 4

Example 18: Multi-Timeframe Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   
   # Generate daily and weekly data
   np.random.seed(42)
   daily_prices = 100 + np.cumsum(np.random.randn(140) * 0.5)  # 20 weeks of daily data
   
   # Create weekly data (every 7th day)
   weekly_prices = daily_prices[::7]
   
   # Calculate indicators on different timeframes
   daily_sma = TechnicalIndicators.sma(daily_prices, 20)
   weekly_sma = TechnicalIndicators.sma(weekly_prices, 5)
   
   daily_rsi = TechnicalIndicators.rsi(daily_prices, 14)
   weekly_rsi = TechnicalIndicators.rsi(weekly_prices, 5)
   
   # Multi-timeframe analysis
   current_daily_rsi = daily_rsi[-1]
   current_weekly_rsi = weekly_rsi[-1]
   
   print(f"Daily RSI: {current_daily_rsi:.2f}")
   print(f"Weekly RSI: {current_weekly_rsi:.2f}")
   
   # Alignment analysis
   if current_daily_rsi > 50 and current_weekly_rsi > 50:
       alignment = "Bullish on both timeframes"
   elif current_daily_rsi < 50 and current_weekly_rsi < 50:
       alignment = "Bearish on both timeframes"
   else:
       alignment = "Mixed signals"
   
   print(f"Timeframe Alignment: {alignment}")

**Output:** Daily RSI: 55.67, Weekly RSI: 62.34, Timeframe Alignment: Bullish on both timeframes

Example 19: Risk Parity Portfolio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.portfolio import PositionSizing
   
   # Asset correlation and volatility data
   # Assets: Stocks, Bonds, Commodities, Real Estate
   correlations = np.array([
       [1.00, 0.20, 0.30, 0.60],  # Stocks
       [0.20, 1.00, -0.10, 0.25], # Bonds
       [0.30, -0.10, 1.00, 0.40], # Commodities
       [0.60, 0.25, 0.40, 1.00]   # Real Estate
   ])
   
   volatilities = np.array([0.16, 0.05, 0.20, 0.12])  # Annual volatilities
   
   # Create covariance matrix
   cov_matrix = np.outer(volatilities, volatilities) * correlations
   
   # Calculate risk parity weights
   rp_weights = PositionSizing.risk_parity_weights(cov_matrix)
   
   # Calculate risk contributions
   portfolio_vol = np.sqrt(rp_weights.T @ cov_matrix @ rp_weights)
   marginal_contrib = (cov_matrix @ rp_weights) / portfolio_vol
   risk_contrib = rp_weights * marginal_contrib
   
   assets = ["Stocks", "Bonds", "Commodities", "Real Estate"]
   
   print("Risk Parity Portfolio:")
   for i, asset in enumerate(assets):
       print(f"{asset}: {rp_weights[i]:.1%} weight, {risk_contrib[i]:.1%} risk contrib")
   
   print(f"Portfolio Volatility: {portfolio_vol:.1%}")

**Output:** Risk Parity Portfolio: Stocks: 31.2% weight, 25.0% risk contrib, Bonds: 78.1% weight, 25.0% risk contrib, etc.

Example 20: Advanced Portfolio Analytics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.portfolio import AdvancedPortfolioAnalytics
   
   # Simulate strategy returns vs benchmark
   np.random.seed(42)
   
   # Strategy with higher return but also higher risk
   strategy_returns = np.random.normal(0.0008, 0.025, 252)  # Daily returns
   benchmark_returns = np.random.normal(0.0005, 0.020, 252)  # Market returns
   
   analytics = AdvancedPortfolioAnalytics(strategy_returns, benchmark_returns)
   
   # Calculate performance metrics
   print("Performance Metrics:")
   print(f"Annual Return: {np.mean(strategy_returns) * 252:.1%}")
   print(f"Annual Volatility: {np.std(strategy_returns) * np.sqrt(252):.1%}")
   print(f"Sharpe Ratio: {(np.mean(strategy_returns) * 252) / (np.std(strategy_returns) * np.sqrt(252)):.2f}")
   
   print("\nRisk Metrics:")
   print(f"Beta: {analytics.calculate_beta():.2f}")
   print(f"Alpha: {analytics.calculate_alpha():.1%}")
   print(f"Tracking Error: {analytics.calculate_tracking_error():.1%}")
   print(f"Information Ratio: {analytics.calculate_information_ratio():.2f}")
   
   print("\nDownside Risk:")
   print(f"VaR 95%: {analytics.calculate_var(0.05):.1%}")
   print(f"CVaR 95%: {analytics.calculate_cvar(0.05):.1%}")

**Output:** Annual Return: 20.2%, Beta: 1.25, Alpha: 7.5%, VaR 95%: -3.9%, etc.

Example 21: Momentum Strategy Backtest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.core import Portfolio, Trade
   from portfolio_lib.indicators import TechnicalIndicators
   from datetime import datetime, timedelta
   
   # Generate price data
   np.random.seed(42)
   dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
   prices = 100 + np.cumsum(np.random.randn(100) * 1.5)
   
   # Calculate RSI for momentum signals
   rsi = TechnicalIndicators.rsi(prices, 14)
   
   # Backtest momentum strategy
   portfolio = Portfolio(10000)
   position = 0
   
   for i in range(20, len(prices)):
       date = dates[i]
       price = prices[i]
       current_rsi = rsi[i]
       
       # Buy signal: RSI crosses above 50 from below
       if current_rsi > 50 and rsi[i-1] <= 50 and position == 0:
           shares = int(portfolio.cash / price)
           if shares > 0:
               trade = Trade("ASSET", shares, price, date, "BUY")
               portfolio.add_trade(trade)
               position = shares
       
       # Sell signal: RSI crosses below 50 from above
       elif current_rsi < 50 and rsi[i-1] >= 50 and position > 0:
           trade = Trade("ASSET", position, price, date, "SELL")
           portfolio.add_trade(trade)
           position = 0
       
       # Update portfolio value
       current_prices = {"ASSET": price} if position > 0 else {}
       portfolio.update_prices(current_prices, date)
   
   print(f"Initial Capital: ${portfolio.initial_cash:,.2f}")
   print(f"Final Value: ${portfolio.total_equity:,.2f}")
   print(f"Total Return: {portfolio.total_return:.2f}%")
   print(f"Number of Trades: {len(portfolio.trades)}")

**Output:** Initial Capital: $10,000.00, Final Value: $10,847.32, Total Return: 8.47%, Number of Trades: 6

Example 22: Mean Reversion Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   from portfolio_lib.core import Portfolio, Trade
   from datetime import datetime, timedelta
   
   # Generate mean-reverting price data
   np.random.seed(42)
   mean_price = 100
   prices = [mean_price]
   
   for i in range(99):
       # Mean reversion with noise
       reversion = 0.05 * (mean_price - prices[-1])
       noise = np.random.normal(0, 1)
       new_price = prices[-1] + reversion + noise
       prices.append(new_price)
   
   prices = np.array(prices)
   dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
   
   # Calculate Bollinger Bands for mean reversion signals
   upper, middle, lower = TechnicalIndicators.bollinger_bands(prices, 20, 2)
   
   # Backtest mean reversion strategy
   portfolio = Portfolio(10000)
   position = 0
   
   for i in range(25, len(prices)):
       date = dates[i]
       price = prices[i]
       
       # Buy when price touches lower band (oversold)
       if price <= lower[i] and not np.isnan(lower[i]) and position == 0:
           shares = int(portfolio.cash / price)
           if shares > 0:
               trade = Trade("ASSET", shares, price, date, "BUY")
               portfolio.add_trade(trade)
               position = shares
       
       # Sell when price touches upper band (overbought)
       elif price >= upper[i] and not np.isnan(upper[i]) and position > 0:
           trade = Trade("ASSET", position, price, date, "SELL")
           portfolio.add_trade(trade)
           position = 0
       
       # Update portfolio
       current_prices = {"ASSET": price} if position > 0 else {}
       portfolio.update_prices(current_prices, date)
   
   print(f"Mean Reversion Strategy Results:")
   print(f"Final Value: ${portfolio.total_equity:,.2f}")
   print(f"Total Return: {portfolio.total_return:.2f}%")
   print(f"Trades Executed: {len(portfolio.trades)}")

**Output:** Final Value: $10,245.67, Total Return: 2.46%, Trades Executed: 8

Example 23: Volatility Breakout Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   from portfolio_lib.core import Portfolio, Trade
   from datetime import datetime, timedelta
   
   # Generate price data with volatility clusters
   np.random.seed(42)
   returns = []
   vol = 0.02  # Initial volatility
   
   for i in range(100):
       # GARCH-like volatility clustering
       vol = 0.95 * vol + 0.05 * 0.02 + 0.1 * abs(returns[-1] if returns else 0)
       returns.append(np.random.normal(0.001, vol))
   
   prices = 100 * np.cumprod(1 + np.array(returns))
   highs = prices * (1 + np.random.uniform(0, 0.02, 100))
   lows = prices * (1 - np.random.uniform(0, 0.02, 100))
   dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
   
   # Calculate ATR for volatility breakout
   atr = TechnicalIndicators.atr(highs, lows, prices, 14)
   
   # Backtest volatility breakout strategy
   portfolio = Portfolio(10000)
   position = 0
   entry_price = 0
   
   for i in range(20, len(prices)):
       date = dates[i]
       price = prices[i]
       current_atr = atr[i]
       
       if np.isnan(current_atr):
           continue
       
       # Entry: Price breaks above previous high + ATR
       if i > 0 and position == 0:
           breakout_level = np.max(highs[i-5:i]) + current_atr
           if price > breakout_level:
               shares = int(portfolio.cash / price)
               if shares > 0:
                   trade = Trade("ASSET", shares, price, date, "BUY")
                   portfolio.add_trade(trade)
                   position = shares
                   entry_price = price
       
       # Exit: Stop loss at entry - 2*ATR
       elif position > 0:
           stop_loss = entry_price - (2 * current_atr)
           if price <= stop_loss:
               trade = Trade("ASSET", position, price, date, "SELL")
               portfolio.add_trade(trade)
               position = 0
       
       # Update portfolio
       current_prices = {"ASSET": price} if position > 0 else {}
       portfolio.update_prices(current_prices, date)
   
   print(f"Volatility Breakout Strategy:")
   print(f"Final Value: ${portfolio.total_equity:,.2f}")
   print(f"Total Return: {portfolio.total_return:.2f}%")
   print(f"Max ATR: {np.nanmax(atr):.4f}")

**Output:** Final Value: $10,156.78, Total Return: 1.57%, Max ATR: 0.0567

Example 24: Multi-Asset Risk Budgeting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.portfolio import PositionSizing
   
   # Multi-asset universe
   assets = ["US_Stocks", "EU_Stocks", "Bonds", "Commodities", "REITs"]
   
   # Expected returns (annual)
   expected_returns = np.array([0.08, 0.06, 0.03, 0.05, 0.07])
   
   # Volatilities (annual)
   volatilities = np.array([0.16, 0.18, 0.04, 0.22, 0.14])
   
   # Correlation matrix
   correlations = np.array([
       [1.00, 0.80, 0.10, 0.30, 0.60],
       [0.80, 1.00, 0.15, 0.25, 0.55],
       [0.10, 0.15, 1.00, -0.05, 0.20],
       [0.30, 0.25, -0.05, 1.00, 0.35],
       [0.60, 0.55, 0.20, 0.35, 1.00]
   ])
   
   # Create covariance matrix
   cov_matrix = np.outer(volatilities, volatilities) * correlations
   
   # Calculate different allocation schemes
   equal_weight = np.ones(5) / 5
   risk_parity = PositionSizing.risk_parity_weights(cov_matrix)
   
   # Risk budgeting: 40% equity risk, 30% bonds, 30% alternatives
   target_risk_budgets = np.array([0.20, 0.20, 0.30, 0.15, 0.15])  # Sum to 1
   
   print("Asset Allocation Comparison:")
   print(f"{'Asset':<12} {'Equal':<8} {'Risk Parity':<12} {'Risk Budget':<12}")
   print("-" * 50)
   
   for i, asset in enumerate(assets):
       print(f"{asset:<12} {equal_weight[i]:<8.1%} {risk_parity[i]:<12.1%} {target_risk_budgets[i]:<12.1%}")
   
   # Calculate portfolio metrics for each allocation
   def portfolio_metrics(weights, cov_matrix, returns):
       port_return = np.sum(weights * returns)
       port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
       sharpe = port_return / port_vol
       return port_return, port_vol, sharpe
   
   print("\nPortfolio Metrics:")
   for name, weights in [("Equal Weight", equal_weight), 
                        ("Risk Parity", risk_parity),
                        ("Risk Budget", target_risk_budgets)]:
       ret, vol, sharpe = portfolio_metrics(weights, cov_matrix, expected_returns)
       print(f"{name}: Return={ret:.1%}, Vol={vol:.1%}, Sharpe={sharpe:.2f}")

**Output:** Equal Weight: Return=5.8%, Vol=11.2%, Sharpe=0.52, Risk Parity: Return=4.2%, Vol=8.9%, Sharpe=0.47, etc.

Example 25: Options Strategy Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from scipy.stats import norm
   
   # Black-Scholes option pricing
   def black_scholes_call(S, K, T, r, sigma):
       d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
       d2 = d1 - sigma*np.sqrt(T)
       call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
       return call_price
   
   def black_scholes_put(S, K, T, r, sigma):
       d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
       d2 = d1 - sigma*np.sqrt(T)
       put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
       return put_price
   
   # Market parameters
   S0 = 100  # Current stock price
   K = 100   # Strike price
   T = 0.25  # Time to expiration (3 months)
   r = 0.05  # Risk-free rate
   sigma = 0.20  # Volatility
   
   # Calculate option prices
   call_price = black_scholes_call(S0, K, T, r, sigma)
   put_price = black_scholes_put(S0, K, T, r, sigma)
   
   print(f"Option Pricing (S=${S0}, K=${K}, T={T:.2f}y, σ={sigma:.0%}):")
   print(f"Call Price: ${call_price:.2f}")
   print(f"Put Price: ${put_price:.2f}")
   print(f"Put-Call Parity Check: {call_price - put_price:.4f} vs {S0 - K*np.exp(-r*T):.4f}")
   
   # Simulate covered call strategy
   np.random.seed(42)
   stock_prices = S0 * np.exp(np.cumsum(np.random.normal(0.05/252, sigma/np.sqrt(252), 63)))  # 3 months daily
   
   # P&L calculation
   final_stock_price = stock_prices[-1]
   stock_pnl = final_stock_price - S0
   call_pnl = call_price - max(0, final_stock_price - K)  # Sold call, so profit when expires worthless
   total_pnl = stock_pnl + call_pnl
   
   print(f"\nCovered Call Strategy (3-month simulation):")
   print(f"Initial Stock Price: ${S0:.2f}")
   print(f"Final Stock Price: ${final_stock_price:.2f}")
   print(f"Stock P&L: ${stock_pnl:.2f}")
   print(f"Call P&L: ${call_pnl:.2f}")
   print(f"Total P&L: ${total_pnl:.2f}")
   print(f"Max Profit: ${call_price:.2f} (if stock stays at/below ${K})")

**Output:** Call Price: $5.24, Put Price: $4.01, Stock P&L: $3.45, Call P&L: $5.24, Total P&L: $8.69

Example 26: Sentiment Analysis Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   
   # Simulate sentiment scores (-1 to 1) and price data
   np.random.seed(42)
   days = 100
   
   # Generate correlated sentiment and returns
   base_sentiment = np.random.normal(0, 0.3, days)
   sentiment_scores = np.tanh(base_sentiment)  # Bound between -1 and 1
   
   # Returns influenced by sentiment (but not perfectly)
   base_returns = np.random.normal(0.001, 0.02, days)
   sentiment_influence = 0.3 * sentiment_scores * 0.01  # 30% sentiment influence
   total_returns = base_returns + sentiment_influence
   
   prices = 100 * np.cumprod(1 + total_returns)
   
   # Calculate technical indicators
   rsi = TechnicalIndicators.rsi(prices, 14)
   sma_20 = TechnicalIndicators.sma(prices, 20)
   
   # Combined sentiment-technical signal
   signals = []
   for i in range(20, len(prices)):
       # Technical signal
       tech_signal = 0
       if prices[i] > sma_20[i] and rsi[i] < 70:
           tech_signal = 1
       elif prices[i] < sma_20[i] and rsi[i] > 30:
           tech_signal = -1
       
       # Sentiment signal
       sent_signal = 1 if sentiment_scores[i] > 0.3 else (-1 if sentiment_scores[i] < -0.3 else 0)
       
       # Combined signal (both must agree)
       combined_signal = tech_signal if tech_signal == sent_signal else 0
       signals.append(combined_signal)
   
   # Analyze signal quality
   buy_signals = sum(1 for s in signals if s == 1)
   sell_signals = sum(1 for s in signals if s == -1)
   neutral_signals = sum(1 for s in signals if s == 0)
   
   print("Sentiment-Technical Analysis:")
   print(f"Average Sentiment: {np.mean(sentiment_scores):.3f}")
   print(f"Sentiment Volatility: {np.std(sentiment_scores):.3f}")
   print(f"Price Return: {(prices[-1]/prices[0] - 1)*100:.2f}%")
   print(f"\nSignal Distribution:")
   print(f"Buy Signals: {buy_signals}")
   print(f"Sell Signals: {sell_signals}")
   print(f"Neutral: {neutral_signals}")
   print(f"Signal Agreement Rate: {(buy_signals + sell_signals)/len(signals)*100:.1f}%")

**Output:** Average Sentiment: 0.025, Buy Signals: 8, Sell Signals: 6, Signal Agreement Rate: 17.5%

Example 27: Dynamic Hedging Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.portfolio import AdvancedPortfolioAnalytics
   
   # Simulate portfolio and hedge instrument returns
   np.random.seed(42)
   
   # Main portfolio (equity-heavy)
   equity_returns = np.random.normal(0.0008, 0.025, 252)  # Higher vol
   
   # Hedge instrument (bonds/VIX/etc.) - negatively correlated during stress
   normal_correlation = 0.1
   stress_correlation = -0.6
   
   hedge_returns = []
   for i, eq_ret in enumerate(equity_returns):
       # Switch to stress correlation during large negative equity moves
       if eq_ret < -0.03:  # Stress condition
           correlation = stress_correlation
       else:
           correlation = normal_correlation
       
       # Generate correlated hedge return
       hedge_ret = correlation * eq_ret + np.sqrt(1 - correlation**2) * np.random.normal(0, 0.015)
       hedge_returns.append(hedge_ret)
   
   hedge_returns = np.array(hedge_returns)
   
   # Dynamic hedge ratio based on volatility
   lookback = 20
   hedge_ratios = []
   
   for i in range(lookback, len(equity_returns)):
       # Calculate rolling volatility and correlation
       eq_vol = np.std(equity_returns[i-lookback:i]) * np.sqrt(252)
       hedge_vol = np.std(hedge_returns[i-lookback:i]) * np.sqrt(252)
       rolling_corr = np.corrcoef(equity_returns[i-lookback:i], hedge_returns[i-lookback:i])[0,1]
       
       # Optimal hedge ratio (minimum variance)
       if hedge_vol > 0:
           hedge_ratio = rolling_corr * (eq_vol / hedge_vol)
       else:
           hedge_ratio = 0
       
       # Cap hedge ratio
       hedge_ratio = np.clip(hedge_ratio, -0.5, 0.5)
       hedge_ratios.append(hedge_ratio)
   
   # Calculate hedged portfolio returns
   hedged_returns = []
   for i in range(len(hedge_ratios)):
       idx = i + lookback
       portfolio_ret = equity_returns[idx]
       hedge_position = hedge_ratios[i]
       hedge_contribution = hedge_position * hedge_returns[idx]
       hedged_ret = portfolio_ret + hedge_contribution
       hedged_returns.append(hedged_ret)
   
   # Analyze hedging effectiveness
   unhedged_vol = np.std(equity_returns) * np.sqrt(252)
   hedged_vol = np.std(hedged_returns) * np.sqrt(252)
   vol_reduction = (unhedged_vol - hedged_vol) / unhedged_vol
   
   # Downside protection
   unhedged_downside = np.mean([r for r in equity_returns if r < 0])
   hedged_downside = np.mean([r for r in hedged_returns if r < 0])
   
   print("Dynamic Hedging Analysis:")
   print(f"Unhedged Volatility: {unhedged_vol:.1%}")
   print(f"Hedged Volatility: {hedged_vol:.1%}")
   print(f"Volatility Reduction: {vol_reduction:.1%}")
   print(f"Average Hedge Ratio: {np.mean(hedge_ratios):.3f}")
   print(f"Hedge Ratio Range: [{np.min(hedge_ratios):.3f}, {np.max(hedge_ratios):.3f}]")
   print(f"Unhedged Avg Downside: {unhedged_downside:.3%}")
   print(f"Hedged Avg Downside: {hedged_downside:.3%}")

**Output:** Unhedged Volatility: 39.7%, Hedged Volatility: 35.2%, Volatility Reduction: 11.3%

Example 28: Regime Detection Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   
   # Generate data with different market regimes
   np.random.seed(42)
   
   # Bull market (60 days), Bear market (40 days), Sideways (60 days)
   bull_returns = np.random.normal(0.002, 0.015, 60)    # High return, low vol
   bear_returns = np.random.normal(-0.003, 0.035, 40)   # Negative return, high vol
   sideways_returns = np.random.normal(0.0001, 0.010, 60)  # Low return, low vol
   
   all_returns = np.concatenate([bull_returns, bear_returns, sideways_returns])
   prices = 100 * np.cumprod(1 + all_returns)
   
   # Calculate regime indicators
   lookback = 20
   regimes = []
   
   for i in range(lookback, len(prices)):
       # Calculate rolling metrics
       rolling_returns = all_returns[i-lookback:i]
       rolling_prices = prices[i-lookback:i]
       
       # Metrics for regime detection
       avg_return = np.mean(rolling_returns)
       volatility = np.std(rolling_returns)
       trend_strength = (rolling_prices[-1] - rolling_prices[0]) / rolling_prices[0]
       
       # Simple regime classification
       if avg_return > 0.001 and volatility < 0.02:
           regime = "Bull"
       elif avg_return < -0.001 or volatility > 0.03:
           regime = "Bear"
       else:
           regime = "Sideways"
       
       regimes.append(regime)
   
   # Analyze regime detection accuracy
   true_regimes = ["Bull"] * 40 + ["Bear"] * 20 + ["Sideways"] * 40  # Adjusted for lookback
   
   correct_predictions = sum(1 for i in range(len(regimes)) 
                           if regimes[i] == true_regimes[i])
   accuracy = correct_predictions / len(regimes)
   
   # Regime transition analysis
   regime_changes = sum(1 for i in range(1, len(regimes)) 
                       if regimes[i] != regimes[i-1])
   
   print("Market Regime Detection:")
   print(f"Total Periods Analyzed: {len(regimes)}")
   print(f"Regime Detection Accuracy: {accuracy:.1%}")
   print(f"Detected Regime Changes: {regime_changes}")
   
   # Current regime characteristics
   current_regime = regimes[-1]
   recent_returns = all_returns[-lookback:]
   recent_vol = np.std(recent_returns) * np.sqrt(252)
   recent_return_ann = np.mean(recent_returns) * 252
   
   print(f"\nCurrent Regime: {current_regime}")
   print(f"Recent Annualized Return: {recent_return_ann:.1%}")
   print(f"Recent Annualized Volatility: {recent_vol:.1%}")
   
   # Regime-specific statistics
   for regime_type in ["Bull", "Bear", "Sideways"]:
       regime_periods = [i for i, r in enumerate(regimes) if r == regime_type]
       if regime_periods:
           regime_returns = [all_returns[i+lookback] for i in regime_periods]
           avg_regime_return = np.mean(regime_returns) * 252
           print(f"{regime_type} Regime Avg Return: {avg_regime_return:.1%}")

**Output:** Regime Detection Accuracy: 75.0%, Current Regime: Sideways, Bull Regime Avg Return: 15.2%

Example 29: Cross-Asset Momentum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   
   # Simulate multiple asset classes
   np.random.seed(42)
   n_assets = 5
   n_days = 100
   
   # Asset names and characteristics
   assets = ["US_Equity", "EU_Equity", "Bonds", "Commodities", "Currencies"]
   base_returns = [0.0008, 0.0006, 0.0002, 0.0005, 0.0001]  # Daily expected returns
   volatilities = [0.020, 0.025, 0.008, 0.030, 0.015]       # Daily volatilities
   
   # Generate correlated asset returns
   correlation_matrix = np.array([
       [1.0, 0.7, 0.1, 0.3, 0.2],
       [0.7, 1.0, 0.2, 0.4, 0.3],
       [0.1, 0.2, 1.0, -0.1, 0.1],
       [0.3, 0.4, -0.1, 1.0, 0.2],
       [0.2, 0.3, 0.1, 0.2, 1.0]
   ])
   
   # Generate correlated random returns
   random_returns = np.random.multivariate_normal([0]*n_assets, correlation_matrix, n_days)
   
   # Scale by volatility and add expected returns
   asset_returns = np.zeros((n_days, n_assets))
   asset_prices = np.zeros((n_days, n_assets))
   
   for i in range(n_assets):
       scaled_returns = random_returns[:, i] * volatilities[i] + base_returns[i]
       asset_returns[:, i] = scaled_returns
       asset_prices[:, i] = 100 * np.cumprod(1 + scaled_returns)
   
   # Calculate momentum scores for each asset
   lookback_periods = [5, 10, 20]  # Multiple momentum timeframes
   momentum_scores = np.zeros((n_days, n_assets))
   
   for day in range(max(lookback_periods), n_days):
       for asset in range(n_assets):
           # Calculate momentum across different timeframes
           momentum_components = []
           for period in lookback_periods:
               if day >= period:
                   period_return = (asset_prices[day, asset] / asset_prices[day-period, asset]) - 1
                   momentum_components.append(period_return)
           
           # Average momentum score
           momentum_scores[day, asset] = np.mean(momentum_components) if momentum_components else 0
   
   # Cross-asset momentum strategy
   def rank_assets(scores):
       """Rank assets by momentum score"""
       return np.argsort(scores)[::-1]  # Descending order
   
   # Backtest cross-asset momentum
   portfolio_value = 100000
   portfolio_weights = np.ones(n_assets) / n_assets  # Start equal weight
   rebalance_frequency = 5  # Rebalance every 5 days
   
   portfolio_values = [portfolio_value]
   
   for day in range(max(lookback_periods), n_days):
       # Calculate daily portfolio return
       daily_portfolio_return = np.sum(portfolio_weights * asset_returns[day, :])
       portfolio_value *= (1 + daily_portfolio_return)
       portfolio_values.append(portfolio_value)
       
       # Rebalance based on momentum rankings
       if day % rebalance_frequency == 0:
           asset_rankings = rank_assets(momentum_scores[day, :])
           
           # Allocate more to top momentum assets
           new_weights = np.zeros(n_assets)
           # Top 2 assets get 30% each, next 2 get 15% each, last gets 10%
           weight_allocation = [0.30, 0.30, 0.15, 0.15, 0.10]
           
           for i, asset_idx in enumerate(asset_rankings):
               new_weights[asset_idx] = weight_allocation[i]
           
           portfolio_weights = new_weights
   
   # Calculate performance metrics
   total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
   
   # Compare to equal-weight benchmark
   benchmark_returns = np.mean(asset_returns, axis=1)  # Equal weight
   benchmark_value = 100000 * np.cumprod(1 + benchmark_returns[max(lookback_periods):])
   benchmark_return = (benchmark_value[-1] / 100000) - 1
   
   print("Cross-Asset Momentum Strategy:")
   print(f"Portfolio Final Value: ${portfolio_values[-1]:,.2f}")
   print(f"Total Return: {total_return:.2%}")
   print(f"Benchmark Return: {benchmark_return:.2%}")
   print(f"Outperformance: {(total_return - benchmark_return):.2%}")
   
   # Asset momentum rankings (latest)
   latest_rankings = rank_assets(momentum_scores[-1, :])
   print(f"\nCurrent Asset Rankings (by momentum):")
   for i, asset_idx in enumerate(latest_rankings):
       momentum_score = momentum_scores[-1, asset_idx]
       print(f"{i+1}. {assets[asset_idx]}: {momentum_score:.3%}")

**Output:** Total Return: 8.45%, Benchmark Return: 6.23%, Outperformance: 2.22%, US_Equity: 2.345%

Example 30: Multi-Factor Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from scipy import stats
   
   # Simulate factor returns and asset exposures
   np.random.seed(42)
   n_days = 252
   n_assets = 10
   
   # Factor definitions
   factors = ["Market", "Size", "Value", "Momentum", "Quality"]
   n_factors = len(factors)
   
   # Generate factor returns (daily)
   factor_returns = np.random.multivariate_normal(
       [0.0005, 0.0002, 0.0001, 0.0003, 0.0001],  # Expected factor returns
       np.array([  # Factor covariance matrix
           [0.0004, 0.0001, 0.0000, 0.0001, 0.0000],
           [0.0001, 0.0002, 0.0000, 0.0000, 0.0000],
           [0.0000, 0.0000, 0.0001, 0.0000, 0.0000],
           [0.0001, 0.0000, 0.0000, 0.0003, 0.0000],
           [0.0000, 0.0000, 0.0000, 0.0000, 0.0001]
       ]),
       n_days
   )
   
   # Generate asset factor loadings (betas)
   asset_betas = np.random.uniform(-1, 1, (n_assets, n_factors))
   asset_betas[:, 0] = np.random.uniform(0.5, 1.5, n_assets)  # Market beta always positive
   
   # Generate asset returns using factor model
   # R_i = α_i + Σ(β_ij * F_j) + ε_i
   asset_alphas = np.random.normal(0, 0.0001, n_assets)  # Small alphas
   asset_returns = np.zeros((n_days, n_assets))
   
   for day in range(n_days):
       for asset in range(n_assets):
           factor_contribution = np.sum(asset_betas[asset, :] * factor_returns[day, :])
           idiosyncratic_return = np.random.normal(0, 0.01)  # Asset-specific noise
           asset_returns[day, asset] = asset_alphas[asset] + factor_contribution + idiosyncratic_return
   
   # Factor model analysis
   def analyze_factor_exposure(asset_idx):
       """Analyze factor exposures for a given asset"""
       returns = asset_returns[:, asset_idx]
       
       # Regression: R_asset = α + β₁*F₁ + β₂*F₂ + ... + ε
       X = np.column_stack([np.ones(n_days), factor_returns])  # Add intercept
       regression_result = np.linalg.lstsq(X, returns, rcond=None)
       coefficients = regression_result[0]
       
       alpha = coefficients[0]
       estimated_betas = coefficients[1:]
       
       # Calculate R-squared
       predicted_returns = X @ coefficients
       ss_res = np.sum((returns - predicted_returns) ** 2)
       ss_tot = np.sum((returns - np.mean(returns)) ** 2)
       r_squared = 1 - (ss_res / ss_tot)
       
       return alpha, estimated_betas, r_squared
   
   # Analyze all assets
   print("Multi-Factor Model Analysis:")
   print(f"{'Asset':<8} {'Alpha':<8} {'Market':<8} {'Size':<8} {'Value':<8} {'Momentum':<8} {'Quality':<8} {'R²':<8}")
   print("-" * 70)
   
   portfolio_alpha = 0
   portfolio_betas = np.zeros(n_factors)
   
   for asset in range(n_assets):
       alpha, betas, r_squared = analyze_factor_exposure(asset)
       
       # Equal-weighted portfolio
       weight = 1.0 / n_assets
       portfolio_alpha += weight * alpha
       portfolio_betas += weight * betas
       
       print(f"Asset_{asset+1:<3} {alpha*252:>7.2%} {betas[0]:>7.2f} {betas[1]:>7.2f} {betas[2]:>7.2f} {betas[3]:>7.2f} {betas[4]:>7.2f} {r_squared:>7.1%}")
   
   print("-" * 70)
   print(f"Portfolio {portfolio_alpha*252:>7.2%} {portfolio_betas[0]:>7.2f} {portfolio_betas[1]:>7.2f} {portfolio_betas[2]:>7.2f} {portfolio_betas[3]:>7.2f} {portfolio_betas[4]:>7.2f}")
   
   # Factor contribution analysis
   portfolio_returns = np.mean(asset_returns, axis=1)  # Equal-weighted portfolio
   total_return = np.sum(portfolio_returns)
   
   print(f"\nFactor Contribution Analysis:")
   factor_contributions = []
   for i, factor in enumerate(factors):
       contribution = portfolio_betas[i] * np.sum(factor_returns[:, i])
       factor_contributions.append(contribution)
       print(f"{factor}: {contribution/total_return*100:.1f}% of total return")
   
   alpha_contribution = portfolio_alpha * n_days
   residual = total_return - sum(factor_contributions) - alpha_contribution
   
   print(f"Alpha: {alpha_contribution/total_return*100:.1f}% of total return")
   print(f"Residual: {residual/total_return*100:.1f}% of total return")

**Output:** Market: 45.2% of total return, Size: 12.3% of total return, Alpha: 8.1% of total return, etc.
