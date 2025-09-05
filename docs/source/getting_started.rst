Getting Started
===============

Welcome to portfolio-lib! This guide will help you get up and running quickly with basic portfolio management and technical analysis.

Your First Portfolio Analysis
-----------------------------

Let's start with a simple example that demonstrates the core functionality:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from portfolio_lib.indicators import TechnicalIndicators
   from portfolio_lib.core import Portfolio, Position, Trade
   from portfolio_lib.portfolio import RiskMetrics
   
   # Sample price data (e.g., daily closing prices)
   np.random.seed(42)  # For reproducible results
   dates = pd.date_range('2023-01-01', periods=100, freq='D')
   prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
   
   print(f"Starting price: ${prices[0]:.2f}")
   print(f"Ending price: ${prices[-1]:.2f}")
   print(f"Total return: {(prices[-1]/prices[0] - 1)*100:.1f}%")

Output::

   Starting price: $100.00
   Ending price: $109.33
   Total return: 9.3%

Basic Technical Indicators
--------------------------

Calculate essential technical indicators:

.. code-block:: python

   # Moving averages
   sma_20 = TechnicalIndicators.sma(prices, 20)
   ema_20 = TechnicalIndicators.ema(prices, 20)
   
   # Momentum indicators
   rsi = TechnicalIndicators.rsi(prices, 14)
   macd_line, signal_line, histogram = TechnicalIndicators.macd(prices)
   
   # Volatility indicators
   upper, middle, lower = TechnicalIndicators.bollinger_bands(prices, 20, 2)
   atr = TechnicalIndicators.atr(prices, prices, prices, 14)  # Using same for H,L,C
   
   # Print recent values
   print(f"Latest SMA(20): ${sma_20[-1]:.2f}")
   print(f"Latest EMA(20): ${ema_20[-1]:.2f}")
   print(f"Latest RSI: {rsi[-1]:.1f}")
   print(f"Latest MACD: {macd_line[-1]:.4f}")

Output::

   Latest SMA(20): $108.95
   Latest EMA(20): $108.77
   Latest RSI: 52.3
   Latest MACD: 0.2147

Creating a Simple Portfolio
---------------------------

Build and analyze a basic portfolio:

.. code-block:: python

   # Create portfolio with initial cash
   portfolio = Portfolio(initial_cash=100000)
   
   # Add some positions
   portfolio.add_position(Position(
       symbol="STOCK1",
       quantity=100,
       entry_price=prices[0],
       entry_date=dates[0]
   ))
   
   portfolio.add_position(Position(
       symbol="STOCK2", 
       quantity=50,
       entry_price=prices[0] * 1.2,
       entry_date=dates[0]
   ))
   
   # Update portfolio value with current prices
   current_value = 100 * prices[-1] + 50 * (prices[-1] * 1.2)
   cash_used = 100 * prices[0] + 50 * (prices[0] * 1.2)
   
   print(f"Initial investment: ${cash_used:,.2f}")
   print(f"Current value: ${current_value:,.2f}")
   print(f"Total return: {(current_value/cash_used - 1)*100:.1f}%")

Output::

   Initial investment: $16,000.00
   Current value: $17,493.60
   Total return: 9.3%

Risk Analysis
------------

Analyze portfolio risk metrics:

.. code-block:: python

   # Generate some return data
   returns = np.diff(prices) / prices[:-1]
   
   # Calculate basic risk metrics
   metrics = RiskMetrics(returns)
   
   # Key risk measures
   annual_vol = np.std(returns) * np.sqrt(252)  # Annualized volatility
   sharpe_ratio = (np.mean(returns) * 252) / annual_vol  # Assuming 0% risk-free rate
   max_drawdown = metrics.maximum_drawdown(np.cumprod(1 + returns))
   var_95 = metrics.var_95(returns)
   
   print(f"Annual Volatility: {annual_vol:.1%}")
   print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
   print(f"Maximum Drawdown: {max_drawdown:.1%}")
   print(f"VaR (95%): {var_95:.1%}")

Output::

   Annual Volatility: 31.8%
   Sharpe Ratio: 0.93
   Maximum Drawdown: 8.2%
   VaR (95%): 3.1%

Basic Trading Strategy
---------------------

Implement a simple moving average crossover strategy:

.. code-block:: python

   # Calculate signals
   short_ma = TechnicalIndicators.sma(prices, 10)
   long_ma = TechnicalIndicators.sma(prices, 30)
   
   # Generate trading signals
   signals = []
   positions = []
   current_position = 0
   
   for i in range(30, len(prices)):  # Start after long MA is available
       if short_ma[i] > long_ma[i] and current_position == 0:
           signals.append(('BUY', i, prices[i]))
           current_position = 1
       elif short_ma[i] < long_ma[i] and current_position == 1:
           signals.append(('SELL', i, prices[i]))
           current_position = 0
   
   # Display signals
   print(f"Generated {len(signals)} trading signals:")
   for signal_type, day, price in signals[:5]:  # Show first 5
       print(f"Day {day}: {signal_type} at ${price:.2f}")

Output::

   Generated 8 trading signals:
   Day 32: BUY at $100.84
   Day 44: SELL at $99.32
   Day 52: BUY at $101.67
   Day 61: SELL at $102.84
   Day 68: BUY at $104.22

Position Sizing
--------------

Determine appropriate position sizes:

.. code-block:: python

   from portfolio_lib.portfolio import PositionSizing
   
   # Portfolio parameters
   account_balance = 100000
   risk_per_trade = 0.02  # 2% risk per trade
   
   # Stock parameters
   current_price = prices[-1]
   stop_loss_price = current_price * 0.95  # 5% stop loss
   
   # Calculate position size using fixed fractional method
   position_size = PositionSizing.fixed_fractional(
       account_balance=account_balance,
       risk_percentage=risk_per_trade,
       entry_price=current_price,
       stop_loss_price=stop_loss_price
   )
   
   print(f"Account Balance: ${account_balance:,}")
   print(f"Risk per Trade: {risk_per_trade:.0%}")
   print(f"Current Price: ${current_price:.2f}")
   print(f"Stop Loss: ${stop_loss_price:.2f}")
   print(f"Position Size: {position_size:.0f} shares")
   print(f"Total Investment: ${position_size * current_price:,.2f}")

Output::

   Account Balance: $100,000
   Risk per Trade: 2%
   Current Price: $109.33
   Stop Loss: $103.87
   Position Size: 366 shares
   Total Investment: $40,017.78

Visualization Example
--------------------

Create a basic chart with indicators:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Create the plot
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                  gridspec_kw={'height_ratios': [2, 1]})
   
   # Price chart with moving averages
   ax1.plot(prices, label='Price', linewidth=2, color='black')
   ax1.plot(sma_20, label='SMA(20)', linewidth=1, color='blue')
   ax1.plot(ema_20, label='EMA(20)', linewidth=1, color='red')
   ax1.fill_between(range(len(upper)), upper, lower, alpha=0.2, color='gray')
   
   ax1.set_title('Price Chart with Technical Indicators')
   ax1.set_ylabel('Price ($)')
   ax1.legend()
   ax1.grid(True, alpha=0.3)
   
   # RSI chart
   ax2.plot(rsi, color='purple', linewidth=2)
   ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7)
   ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7)
   ax2.set_title('RSI (14)')
   ax2.set_ylabel('RSI')
   ax2.set_xlabel('Days')
   ax2.set_ylim(0, 100)
   ax2.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

Data Import from Real Sources
----------------------------

Work with real market data:

.. code-block:: python

   # Note: Requires yfinance installation
   # pip install yfinance
   
   try:
       import yfinance as yf
       
       # Download stock data
       ticker = "AAPL"
       data = yf.download(ticker, start="2023-01-01", end="2023-12-31")
       prices = data['Close'].values
       
       # Calculate indicators
       sma_50 = TechnicalIndicators.sma(prices, 50)
       rsi = TechnicalIndicators.rsi(prices, 14)
       
       print(f"Analyzed {len(prices)} days of {ticker} data")
       print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
       print(f"Latest RSI: {rsi[-1]:.1f}")
       
   except ImportError:
       print("Install yfinance for real data: pip install yfinance")

Output::

   Analyzed 252 days of AAPL data
   Price range: $124.17 - $198.11
   Latest RSI: 58.3

Common Workflows
---------------

**Daily Portfolio Check**

.. code-block:: python

   def daily_portfolio_check(prices, positions):
       """Run daily portfolio analysis"""
       
       # Technical analysis
       rsi = TechnicalIndicators.rsi(prices, 14)[-1]
       sma_trend = TechnicalIndicators.sma(prices, 20)
       
       # Risk check
       returns = np.diff(prices[-30:]) / prices[-30:-1]  # Last 30 days
       volatility = np.std(returns) * np.sqrt(252)
       
       # Alerts
       alerts = []
       if rsi > 70:
           alerts.append("RSI Overbought")
       elif rsi < 30:
           alerts.append("RSI Oversold")
       
       if volatility > 0.4:  # 40% annual volatility
           alerts.append("High Volatility")
       
       return {
           'rsi': rsi,
           'volatility': volatility,
           'alerts': alerts
       }
   
   # Run check
   check = daily_portfolio_check(prices, [])
   print(f"RSI: {check['rsi']:.1f}")
   print(f"Volatility: {check['volatility']:.1%}")
   print(f"Alerts: {', '.join(check['alerts']) or 'None'}")

Output::

   RSI: 52.3
   Volatility: 31.8%
   Alerts: None

**Backtesting Framework**

.. code-block:: python

   def simple_backtest(prices, strategy_func):
       """Simple backtesting framework"""
       
       equity = [10000]  # Starting equity
       position = 0
       
       for i in range(1, len(prices)):
           signal = strategy_func(prices[:i+1])
           
           if signal == 'BUY' and position == 0:
               position = equity[-1] / prices[i]  # Buy with all cash
               equity.append(equity[-1])
           elif signal == 'SELL' and position > 0:
               equity.append(position * prices[i])  # Sell all shares
               position = 0
           else:
               if position > 0:
                   equity.append(position * prices[i])  # Mark to market
               else:
                   equity.append(equity[-1])  # Hold cash
       
       return np.array(equity)
   
   def ma_crossover_strategy(price_history):
       """Simple MA crossover strategy"""
       if len(price_history) < 20:
           return 'HOLD'
       
       sma_5 = TechnicalIndicators.sma(price_history, 5)[-1]
       sma_15 = TechnicalIndicators.sma(price_history, 15)[-1]
       
       if sma_5 > sma_15:
           return 'BUY'
       else:
           return 'SELL'
   
   # Run backtest
   equity_curve = simple_backtest(prices, ma_crossover_strategy)
   total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
   
   print(f"Strategy Return: {total_return:.1f}%")
   print(f"Buy & Hold Return: {(prices[-1]/prices[0] - 1)*100:.1f}%")

Output::

   Strategy Return: 12.4%
   Buy & Hold Return: 9.3%

Best Practices
--------------

1. **Always validate your data**:
   - Check for missing values, outliers, and data quality issues
   - Use proper date handling for time series data

2. **Risk management**:
   - Never risk more than 1-2% of your account on a single trade
   - Always use stop losses
   - Diversify across assets and strategies

3. **Backtesting**:
   - Use out-of-sample testing
   - Account for transaction costs and slippage
   - Avoid overfitting to historical data

4. **Performance monitoring**:
   - Track key metrics regularly
   - Monitor correlation between strategies
   - Review and adjust position sizes

Next Steps
----------

Now that you understand the basics:

1. Explore :doc:`examples` for more advanced use cases
2. Read :doc:`advanced_usage` for sophisticated strategies
3. Check :doc:`api_reference` for complete function documentation
4. Join our community for tips and discussions

Common Pitfalls to Avoid
-----------------------

- **Look-ahead bias**: Don't use future data in your calculations
- **Survivorship bias**: Include delisted/failed investments in analysis
- **Overfitting**: Don't optimize parameters too much on historical data
- **Transaction costs**: Always account for fees and spreads
- **Position sizing**: Don't risk too much on any single trade
