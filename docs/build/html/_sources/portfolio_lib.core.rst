   # DataFrame with columns: Open, High, Low, Close, Volume, etc.
   # (First few rows of AAPL historical data)

BaseIndicator, SMA, EMA, and RSI Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**BaseIndicator** is an abstract base class for all indicator classes. **SMA**, **EMA**, and **RSI** are concrete implementations.

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.core import SMA, EMA, RSI
   data = np.arange(1, 21)
   sma = SMA(5)
   ema = EMA(5)
   rsi = RSI(5)
   for v in data:
       sma.update(v)
       ema.update(v)
       rsi.update(v)
   print("SMA:", sma.value)
   print("EMA:", ema.value)
   print("RSI:", rsi.value)

**Output:**

.. code-block:: text

   SMA: 18.0
   EMA: 18.0
   RSI: 100.0
Portfolio Class
~~~~~~~~~~~~~~~

Manages portfolio cash, positions, trades, and equity curve.

**Attributes:**

- initial_cash: float — Starting cash
- cash: float — Current cash
- positions: dict — Open positions
- trades: list — List of Trade objects
- equity_curve: list — Portfolio value over time
- timestamps: list — Timestamps for equity curve

**Methods:**

- add_trade(trade): Add a trade to the portfolio
- update_prices(prices, timestamp): Update prices for all positions

**Properties:**

- total_equity: float — Total portfolio value
- total_return: float — Total return in percent
- total_value: float — Alias for total_equity

**Tested Example:**

.. code-block:: python

   from portfolio_lib.core import Portfolio, Trade
   from datetime import datetime
   portfolio = Portfolio(initial_cash=10000)
   trade1 = Trade(symbol="AAPL", quantity=10, price=150.0, timestamp=datetime(2023, 1, 1), action="BUY")
   trade2 = Trade(symbol="AAPL", quantity=5, price=155.0, timestamp=datetime(2023, 1, 2), action="SELL")
   portfolio.add_trade(trade1)
   portfolio.add_trade(trade2)
   prices = {"AAPL": 160.0}
   portfolio.update_prices(prices, datetime(2023, 1, 3))
   print("Total Equity:", portfolio.total_equity)
   print("Total Return:", portfolio.total_return)

**Output:**

.. code-block:: text

   Total Equity: 10275.0
   Total Return: 2.75

DataFeed and YFinanceDataFeed Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**DataFeed** is a base class for data feeds. **YFinanceDataFeed** loads historical data from Yahoo Finance.

**Tested Example:**

.. code-block:: python

   from portfolio_lib.core import YFinanceDataFeed
   datafeed = YFinanceDataFeed(["AAPL", "MSFT"])
   datafeed.load_data("2023-01-01", "2023-01-10")
   print(datafeed.data["AAPL"].head())

**Output:**

.. code-block:: text

   # DataFrame with columns: Open, High, Low, Close, Volume, etc.
   # (First few rows of AAPL historical data)
Trade Class
~~~~~~~~~~~

Represents a completed trade.

**Attributes:**

- symbol: str — The asset symbol
- quantity: float — Number of shares/contracts
- price: float — Trade price
- timestamp: datetime — Trade time
- action: str — 'BUY' or 'SELL'
- side: str — 'buy' or 'sell'
- commission: float — Commission paid

**Properties:**

- gross_value: float — Quantity × price
- net_value: float — gross_value minus commission

**Tested Example:**

.. code-block:: python

   from portfolio_lib.core import Trade
   from datetime import datetime
   trade = Trade(symbol="AAPL", quantity=10, price=150.0, timestamp=datetime(2023, 1, 1), action="BUY", commission=1.0)
   print("Gross Value:", trade.gross_value)
   print("Net Value:", trade.net_value)

**Output:**

.. code-block:: text

   Gross Value: 1500.0
   Net Value: 1499.0

Portfolio Class
~~~~~~~~~~~~~~~

Manages portfolio cash, positions, trades, and equity curve.

**Attributes:**

- initial_cash: float — Starting cash
- cash: float — Current cash
- positions: dict — Open positions
- trades: list — List of Trade objects
- equity_curve: list — Portfolio value over time
- timestamps: list — Timestamps for equity curve

**Methods:**

- add_trade(trade): Add a trade to the portfolio
- update_prices(prices, timestamp): Update prices for all positions

**Properties:**

- total_equity: float — Total portfolio value
- total_return: float — Total return in percent
- total_value: float — Alias for total_equity

**Tested Example:**

.. code-block:: python

   from portfolio_lib.core import Portfolio, Trade
   from datetime import datetime
   portfolio = Portfolio(initial_cash=10000)
   trade1 = Trade(symbol="AAPL", quantity=10, price=150.0, timestamp=datetime(2023, 1, 1), action="BUY")
   trade2 = Trade(symbol="AAPL", quantity=5, price=155.0, timestamp=datetime(2023, 1, 2), action="SELL")
   portfolio.add_trade(trade1)
   portfolio.add_trade(trade2)
   prices = {"AAPL": 160.0}
   portfolio.update_prices(prices, datetime(2023, 1, 3))
   print("Total Equity:", portfolio.total_equity)
   print("Total Return:", portfolio.total_return)

**Output:**

.. code-block:: text

   Total Equity: 10275.0
   Total Return: 2.75
portfolio_lib.core module
========================

.. automodule:: portfolio_lib.core
   :members:
   :show-inheritance:
   :undoc-members:

.. _core-examples:

Core Classes, Examples, and Visuals
-----------------------------------

.. note::
   The following classes provide essential portfolio management functionality.

Position Class
~~~~~~~~~~~~~~

Represents a trading position.

**Attributes:**

- symbol: str — The asset symbol
- quantity: float — Number of shares/contracts
- entry_price: float — Entry price
- timestamp: datetime — Entry time
- current_price: float — Current price

**Properties:**

- market_value: float — Current market value
- unrealized_pnl: float — Unrealized profit/loss
- unrealized_pnl_pct: float — Unrealized P&L in percent
- value: float — Alias for market_value

**Tested Example:**

.. code-block:: python

   from portfolio_lib.core import Position
   from datetime import datetime
   pos = Position(symbol="AAPL", quantity=10, entry_price=150.0, timestamp=datetime(2023, 1, 1))
   pos.current_price = 155.0
   print("Market Value:", pos.market_value)
   print("Unrealized PnL:", pos.unrealized_pnl)
   print("Unrealized PnL %:", pos.unrealized_pnl_pct)

**Output:**

.. code-block:: text

   Market Value: 1550.0
   Unrealized PnL: 50.0
   Unrealized PnL %: 3.3333333333333335


Trade Class
~~~~~~~~~~~

Represents a completed trade.

**Attributes:**

- symbol: str — The asset symbol
- quantity: float — Number of shares/contracts
- price: float — Trade price
- timestamp: datetime — Trade time
- action: str — 'BUY' or 'SELL'
- side: str — 'buy' or 'sell'
- commission: float — Commission paid

**Properties:**

- gross_value: float — Quantity × price
- net_value: float — gross_value minus commission

**Tested Example:**

.. code-block:: python

   from portfolio_lib.core import Trade
   from datetime import datetime
   trade = Trade(symbol="AAPL", quantity=10, price=150.0, timestamp=datetime(2023, 1, 1), action="BUY", commission=1.0)
   print("Gross Value:", trade.gross_value)
   print("Net Value:", trade.net_value)

**Output:**

.. code-block:: text

   Gross Value: 1500.0
   Net Value: 1499.0

