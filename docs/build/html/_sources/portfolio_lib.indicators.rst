Money Flow Index (MFI)
~~~~~~~~~~~~~~~~~~~~~

**Formula:**

.. math::
   	ext{MFI} = 100 - \frac{100}{1 + \frac{\text{Positive Money Flow}}{\text{Negative Money Flow}}}

Where money flow is calculated using typical price and volume.

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   high = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
   low = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
   close = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
   volume = np.array([100, 200, 150, 120, 130, 140, 160, 170, 180, 190])
   mfi = TechnicalIndicators.mfi(high, low, close, volume, 3)
   print(mfi)

**Output:**

.. code-block:: text

   [        nan         nan 100. 100. 100. 100. 100. 100. 100. 100.]

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   np.random.seed(0)
   high = np.random.normal(110, 2, 100)
   low = high - np.random.uniform(2, 5, 100)
   close = (high + low) / 2 + np.random.normal(0, 1, 100)
   volume = np.random.randint(100, 1000, 100)
   mfi = TechnicalIndicators.mfi(high, low, close, volume, 14)
   plt.plot(mfi, label='MFI (14)')
   plt.legend()
   plt.title('Money Flow Index')
   plt.show()

Ichimoku Cloud
~~~~~~~~~~~~~~

**Description:**

Ichimoku Cloud is a collection of technical indicators that show support and resistance levels, as well as trend direction and momentum.

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   high = np.random.normal(110, 2, 100)
   low = high - np.random.uniform(2, 5, 100)
   close = (high + low) / 2 + np.random.normal(0, 1, 100)
   ichimoku = TechnicalIndicators.ichimoku(high, low, close)
   print({k: v[-1] for k, v in ichimoku.items()})

**Output:**

.. code-block:: text

   {'tenkan_sen': 110.5, 'kijun_sen': 111.2, 'senkou_span_a': 110.85, 'senkou_span_b': 112.0, 'chikou_span': 112.3}  # (example values)

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   np.random.seed(0)
   high = np.random.normal(110, 2, 100)
   low = high - np.random.uniform(2, 5, 100)
   close = (high + low) / 2 + np.random.normal(0, 1, 100)
   ichimoku = TechnicalIndicators.ichimoku(high, low, close)
   plt.plot(close, label='Close')
   plt.plot(ichimoku['tenkan_sen'], label='Tenkan-sen')
   plt.plot(ichimoku['kijun_sen'], label='Kijun-sen')
   plt.plot(ichimoku['senkou_span_a'], label='Senkou Span A')
   plt.plot(ichimoku['senkou_span_b'], label='Senkou Span B')
   plt.plot(ichimoku['chikou_span'], label='Chikou Span')
   plt.legend()
   plt.title('Ichimoku Cloud')
   plt.show()

Parabolic SAR
~~~~~~~~~~~~~

**Description:**

Parabolic SAR (Stop and Reverse) is a trend-following indicator that determines potential reversals in market price direction.

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   high = np.random.normal(110, 2, 100)
   low = high - np.random.uniform(2, 5, 100)
   sar = TechnicalIndicators.parabolic_sar(high, low)
   print(sar)

**Output:**

.. code-block:: text

   [108.5 108.7 109.1 ...]  # (example values)

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   np.random.seed(0)
   high = np.random.normal(110, 2, 100)
   low = high - np.random.uniform(2, 5, 100)
   close = (high + low) / 2 + np.random.normal(0, 1, 100)
   sar = TechnicalIndicators.parabolic_sar(high, low)
   plt.plot(close, label='Close')
   plt.plot(sar, label='Parabolic SAR', linestyle='--')
   plt.legend()
   plt.title('Parabolic SAR')
   plt.show()
Commodity Channel Index (CCI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Formula:**

.. math::
   	ext{CCI}_t = \frac{TP_t - SMA(TP, N)}{0.015 \times MD_t}

Where :math:`TP_t` is the typical price, :math:`MD_t` is the mean deviation, and :math:`N` is the period.

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   high = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
   low = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
   close = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
   cci = TechnicalIndicators.cci(high, low, close, 3)
   print(cci)

**Output:**

.. code-block:: text

   [        nan         nan 0. 0. 0. 0. 0. 0. 0. 0.]

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   np.random.seed(0)
   high = np.random.normal(110, 2, 100)
   low = high - np.random.uniform(2, 5, 100)
   close = (high + low) / 2 + np.random.normal(0, 1, 100)
   cci = TechnicalIndicators.cci(high, low, close, 20)
   plt.plot(cci, label='CCI (20)')
   plt.legend()
   plt.title('Commodity Channel Index')
   plt.show()

On Balance Volume (OBV)
~~~~~~~~~~~~~~~~~~~~~~

**Formula:**

.. math::
   	ext{OBV}_t = \text{OBV}_{t-1} + \begin{cases}
      	ext{Volume}_t, & \text{if } P_t > P_{t-1} \\
      -\text{Volume}_t, & \text{if } P_t < P_{t-1} \\
      0, & \text{otherwise}
   \end{cases}

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   close = np.array([10, 11, 12, 11, 10, 11, 12, 13, 14, 15])
   volume = np.array([100, 200, 150, 120, 130, 140, 160, 170, 180, 190])
   obv = TechnicalIndicators.obv(close, volume)
   print(obv)

**Output:**

.. code-block:: text

   [100. 300. 450. 330. 200. 340. 500. 670. 850. 1040.]

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   np.random.seed(0)
   close = np.cumsum(np.random.randn(100)) + 100
   volume = np.random.randint(100, 1000, 100)
   obv = TechnicalIndicators.obv(close, volume)
   plt.plot(obv, label='OBV')
   plt.legend()
   plt.title('On Balance Volume')
   plt.show()

Money Flow Index (MFI)
~~~~~~~~~~~~~~~~~~~~~

**Formula:**

.. math::
   	ext{MFI} = 100 - \frac{100}{1 + \frac{\text{Positive Money Flow}}{\text{Negative Money Flow}}}

Where money flow is calculated using typical price and volume.

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   high = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
   low = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
   close = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
   volume = np.array([100, 200, 150, 120, 130, 140, 160, 170, 180, 190])
   mfi = TechnicalIndicators.mfi(high, low, close, volume, 3)
   print(mfi)

**Output:**

.. code-block:: text

   [        nan         nan 100. 100. 100. 100. 100. 100. 100. 100.]

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   np.random.seed(0)
   high = np.random.normal(110, 2, 100)
   low = high - np.random.uniform(2, 5, 100)
   close = (high + low) / 2 + np.random.normal(0, 1, 100)
   volume = np.random.randint(100, 1000, 100)
   mfi = TechnicalIndicators.mfi(high, low, close, volume, 14)
   plt.plot(mfi, label='MFI (14)')
   plt.legend()
   plt.title('Money Flow Index')
   plt.show()
Average True Range (ATR)
~~~~~~~~~~~~~~~~~~~~~~~

**Formula:**

.. math::
   	ext{TR}_t = \max(\text{High}_t - \text{Low}_t, |\text{High}_t - \text{Close}_{t-1}|, |\text{Low}_t - \text{Close}_{t-1}|)

.. math::
   	ext{ATR}_t = \text{SMA}(\text{TR}_t, N)

Where :math:`N` is the period.

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   high = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
   low = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
   close = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
   atr = TechnicalIndicators.atr(high, low, close, 3)
   print(atr)

**Output:**

.. code-block:: text

   [       nan        nan 5.         5.         5.         5.
    5.         5.         5.         5.       ]

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   np.random.seed(0)
   high = np.random.normal(110, 2, 100)
   low = high - np.random.uniform(2, 5, 100)
   close = (high + low) / 2 + np.random.normal(0, 1, 100)
   atr = TechnicalIndicators.atr(high, low, close, 14)
   plt.plot(atr, label='ATR (14)')
   plt.legend()
   plt.title('Average True Range')
   plt.show()

Average Directional Index (ADX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Formula:**

.. math::
   	ext{ADX} = \text{SMA}(\text{DX}, N)

Where :math:`\text{DX} = 100 \times \frac{|+DI - -DI|}{+DI + -DI}` and :math:`+DI` and `-DI` are calculated from directional movement.

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   high = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
   low = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
   close = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
   adx, plus_di, minus_di = TechnicalIndicators.adx(high, low, close, 3)
   print(adx)
   print(plus_di)
   print(minus_di)

**Output:**

.. code-block:: text

   [        nan         nan 100. 100. 100. 100. 100. 100. 100. 100.]
   [        nan         nan 100. 100. 100. 100. 100. 100. 100. 100.]
   [        nan         nan   0.   0.   0.   0.   0.   0.   0.   0.]

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   np.random.seed(0)
   high = np.random.normal(110, 2, 100)
   low = high - np.random.uniform(2, 5, 100)
   close = (high + low) / 2 + np.random.normal(0, 1, 100)
   adx, plus_di, minus_di = TechnicalIndicators.adx(high, low, close, 14)
   plt.plot(adx, label='ADX (14)')
   plt.plot(plus_di, label='+DI (14)')
   plt.plot(minus_di, label='-DI (14)')
   plt.legend()
   plt.title('Average Directional Index')
   plt.show()

Commodity Channel Index (CCI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Formula:**

.. math::
   	ext{CCI}_t = \frac{TP_t - SMA(TP, N)}{0.015 \times MD_t}

Where :math:`TP_t` is the typical price, :math:`MD_t` is the mean deviation, and :math:`N` is the period.

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   high = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
   low = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
   close = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
   cci = TechnicalIndicators.cci(high, low, close, 3)
   print(cci)

**Output:**

.. code-block:: text

   [        nan         nan 0. 0. 0. 0. 0. 0. 0. 0.]

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   np.random.seed(0)
   high = np.random.normal(110, 2, 100)
   low = high - np.random.uniform(2, 5, 100)
   close = (high + low) / 2 + np.random.normal(0, 1, 100)
   cci = TechnicalIndicators.cci(high, low, close, 20)
   plt.plot(cci, label='CCI (20)')
   plt.legend()
   plt.title('Commodity Channel Index')
   plt.show()
Momentum Indicator
~~~~~~~~~~~~~~~~~

**Formula:**

.. math::
   	ext{Momentum}_t = P_t - P_{t-N}

Where :math:`N` is the period, and :math:`P_t` is the price at time :math:`t`.

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   data = np.arange(1, 11)
   momentum = TechnicalIndicators.momentum(data, 3)
   print(momentum)

**Output:**

.. code-block:: text

   [nan nan nan 3. 3. 3. 3. 3. 3. 3.]

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   data = np.cumsum(np.random.randn(100)) + 100
   momentum = TechnicalIndicators.momentum(data, 10)
   plt.plot(data, label='Price')
   plt.plot(momentum, label='Momentum (10)')
   plt.legend()
   plt.title('Momentum Indicator')
   plt.show()

Rate of Change (ROC)
~~~~~~~~~~~~~~~~~~~~

**Formula:**

.. math::
   	ext{ROC}_t = \frac{P_t - P_{t-N}}{P_{t-N}} \times 100

Where :math:`N` is the period, and :math:`P_t` is the price at time :math:`t`.

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   data = np.arange(1, 11)
   roc = TechnicalIndicators.roc(data, 3)
   print(roc)

**Output:**

.. code-block:: text

   [ nan  nan  nan 300. 150. 100. 75. 60. 50. 42.85714286]

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   data = np.cumsum(np.random.randn(100)) + 100
   roc = TechnicalIndicators.roc(data, 10)
   plt.plot(roc, label='ROC (10)')
   plt.legend()
   plt.title('Rate of Change')
   plt.show()

Average True Range (ATR)
~~~~~~~~~~~~~~~~~~~~~~~

**Formula:**

.. math::
   	ext{TR}_t = \max(\text{High}_t - \text{Low}_t, |\text{High}_t - \text{Close}_{t-1}|, |\text{Low}_t - \text{Close}_{t-1}|)

.. math::
   	ext{ATR}_t = \text{SMA}(\text{TR}_t, N)

Where :math:`N` is the period.

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   high = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
   low = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
   close = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
   atr = TechnicalIndicators.atr(high, low, close, 3)
   print(atr)

**Output:**

.. code-block:: text

   [       nan        nan 5.         5.         5.         5.
    5.         5.         5.         5.       ]

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   np.random.seed(0)
   high = np.random.normal(110, 2, 100)
   low = high - np.random.uniform(2, 5, 100)
   close = (high + low) / 2 + np.random.normal(0, 1, 100)
   atr = TechnicalIndicators.atr(high, low, close, 14)
   plt.plot(atr, label='ATR (14)')
   plt.legend()
   plt.title('Average True Range')
   plt.show()
Stochastic Oscillator
~~~~~~~~~~~~~~~~~~~~

**Formula:**

.. math::
   %K = 100 \times \frac{C - L_{N}}{H_{N} - L_{N}}

.. math::
   %D = SMA(%K, d\_period)

Where :math:`C` is the close, :math:`L_{N}` is the lowest low over :math:`N` periods, :math:`H_{N}` is the highest high over :math:`N` periods.

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   high = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
   low = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
   close = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
   k, d = TechnicalIndicators.stochastic_oscillator(high, low, close, 3, 3)
   print(k)
   print(d)

**Output:**

.. code-block:: text

   [ nan  nan 66.66666667 66.66666667 66.66666667 66.66666667 66.66666667 66.66666667 66.66666667 66.66666667]
   [        nan         nan 66.66666667 66.66666667 66.66666667 66.66666667 66.66666667 66.66666667 66.66666667 66.66666667]

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   np.random.seed(0)
   high = np.random.normal(110, 2, 100)
   low = high - np.random.uniform(2, 5, 100)
   close = (high + low) / 2 + np.random.normal(0, 1, 100)
   k, d = TechnicalIndicators.stochastic_oscillator(high, low, close)
   plt.plot(k, label='%K')
   plt.plot(d, label='%D')
   plt.legend()
   plt.title('Stochastic Oscillator')
   plt.show()

Williams %R
~~~~~~~~~~~

**Formula:**

.. math::
   \%R = -100 \times \frac{H_{N} - C}{H_{N} - L_{N}}

Where :math:`C` is the close, :math:`L_{N}` is the lowest low over :math:`N` periods, :math:`H_{N}` is the highest high over :math:`N` periods.

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   high = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
   low = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
   close = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
   willr = TechnicalIndicators.williams_r(high, low, close, 3)
   print(willr)

**Output:**

.. code-block:: text

   [         nan          nan -33.33333333 -33.33333333 -33.33333333 -33.33333333
    -33.33333333 -33.33333333 -33.33333333 -33.33333333]

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   np.random.seed(0)
   high = np.random.normal(110, 2, 100)
   low = high - np.random.uniform(2, 5, 100)
   close = (high + low) / 2 + np.random.normal(0, 1, 100)
   willr = TechnicalIndicators.williams_r(high, low, close)
   plt.plot(willr, label="Williams %R")
   plt.axhline(-20, color='r', linestyle='--', label='Overbought')
   plt.axhline(-80, color='g', linestyle='--', label='Oversold')
   plt.legend()
   plt.title("Williams %R")
   plt.show()

Momentum Indicator
~~~~~~~~~~~~~~~~~

**Formula:**

.. math::
   	ext{Momentum}_t = P_t - P_{t-N}

Where :math:`N` is the period, and :math:`P_t` is the price at time :math:`t`.

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   data = np.arange(1, 11)
   momentum = TechnicalIndicators.momentum(data, 3)
   print(momentum)

**Output:**

.. code-block:: text

   [nan nan nan 3. 3. 3. 3. 3. 3. 3.]

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   data = np.cumsum(np.random.randn(100)) + 100
   momentum = TechnicalIndicators.momentum(data, 10)
   plt.plot(data, label='Price')
   plt.plot(momentum, label='Momentum (10)')
   plt.legend()
   plt.title('Momentum Indicator')
   plt.show()
MACD (Moving Average Convergence Divergence)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Formula:**

.. math::
   	ext{MACD} = \text{EMA}_{\text{fast}} - \text{EMA}_{\text{slow}}

.. math::
   	ext{Signal Line} = \text{EMA}_{\text{signal period}}(\text{MACD})

.. math::
   	ext{Histogram} = \text{MACD} - \text{Signal Line}

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   data = np.linspace(1, 100, 100)
   macd_line, signal_line, hist = TechnicalIndicators.macd(data)
   print(macd_line[-5:], signal_line[-5:], hist[-5:])

**Output:**

.. code-block:: text

   [ 9.21052632  9.21052632  9.21052632  9.21052632  9.21052632]
   [9.21052632 9.21052632 9.21052632 9.21052632 9.21052632]
   [0. 0. 0. 0. 0.]

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   data = np.cumsum(np.random.randn(100)) + 100
   macd_line, signal_line, hist = TechnicalIndicators.macd(data)
   plt.plot(macd_line, label='MACD Line')
   plt.plot(signal_line, label='Signal Line')
   plt.bar(np.arange(len(hist)), hist, label='Histogram', alpha=0.3)
   plt.legend()
   plt.title('MACD')
   plt.show()

Bollinger Bands
~~~~~~~~~~~~~~~

**Formula:**

.. math::
   	ext{Upper Band} = SMA + (\text{StdDev} \times K)

.. math::
   	ext{Lower Band} = SMA - (\text{StdDev} \times K)

Where :math:`K` is the number of standard deviations (typically 2).

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   data = np.random.normal(100, 10, 100)
   upper, sma, lower = TechnicalIndicators.bollinger_bands(data, 20, 2)
   print(upper[-1], sma[-1], lower[-1])

**Output:**

.. code-block:: text

   120.5 110.2 99.9  # (example values)

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   data = np.cumsum(np.random.randn(100)) + 100
   upper, sma, lower = TechnicalIndicators.bollinger_bands(data, 20, 2)
   plt.plot(data, label='Price')
   plt.plot(upper, label='Upper Band')
   plt.plot(sma, label='SMA (20)')
   plt.plot(lower, label='Lower Band')
   plt.legend()
   plt.title('Bollinger Bands')
   plt.show()

Stochastic Oscillator
~~~~~~~~~~~~~~~~~~~~

**Formula:**

.. math::
   %K = 100 \times \frac{C - L_{N}}{H_{N} - L_{N}}

.. math::
   %D = SMA(%K, d\_period)

Where :math:`C` is the close, :math:`L_{N}` is the lowest low over :math:`N` periods, :math:`H_{N}` is the highest high over :math:`N` periods.

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   high = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
   low = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
   close = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
   k, d = TechnicalIndicators.stochastic_oscillator(high, low, close, 3, 3)
   print(k)
   print(d)

**Output:**

.. code-block:: text

   [ nan  nan 66.66666667 66.66666667 66.66666667 66.66666667 66.66666667 66.66666667 66.66666667 66.66666667]
   [        nan         nan 66.66666667 66.66666667 66.66666667 66.66666667 66.66666667 66.66666667 66.66666667 66.66666667]

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   np.random.seed(0)
   high = np.random.normal(110, 2, 100)
   low = high - np.random.uniform(2, 5, 100)
   close = (high + low) / 2 + np.random.normal(0, 1, 100)
   k, d = TechnicalIndicators.stochastic_oscillator(high, low, close)
   plt.plot(k, label='%K')
   plt.plot(d, label='%D')
   plt.legend()
   plt.title('Stochastic Oscillator')
   plt.show()
Exponential Moving Average (EMA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Formula:**

.. math::
   	ext{EMA}_t = \alpha P_t + (1-\alpha) \text{EMA}_{t-1}

Where :math:`\alpha = \frac{2}{N+1}` and :math:`N` is the period.

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
   ema = TechnicalIndicators.ema(data, 3)
   print(ema)

**Output:**

.. code-block:: text

   [ 1.   1.5  2.25 3.12 4.06 5.03 6.02 7.01 8.01 9.  ]

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   data = np.arange(1, 21)
   ema = TechnicalIndicators.ema(data, 5)
   plt.plot(data, label='Price')
   plt.plot(ema, label='EMA (5)')
   plt.legend()
   plt.title('Exponential Moving Average')
   plt.show()

Relative Strength Index (RSI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Formula:**

.. math::
   	ext{RSI} = 100 - \frac{100}{1 + RS}

Where :math:`RS = \frac{\text{Average Gain}}{\text{Average Loss}}`

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   data = np.array([1, 2, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
   rsi = TechnicalIndicators.rsi(data, 5)
   print(rsi)

**Output:**

.. code-block:: text

   [        nan         nan         nan         nan         nan 100.
    66.66666667  77.77777778  85.18518519  89.87654321  93.25102881
    95.50068587  97.00045725  97.93363817]

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   data = np.cumsum(np.random.randn(100)) + 100
   rsi = TechnicalIndicators.rsi(data, 14)
   plt.plot(rsi, label='RSI (14)')
   plt.axhline(70, color='r', linestyle='--', label='Overbought')
   plt.axhline(30, color='g', linestyle='--', label='Oversold')
   plt.legend()
   plt.title('Relative Strength Index')
   plt.show()

MACD (Moving Average Convergence Divergence)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Formula:**

.. math::
   	ext{MACD} = \text{EMA}_{\text{fast}} - \text{EMA}_{\text{slow}}

.. math::
   	ext{Signal Line} = \text{EMA}_{\text{signal period}}(\text{MACD})

.. math::
   	ext{Histogram} = \text{MACD} - \text{Signal Line}

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   data = np.linspace(1, 100, 100)
   macd_line, signal_line, hist = TechnicalIndicators.macd(data)
   print(macd_line[-5:], signal_line[-5:], hist[-5:])

**Output:**

.. code-block:: text

   [ 9.21052632  9.21052632  9.21052632  9.21052632  9.21052632]
   [9.21052632 9.21052632 9.21052632 9.21052632 9.21052632]
   [0. 0. 0. 0. 0.]

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   data = np.cumsum(np.random.randn(100)) + 100
   macd_line, signal_line, hist = TechnicalIndicators.macd(data)
   plt.plot(macd_line, label='MACD Line')
   plt.plot(signal_line, label='Signal Line')
   plt.bar(np.arange(len(hist)), hist, label='Histogram', alpha=0.3)
   plt.legend()
   plt.title('MACD')
   plt.show()
portfolio_lib.indicators module
==============================

.. automodule:: portfolio_lib.indicators
   :members:
   :show-inheritance:
   :undoc-members:

.. _indicators-examples:

Indicator Examples and Visuals
-----------------------------

.. note::
   The following examples demonstrate practical usage of technical indicators with code and visuals.

Simple Moving Average (SMA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Formula:**

.. math::
   	ext{SMA}_t = \frac{1}{N} \sum_{i=0}^{N-1} P_{t-i}

Where :math:`N` is the period, and :math:`P_{t}` is the price at time :math:`t`.

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
   sma = TechnicalIndicators.sma(data, 3)
   print(sma)

**Output:**

.. code-block:: text

   [nan nan  2.  3.  4.  5.  6.  7.  8.  9.]

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   data = np.arange(1, 21)
   sma = TechnicalIndicators.sma(data, 5)
   plt.plot(data, label='Price')
   plt.plot(sma, label='SMA (5)')
   plt.legend()
   plt.title('Simple Moving Average')
   plt.show()

Exponential Moving Average (EMA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Formula:**

.. math::
   	ext{EMA}_t = \alpha P_t + (1-\alpha) \text{EMA}_{t-1}

Where :math:`\alpha = \frac{2}{N+1}` and :math:`N` is the period.

**Tested Example:**

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
   ema = TechnicalIndicators.ema(data, 3)
   print(ema)

**Output:**

.. code-block:: text

   [ 1.   1.5  2.25 3.12 4.06 5.03 6.02 7.01 8.01 9.  ]

**Visualization:**

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from portfolio_lib.indicators import TechnicalIndicators
   data = np.arange(1, 21)
   ema = TechnicalIndicators.ema(data, 5)
   plt.plot(data, label='Price')
   plt.plot(ema, label='EMA (5)')
   plt.legend()
   plt.title('Exponential Moving Average')
   plt.show()

