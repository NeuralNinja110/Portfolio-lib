Troubleshooting Guide
===================

This guide helps you solve common issues when using portfolio-lib.

Common Installation Issues
-------------------------

ImportError: No module named 'portfolio_lib'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: After installation, you get an import error.

**Possible Causes & Solutions**:

1. **Wrong Python environment**:
   
   .. code-block:: bash
   
      # Check which Python you're using
      which python
      which pip
      
      # Make sure you're in the right environment
      conda env list  # or
      pip list | grep portfolio-lib

2. **Installation in wrong location**:
   
   .. code-block:: bash
   
      # Reinstall in current environment
      pip uninstall portfolio-lib
      pip install portfolio-lib

3. **Development installation issues**:
   
   .. code-block:: bash
   
      # For development install
      cd /path/to/portfolio-lib
      pip install -e .

ModuleNotFoundError: No module named 'numpy'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Missing dependencies.

**Solution**:

.. code-block:: bash

   # Install all dependencies
   pip install numpy pandas matplotlib scipy
   
   # Or reinstall portfolio-lib (should install dependencies)
   pip install --force-reinstall portfolio-lib

Version Conflicts
~~~~~~~~~~~~~~~~~

**Problem**: Dependency version conflicts.

**Solution**:

.. code-block:: bash

   # Create fresh environment
   conda create -n portfolio_clean python=3.9
   conda activate portfolio_clean
   pip install portfolio-lib
   
   # Or update conflicting packages
   pip install --upgrade numpy pandas matplotlib

Common Runtime Errors
---------------------

ValueError: array length mismatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Input arrays have different lengths.

**Example Error**:

.. code-block:: python

   prices = [100, 101, 102]
   volumes = [1000, 1100]  # One element short!
   result = TechnicalIndicators.vwap(prices, volumes)
   # ValueError: arrays must have same length

**Solution**:

.. code-block:: python

   # Always check array lengths
   assert len(prices) == len(volumes), f"Length mismatch: prices={len(prices)}, volumes={len(volumes)}"
   
   # Or align arrays
   min_length = min(len(prices), len(volumes))
   prices = prices[:min_length]
   volumes = volumes[:min_length]

IndexError: index out of range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Accessing array elements that don't exist.

**Example Error**:

.. code-block:: python

   prices = [100, 101, 102]
   sma = TechnicalIndicators.sma(prices, 20)  # Period longer than data
   print(sma[19])  # IndexError!

**Solution**:

.. code-block:: python

   # Check if enough data is available
   period = 20
   if len(prices) >= period:
       sma = TechnicalIndicators.sma(prices, period)
       # Only use valid values (non-NaN)
       valid_sma = sma[~np.isnan(sma)]
   else:
       print(f"Need at least {period} data points, got {len(prices)}")

NaN Values in Results
~~~~~~~~~~~~~~~~~~~~

**Problem**: Getting NaN (Not a Number) values in calculations.

**Example**:

.. code-block:: python

   prices = [100, 101, 102, 103, 104]
   rsi = TechnicalIndicators.rsi(prices, 14)
   print(rsi)  # [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 50.0, ...]

**Explanation**: RSI needs at least 14 periods to calculate. Earlier values are NaN.

**Solution**:

.. code-block:: python

   # Filter out NaN values
   rsi = TechnicalIndicators.rsi(prices, 14)
   valid_rsi = rsi[~np.isnan(rsi)]
   
   # Or check for sufficient data
   if len(prices) >= 14:
       print(f"Latest RSI: {rsi[-1]:.2f}")
   else:
       print("Need more data for RSI calculation")

Memory and Performance Issues
----------------------------

Slow Performance with Large Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Calculations are slow with large amounts of data.

**Solution**:

.. code-block:: python

   import numpy as np
   
   # Use NumPy arrays instead of Python lists
   prices_list = list(range(100000))  # Slow
   prices_array = np.array(prices_list)  # Fast
   
   # Vectorize calculations when possible
   # Instead of:
   results = []
   for i in range(len(prices)):
       results.append(some_calculation(prices[i]))
   
   # Use:
   results = np.vectorize(some_calculation)(prices)

Memory Usage with Large Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Running out of memory with large datasets.

**Solution**:

.. code-block:: python

   # Process data in chunks
   def process_large_dataset(prices, chunk_size=10000):
       results = []
       for i in range(0, len(prices), chunk_size):
           chunk = prices[i:i+chunk_size]
           chunk_result = TechnicalIndicators.sma(chunk, 20)
           results.extend(chunk_result)
       return np.array(results)
   
   # Use appropriate data types
   prices = np.array(prices, dtype=np.float32)  # Use float32 instead of float64 if precision allows

Data Quality Issues
------------------

Missing or Invalid Data
~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Data contains missing values, zeros, or invalid prices.

**Solution**:

.. code-block:: python

   def clean_price_data(prices):
       """Clean and validate price data"""
       prices = np.array(prices, dtype=float)
       
       # Remove or interpolate missing values
       if np.any(np.isnan(prices)):
           print(f"Warning: {np.sum(np.isnan(prices))} missing values found")
           # Forward fill
           mask = np.isnan(prices)
           prices[mask] = np.interp(np.flatnonzero(mask), 
                                   np.flatnonzero(~mask), 
                                   prices[~mask])
       
       # Check for non-positive prices
       if np.any(prices <= 0):
           print("Warning: Non-positive prices found")
           prices = prices[prices > 0]
       
       # Check for extreme outliers (> 10x daily moves)
       returns = np.diff(prices) / prices[:-1]
       outliers = np.abs(returns) > 0.10  # 10% daily moves
       if np.any(outliers):
           print(f"Warning: {np.sum(outliers)} potential outliers found")
       
       return prices
   
   # Usage
   raw_prices = [100, 101, 0, 102, np.nan, 105]  # Problematic data
   clean_prices = clean_price_data(raw_prices)

Date/Time Issues
~~~~~~~~~~~~~~~

**Problem**: Incorrect date handling or time zone issues.

**Solution**:

.. code-block:: python

   import pandas as pd
   
   # Always use proper datetime objects
   dates = pd.date_range('2023-01-01', periods=100, freq='D')
   
   # Handle time zones explicitly
   dates_utc = dates.tz_localize('UTC')
   dates_ny = dates_utc.tz_convert('America/New_York')
   
   # Align data with trading calendar
   from pandas.tseries.offsets import BDay
   business_days = pd.date_range('2023-01-01', periods=100, freq=BDay())

Calculation Errors
-----------------

Incorrect Technical Indicator Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Indicator values don't match expected results.

**Debugging Steps**:

.. code-block:: python

   # 1. Verify input data
   print(f"Price data: {prices[:10]}...")  # First 10 values
   print(f"Data type: {type(prices[0])}")
   print(f"Data length: {len(prices)}")
   
   # 2. Check parameters
   period = 14
   print(f"Calculation period: {period}")
   
   # 3. Manual calculation for verification
   def manual_sma(prices, period):
       if len(prices) < period:
           return np.nan
       return np.mean(prices[-period:])
   
   # Compare with library function
   lib_sma = TechnicalIndicators.sma(prices, period)[-1]
   manual_sma_val = manual_sma(prices, period)
   
   print(f"Library SMA: {lib_sma}")
   print(f"Manual SMA: {manual_sma_val}")
   print(f"Difference: {abs(lib_sma - manual_sma_val)}")

Unexpected Portfolio Values
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Portfolio calculations give unexpected results.

**Debugging**:

.. code-block:: python

   # Track each component
   portfolio = Portfolio(initial_cash=100000)
   
   print(f"Initial cash: ${portfolio.cash:,.2f}")
   
   # Add position and track changes
   position = Position("STOCK1", 100, 50.0, pd.Timestamp.now())
   portfolio.add_position(position)
   
   print(f"After adding position:")
   print(f"  Cash: ${portfolio.cash:,.2f}")
   print(f"  Position value: ${position.quantity * position.entry_price:,.2f}")
   print(f"  Total value: ${portfolio.get_total_value():,.2f}")

API and Integration Issues
-------------------------

Wrong Function Signatures
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Function called with incorrect parameters.

**Solution**:

.. code-block:: python

   # Check function documentation
   help(TechnicalIndicators.bollinger_bands)
   
   # Use correct parameters
   upper, middle, lower = TechnicalIndicators.bollinger_bands(
       prices=prices,
       period=20,
       std_dev=2.0
   )

Type Errors
~~~~~~~~~~

**Problem**: Passing wrong data types to functions.

**Solution**:

.. code-block:: python

   # Ensure correct types
   prices = np.array(prices, dtype=float)  # Convert to float array
   period = int(period)                    # Ensure integer period
   
   # Validate inputs
   assert isinstance(prices, (list, np.ndarray)), "Prices must be array-like"
   assert len(prices) > 0, "Prices array cannot be empty"
   assert period > 0, "Period must be positive"

Environment and System Issues
----------------------------

Platform-Specific Issues
~~~~~~~~~~~~~~~~~~~~~~~~

**Windows Issues**:

.. code-block:: bash

   # Install Visual C++ Build Tools if needed
   # Use pre-compiled wheels
   pip install --only-binary=all portfolio-lib

**macOS Issues**:

.. code-block:: bash

   # Install Xcode command line tools
   xcode-select --install
   
   # Use Homebrew Python
   brew install python
   pip3 install portfolio-lib

**Linux Issues**:

.. code-block:: bash

   # Install development packages
   sudo apt-get install python3-dev build-essential
   
   # Or use conda
   conda install portfolio-lib

Jupyter Notebook Issues
~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Issues running in Jupyter notebooks.

**Solutions**:

.. code-block:: bash

   # Restart kernel if imports fail
   # Kernel -> Restart & Clear Output
   
   # Check kernel Python version
   import sys
   print(sys.executable)
   print(sys.version)
   
   # Install in notebook environment
   pip install portfolio-libPerformance Optimization
-----------------------

Slow Calculation Tips
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 1. Use NumPy arrays
   prices = np.array(prices, dtype=np.float64)
   
   # 2. Avoid loops when possible
   # Instead of:
   smas = []
   for i in range(20, len(prices)):
       sma = np.mean(prices[i-20:i])
       smas.append(sma)
   
   # Use vectorized operations:
   smas = TechnicalIndicators.sma(prices, 20)
   
   # 3. Cache calculations
   from functools import lru_cache
   
   @lru_cache(maxsize=128)
   def cached_sma(prices_tuple, period):
       return TechnicalIndicators.sma(np.array(prices_tuple), period)

Memory Optimization
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use appropriate data types
   prices = np.array(prices, dtype=np.float32)  # vs np.float64
   
   # Process in chunks for large datasets
   def process_chunks(data, chunk_size=1000):
       for i in range(0, len(data), chunk_size):
           yield data[i:i+chunk_size]
   
   # Clean up large objects
   del large_array
   import gc
   gc.collect()

Getting Help
-----------

Debug Information Collection
~~~~~~~~~~~~~~~~~~~~~~~~~~

When reporting issues, include:

.. code-block:: python

   import portfolio_lib
   import numpy as np
   import pandas as pd
   import sys
   
   print("Debug Information:")
   print(f"portfolio-lib version: {portfolio_lib.__version__}")
   print(f"NumPy version: {np.__version__}")
   print(f"Pandas version: {pd.__version__}")
   print(f"Python version: {sys.version}")
   print(f"Platform: {sys.platform}")

Minimal Reproducible Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a minimal example that reproduces the issue:

.. code-block:: python

   import numpy as np
   from portfolio_lib.indicators import TechnicalIndicators
   
   # Minimal data that causes the issue
   prices = [100, 101, 99, 102, 98]  # Your problematic data
   
   try:
       result = TechnicalIndicators.sma(prices, 3)
       print(f"Result: {result}")
   except Exception as e:
       print(f"Error: {e}")
       print(f"Error type: {type(e).__name__}")

Community Resources
~~~~~~~~~~~~~~~~~

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Complete API reference
- **Stack Overflow**: Tag questions with `portfolio-lib`
- **Community Forums**: Join discussions with other users

Frequently Asked Questions
-------------------------

**Q: Why are my indicator values different from other libraries?**

A: Different libraries may use slightly different calculation methods. Check:
- How NaN values are handled
- Whether adjustments are made for weekends/holidays
- Rounding precision differences

**Q: Can I use real-time data feeds?**

A: Yes, but ensure your data feed provides clean, properly formatted data. Implement error handling for data quality issues.

**Q: How do I handle corporate actions (splits, dividends)?**

A: Adjust historical prices for splits and consider dividend reinvestment for accurate return calculations.

**Q: What's the best way to backtest strategies?**

A: Use out-of-sample data, account for transaction costs, avoid look-ahead bias, and test across different market conditions.

Error Code Reference
-------------------

Common error patterns and their meanings:

- **E001**: Array length mismatch - Input arrays have different sizes
- **E002**: Insufficient data - Not enough data points for calculation
- **E003**: Invalid parameter - Parameter out of valid range
- **E004**: Type error - Wrong data type provided
- **E005**: Memory error - Dataset too large for available memory

Preventive Measures
------------------

.. code-block:: python

   # Always validate inputs
   def validate_price_data(prices, min_length=1):
       if not isinstance(prices, (list, np.ndarray)):
           raise TypeError("Prices must be array-like")
       
       prices = np.array(prices, dtype=float)
       
       if len(prices) < min_length:
           raise ValueError(f"Need at least {min_length} data points")
       
       if np.any(np.isnan(prices)):
           raise ValueError("Prices contain NaN values")
       
       if np.any(prices <= 0):
           raise ValueError("Prices must be positive")
       
       return prices
   
   # Use in your code
   try:
       clean_prices = validate_price_data(raw_prices, min_length=20)
       sma = TechnicalIndicators.sma(clean_prices, 20)
   except (TypeError, ValueError) as e:
       print(f"Data validation failed: {e}")
