Installation Guide
==================

This guide will help you install and set up portfolio-lib in your environment.

Requirements
------------

**Python Version**
- Python 3.8 or higher
- Recommended: Python 3.9+

**Dependencies**
- numpy >= 1.20.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- scipy >= 1.7.0

Installation Methods
-------------------

Method 1: Using pip (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install portfolio-lib

Method 2: From Source
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/NeuralNinja110/Portfolio-lib.git
   cd portfolio-lib
   pip install -e .

Method 3: Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For contributors and developers:

.. code-block:: bash

   git clone https://github.com/NeuralNinja110/Portfolio-lib.git
   cd portfolio-lib
   pip install -e ".[dev]"

This installs additional development dependencies like pytest, black, and flake8.

Virtual Environment Setup
-------------------------

It's recommended to use a virtual environment:

**Using venv (Python 3.3+)**

.. code-block:: bash

   python -m venv portfolio_env
   source portfolio_env/bin/activate  # On Windows: portfolio_env\Scripts\activate
   pip install portfolio-lib

**Using conda**

.. code-block:: bash

   conda create -n portfolio_env python=3.9
   conda activate portfolio_env
   pip install portfolio-lib

Verification
-----------

Test your installation:

.. code-block:: python

   import portfolio_lib
   from portfolio_lib.indicators import TechnicalIndicators
   from portfolio_lib.core import Portfolio
   from portfolio_lib.portfolio import RiskMetrics
   
   print(f"portfolio-lib version: {portfolio_lib.__version__}")
   
   # Quick test
   import numpy as np
   prices = np.array([100, 102, 101, 103, 105, 104, 106])
   sma = TechnicalIndicators.sma(prices, 5)
   print(f"SMA calculation successful: {sma[-1]:.2f}")

Expected output:

.. code-block:: text

   portfolio-lib version: 1.0.0
   SMA calculation successful: 103.20

Common Installation Issues
-------------------------

**Issue: Import Error**

.. code-block:: text

   ImportError: No module named 'portfolio_lib'

**Solution:**
- Verify installation: ``pip list | grep portfolio-lib``
- Check virtual environment is activated
- Reinstall: ``pip uninstall portfolio-lib && pip install portfolio-lib``

**Issue: Dependency Conflicts**

.. code-block:: text

   ERROR: package-name has requirement numpy>=1.20.0, but you have numpy 1.19.0

**Solution:**
- Update dependencies: ``pip install --upgrade numpy pandas matplotlib scipy``
- Use fresh virtual environment

**Issue: Windows Installation Problems**

**Solution:**
- Install Visual C++ Build Tools
- Use pre-compiled wheels: ``pip install --only-binary=all portfolio-lib``

Optional Dependencies
--------------------

For enhanced functionality, install optional packages:

**Plotting and Visualization**
   
.. code-block:: bash

   pip install plotly>=5.0.0 seaborn>=0.11.0

**Data Sources**

.. code-block:: bash

   pip install yfinance>=0.1.70 alpha-vantage>=2.3.0

**Advanced Analytics**

.. code-block:: bash

   pip install scikit-learn>=1.0.0 statsmodels>=0.13.0

**Performance Optimization**

.. code-block:: bash

   pip install numba>=0.56.0 cython>=0.29.0

Complete Installation
--------------------

For a full-featured installation:

.. code-block:: bash

   pip install portfolio-lib[complete]

This includes all optional dependencies for maximum functionality.

Docker Installation
------------------

Use the official Docker image:

.. code-block:: bash

   docker pull portfolio-lib/portfolio-lib:latest
   docker run -it portfolio-lib/portfolio-lib:latest python

Or build from source:

.. code-block:: bash

   git clone https://github.com/NeuralNinja110/Portfolio-lib.git
   cd portfolio-lib
   docker build -t my-portfolio-lib .
   docker run -it my-portfolio-lib python

Jupyter Integration
------------------

For Jupyter notebook users:

.. code-block:: bash

   pip install portfolio-lib jupyter matplotlib
   jupyter notebook

Or use JupyterLab:

.. code-block:: bash

   pip install portfolio-lib jupyterlab matplotlib
   jupyter lab

IDE Setup
---------

**VS Code**

1. Install Python extension
2. Select correct Python interpreter (Ctrl+Shift+P → "Python: Select Interpreter")
3. Install IntelliSense: the extension will auto-detect portfolio-lib

**PyCharm**

1. Open project settings
2. Configure Python interpreter to your virtual environment
3. Portfolio-lib will be available for auto-completion

**Spyder**

1. Install in same environment as portfolio-lib
2. Restart Spyder after installation

Update and Uninstall
-------------------

**Update to latest version:**

.. code-block:: bash

   pip install --upgrade portfolio-lib

**Check current version:**

.. code-block:: bash

   pip show portfolio-lib

**Uninstall:**

.. code-block:: bash

   pip uninstall portfolio-lib

Next Steps
----------

After successful installation:

1. Read the :doc:`getting_started` guide
2. Explore :doc:`examples` for practical use cases
3. Check :doc:`api_reference` for detailed function documentation

Need Help?
----------

- **GitHub Issues**: Report bugs or request features
- **Documentation**: Complete API reference and examples
- **Community**: Join our Discord/Slack for discussions

System-Specific Notes
--------------------

**macOS**

.. code-block:: bash

   # If using Homebrew Python
   brew install python
   pip3 install portfolio-lib

**Ubuntu/Debian**

.. code-block:: bash

   sudo apt update
   sudo apt install python3-pip python3-venv
   pip3 install portfolio-lib

**CentOS/RHEL**

.. code-block:: bash

   sudo yum install python3-pip
   pip3 install portfolio-lib

**Windows**

- Install Python from python.org
- Use Command Prompt or PowerShell
- Consider using Anaconda distribution for easier dependency management
