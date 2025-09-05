# Portfolio-lib

**Lightweight Python Library for Quantitative Finance and Algorithmic Trading**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/portfolio-lib/badge/?version=latest)](https://portfolio-lib.readthedocs.io/)

**Authors:** Rahul Ashok, Pritham Devaprasad, Siddarth S, Anish R

## Overview

Portfolio-lib is a comprehensive Python library designed for quantitative finance research, algorithmic trading strategy development, and portfolio analysis. The library provides a complete suite of technical indicators, risk management tools, and backtesting capabilities with minimal dependencies and optimal performance.

## Repository Structure

```
Portfolio-lib/
├── portfolio-lib-package/     # Core Python package
│   ├── portfolio_lib/         # Main library modules
│   │   ├── __init__.py       # Package initialization
│   │   ├── core.py           # Technical indicators and utilities
│   │   ├── indicators.py     # High-level indicator functions
│   │   └── portfolio.py      # Portfolio analysis and metrics
│   ├── setup.py              # Package installation configuration
│   ├── requirements.txt      # Runtime dependencies
│   ├── README.md             # Package documentation
│   └── test.py               # Test suite
├── docs/                      # Comprehensive documentation
│   ├── source/               # Documentation source files
│   ├── build/                # Generated documentation
│   ├── requirements.txt      # Documentation dependencies
│   └── README.md             # Documentation overview
├── .github/workflows/        # GitHub Actions for CI/CD
├── .readthedocs.yaml         # Read the Docs configuration
└── README.md                 # This file
```

## Package Features

### Technical Analysis
- **17+ Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic Oscillator, ADX, ATR, CCI, OBV, MFI, Momentum, ROC, Williams %R, Parabolic SAR, Ichimoku Kinko Hyo
- **Advanced Indicators**: 60+ specialized indicators including TRIX, Schiff Trend Cycle, Rainbow Oscillator, Fractal Indicator, Hilbert Transform
- **Mathematical Foundations**: Comprehensive formula implementations with numerical stability

### Portfolio Analytics
- **Risk Metrics**: Value at Risk (VaR), Conditional VaR, Maximum Drawdown, Ulcer Index, Burke Ratio, Sterling Ratio
- **Performance Metrics**: Sharpe Ratio, Sortino Ratio, Calmar Ratio, Treynor Ratio, Information Ratio, Alpha, Beta
- **Attribution Analysis**: Brinson attribution, sector attribution, factor decomposition
- **Position Sizing**: Kelly Criterion, fixed fractional, volatility-based sizing, risk parity

### Data Structures
- **Portfolio Management**: Position tracking, trade execution, cash management
- **Performance Tracking**: Equity curves, return calculations, benchmark comparison
- **Risk Management**: Real-time risk monitoring, exposure limits, correlation analysis

## Installation

### Standard Installation
```bash
pip install portfolio-lib
```

### Development Installation
```bash
git clone https://github.com/NeuralNinja110/Portfolio-lib.git
cd Portfolio-lib/portfolio-lib-package
pip install -e .
```

### Requirements
- Python 3.8 or higher
- NumPy 1.21.0 or higher
- Pandas 1.5.0 or higher
- SciPy 1.9.0 or higher
- Matplotlib 3.5.0 or higher

## Quick Start

### Basic Technical Analysis
```python
import numpy as np
import pandas as pd
from portfolio_lib.indicators import TechnicalIndicators

# Generate sample price data
prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])

# Calculate technical indicators
sma_20 = TechnicalIndicators.sma(prices, period=5)
rsi_14 = TechnicalIndicators.rsi(prices, period=5)
macd_result = TechnicalIndicators.macd(prices)

print(f"Simple Moving Average: {sma_20.iloc[-1]:.2f}")
print(f"RSI: {rsi_14:.2f}")
print(f"MACD: {macd_result['macd'].iloc[-1]:.2f}")
```

### Portfolio Risk Analysis
```python
import numpy as np
from portfolio_lib.portfolio import RiskMetrics

# Sample return data
returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for one year

# Initialize risk metrics
risk = RiskMetrics(returns)

# Calculate risk measures
var_95 = risk.var_95()
cvar_95 = risk.cvar_95()
max_dd = risk.maximum_drawdown()

print(f"Value at Risk (95%): {var_95:.4f}")
print(f"Conditional VaR (95%): {cvar_95:.4f}")
print(f"Maximum Drawdown: {max_dd:.4f}")
```

### Advanced Portfolio Analytics
```python
from portfolio_lib.portfolio import AdvancedPortfolioAnalytics
import pandas as pd

# Sample portfolio and benchmark data
portfolio_returns = pd.Series(np.random.normal(0.0008, 0.015, 252))
benchmark_returns = pd.Series(np.random.normal(0.0005, 0.012, 252))

# Initialize analytics
analytics = AdvancedPortfolioAnalytics(portfolio_returns)

# Calculate advanced metrics
alpha = analytics.calculate_alpha(benchmark_returns)
beta = analytics.calculate_beta(benchmark_returns)
sharpe = analytics.calculate_sharpe_ratio()
information_ratio = analytics.calculate_information_ratio(benchmark_returns)

print(f"Alpha: {alpha:.4f}")
print(f"Beta: {beta:.4f}")
print(f"Sharpe Ratio: {sharpe:.4f}")
print(f"Information Ratio: {information_ratio:.4f}")
```

## Documentation

The repository includes comprehensive documentation covering all aspects of the library:

### User Documentation
- **Installation Guide**: Platform-specific installation instructions and troubleshooting
- **Getting Started Tutorial**: Step-by-step introduction for beginners
- **Advanced Usage Examples**: Complex strategies and analytical techniques
- **API Reference**: Complete function and class documentation

### Technical Documentation
- **Mathematical Formulas**: LaTeX-rendered equations for all indicators and metrics
- **Visual Examples**: Interactive matplotlib plots and charts
- **Performance Analysis**: Comparative studies and benchmarking
- **Code Examples**: 30+ tested examples with expected outputs

### Documentation Access
- **Online**: [Read the Docs](https://portfolio-lib.readthedocs.io/) and [GitHub Pages](https://neuralninja110.github.io/Portfolio-lib/)
- **Local**: Build documentation locally using Sphinx

```bash
cd docs/
pip install -r requirements.txt
make html
# Open build/html/index.html in browser
```

## Core Modules

### portfolio_lib.core
Contains fundamental classes and utilities for technical analysis:
- **Technical Indicators**: Base classes for all indicators (SMA, EMA, RSI, etc.)
- **Price Utilities**: OHLC data handling and validation
- **Portfolio Classes**: Position, Trade, and Portfolio management
- **Performance Metrics**: Calculation engines for returns and risk

### portfolio_lib.indicators
High-level interface for technical indicator calculations:
- **TechnicalIndicators**: Static methods for all indicator calculations
- **Trend Indicators**: Moving averages, ADX, Parabolic SAR
- **Oscillators**: RSI, Stochastic, Williams %R, CCI
- **Volume Indicators**: OBV, MFI, Volume Oscillator
- **Volatility Indicators**: ATR, Bollinger Bands

### portfolio_lib.portfolio
Advanced portfolio analysis and risk management:
- **AdvancedPortfolioAnalytics**: Comprehensive risk and performance metrics
- **RiskMetrics**: VaR, CVaR, drawdown analysis
- **PositionSizing**: Kelly Criterion, risk parity, volatility targeting
- **PerformanceAttribution**: Brinson attribution and factor analysis

## Mathematical Foundation

The library implements mathematically rigorous formulations for all indicators and metrics:

### Technical Indicators
- Moving averages with proper initialization handling
- Oscillators with normalized scaling and boundary conditions
- Momentum indicators with statistical significance testing
- Volume-price relationships with correlation analysis

### Risk Metrics
- Parametric and non-parametric VaR calculations
- Expected shortfall with tail risk analysis
- Drawdown statistics with recovery period estimation
- Higher-order moment analysis (skewness, kurtosis)

### Performance Attribution
- Brinson-Hood-Beebower attribution methodology
- Factor model decomposition using regression analysis
- Risk-adjusted return calculations with benchmark comparison
- Statistical significance testing for performance differences

## Testing and Quality Assurance

### Test Coverage
- Comprehensive unit tests for all functions
- Integration tests for complete workflows
- Performance benchmarks against established libraries
- Numerical accuracy validation using known test cases

### Code Quality
- PEP 8 compliance for coding standards
- Type hints for improved code clarity
- Comprehensive docstrings with examples
- Automated testing via GitHub Actions

## Performance Characteristics

### Computational Efficiency
- Vectorized operations using NumPy for optimal performance
- Memory-efficient algorithms for large datasets
- Lazy evaluation for complex indicator chains
- Parallel processing support for portfolio analysis

### Scalability
- Linear time complexity for most indicators
- Constant memory usage for streaming calculations
- Batch processing capabilities for historical analysis
- Real-time calculation support for live trading

## Contributing

### Development Environment
```bash
git clone https://github.com/NeuralNinja110/Portfolio-lib.git
cd Portfolio-lib/portfolio-lib-package
pip install -e .[dev]
```

### Code Standards
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add unit tests for new functionality
- Update documentation for API changes

### Submission Process
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit pull request with detailed description

## License

This project is licensed under the MIT License. See the LICENSE file for complete terms and conditions.

## Citation

If you use Portfolio-lib in academic research, please cite:

```bibtex
@software{portfolio_lib_2025,
  title={Portfolio-lib: Lightweight Python Library for Quantitative Finance},
  author={Ashok, Rahul and Devaprasad, Pritham and S, Siddarth and R, Anish},
  year={2025},
  url={https://github.com/NeuralNinja110/Portfolio-lib},
  version={1.0.1}
}
```

## Contact and Support

- **Primary Contact**: Rahul Ashok (abcrahul111@gmail.com)
- **Repository**: [GitHub](https://github.com/NeuralNinja110/Portfolio-lib)
- **Documentation**: [Read the Docs](https://portfolio-lib.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/NeuralNinja110/Portfolio-lib/issues)

## Acknowledgments

Portfolio-lib was developed as part of quantitative finance research initiatives. The authors acknowledge the contributions of the open-source community and the mathematical foundations established by prior research in computational finance.

---

**Portfolio-lib: Professional quantitative finance tools for Python developers and researchers.**
