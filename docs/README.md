# Portfolio-lib Documentation

Comprehensive documentation for the portfolio-lib Python package - a quantitative finance library for portfolio analysis and technical indicators.

## 📖 Documentation Structure

### User Guides
- **[Installation Guide](source/installation.rst)** - Setup instructions for all platforms
- **[Getting Started](source/getting_started.rst)** - Beginner-friendly tutorial
- **[Advanced Usage](source/advanced_usage.rst)** - Complex strategies and analysis
- **[Troubleshooting](source/troubleshooting.rst)** - Problem-solving guide

### Technical Reference
- **[API Reference](source/api_reference.rst)** - Complete API documentation
- **[Core Module](source/portfolio_lib.core.rst)** - Technical indicators and utilities
- **[Indicators Module](source/portfolio_lib.indicators.rst)** - High-level indicator functions
- **[Portfolio Module](source/portfolio_lib.portfolio.rst)** - Portfolio analysis and metrics

### Visual Resources
- **[Visual Guide](source/visual_guide.rst)** - Charts, plots, and formula reference
- **[Examples](source/examples.rst)** - 30+ tested code examples
- **[Indicators Comparison](source/indicators_comparison.rst)** - Performance analysis

## 🚀 Quick Start

### View Documentation
- **Online**: [GitHub Pages](#) or [Read the Docs](#)
- **Local**: Open `build/html/index.html` in your browser

### Build Documentation
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r ../portfolio-lib-package/requirements.txt
pip install -e ../portfolio-lib-package/

# Build docs
make clean
make html

# Serve locally
python -m http.server 8000 -d build/html
```

## 📊 Features Covered

### Technical Indicators (17+)
- Moving Averages (SMA, EMA, WMA)
- Momentum Indicators (RSI, MACD, Stochastic)
- Volatility Indicators (ATR, Bollinger Bands)
- Volume Indicators (OBV, MFI)
- Trend Indicators (ADX, Parabolic SAR)

### Portfolio Analytics
- Risk Metrics (VaR, CVaR, Sharpe Ratio)
- Performance Attribution
- Position Sizing Strategies
- Advanced Risk Models
- Drawdown Analysis

### Visual Analytics
- Interactive matplotlib plots
- Risk-return scatter plots
- Correlation heatmaps
- Technical indicator charts
- Performance dashboards

## 🛠 Documentation Tools

### Built With
- **Sphinx** - Documentation generator
- **Read the Docs Theme** - Professional styling
- **MathJax** - Mathematical formulas
- **Matplotlib Plot Directive** - Interactive charts
- **Napoleon** - Google/NumPy docstring support

### Extensions Used
- `sphinx.ext.autodoc` - Automatic API documentation
- `sphinx.ext.napoleon` - Docstring parsing
- `sphinx.ext.mathjax` - Mathematical expressions
- `matplotlib.sphinxext.plot_directive` - Embedded plots
- `sphinx.ext.viewcode` - Source code links
- `sphinx.ext.githubpages` - GitHub Pages support

## 🎯 Target Audience

### Beginner Users
- Clear installation instructions
- Step-by-step tutorials
- Basic portfolio analysis examples
- Troubleshooting common issues

### Advanced Users
- Complex trading strategies
- Multi-factor risk models
- Performance optimization
- Custom indicator development

### Developers
- Complete API reference
- Source code examples
- Extension guidelines
- Integration patterns

## 📁 File Structure

```
docs/
├── source/                    # Source documentation files
│   ├── conf.py               # Sphinx configuration
│   ├── index.rst             # Main documentation page
│   ├── installation.rst      # Installation guide
│   ├── getting_started.rst   # Beginner tutorial
│   ├── advanced_usage.rst    # Advanced examples
│   ├── troubleshooting.rst   # Problem solving
│   ├── api_reference.rst     # API overview
│   ├── portfolio_lib.*.rst   # Module documentation
│   ├── examples.rst          # Code examples
│   ├── visual_guide.rst      # Charts and formulas
│   └── _static/              # Static assets
├── build/                    # Generated documentation
│   └── html/                 # HTML output
├── requirements.txt          # Documentation dependencies
├── Makefile                 # Build commands
└── DEPLOYMENT.md            # Deployment guide
```

## 🔧 Configuration

### Sphinx Settings
- **Theme**: sphinx_rtd_theme
- **Extensions**: autodoc, napoleon, mathjax, plot_directive
- **Math Support**: MathJax 3.0
- **Plot Support**: Matplotlib integration
- **GitHub Integration**: Source links and edit buttons

### Deployment Targets
- **GitHub Pages**: Automatic deployment via GitHub Actions
- **Read the Docs**: Configuration via `.readthedocs.yaml`
- **Local Hosting**: Built-in Python HTTP server

## 📈 Metrics & Analytics

### Documentation Coverage
- ✅ All public APIs documented
- ✅ 30+ working code examples
- ✅ Complete installation guide
- ✅ Comprehensive troubleshooting
- ✅ Mathematical formula reference

### Visual Content
- 📊 7 interactive matplotlib plots
- 📈 Technical indicator comparisons
- 📋 Risk metrics dashboard
- 🔢 Formula reference tables
- 🎨 Custom CSS styling

## 🤝 Contributing

### Documentation Updates
1. Edit `.rst` files in `source/`
2. Add new examples to appropriate sections
3. Update API documentation if needed
4. Test build locally: `make html`
5. Submit pull request

### Adding Examples
1. Create working code in docstrings
2. Include expected outputs
3. Add to `examples.rst` if significant
4. Test all examples before committing

## 📄 License

Documentation is distributed under the same license as portfolio-lib.

## 🆘 Support

- **Issues**: Submit to main repository
- **Build Problems**: Check `DEPLOYMENT.md`
- **Content Questions**: See `troubleshooting.rst`
- **API Questions**: Refer to module documentation

---

*Built with ❤️ using Sphinx and Read the Docs theme*
