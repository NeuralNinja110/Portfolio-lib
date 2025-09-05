"""
Setup configuration for Portfolio-lib
"""

from setuptools import setup, find_packages
import os

# Read README file for long description
def read_readme():
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    return "Lightweight Python backtesting library for algorithmic trading"

# Read requirements from requirements.txt
def read_requirements():
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return ["pandas>=1.5.0", "numpy>=1.21.0", "yfinance>=0.2.0", "scipy>=1.9.0", "matplotlib>=3.5.0"]

setup(
    name="portfolio-lib",
    version="1.0.1",
    
    # Author information (as specified in project requirements)
    author="Rahul Ashok, Pritham Devaprasad, Siddarth S, and Anish R",
    author_email="contact@portfolio-lib.com",
    
    # Package description
    description="Lightweight Python backtesting library for algorithmic trading strategies",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    
    # URLs and metadata
    url="https://github.com/portfolio-lib/portfolio-lib",
    project_urls={
        "Bug Reports": "https://github.com/portfolio-lib/portfolio-lib/issues",
        "Source": "https://github.com/portfolio-lib/portfolio-lib",
        "Documentation": "https://portfolio-lib.readthedocs.io",
    },
    
    # Package configuration
    packages=find_packages(),
    package_data={
        "portfolio_lib": ["*.py"],
    },
    include_package_data=True,
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Package classification
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    
    # Keywords for searchability
    keywords=[
        "backtesting", "trading", "finance", "algorithmic-trading", 
        "quantitative-finance", "portfolio", "technical-analysis",
        "stocks", "crypto", "forex", "investment", "strategy",
        "lightweight", "performance", "analytics", "risk-management"
    ],
    
    # Entry points (if needed for CLI tools)
    entry_points={
        "console_scripts": [
            # Add CLI commands here if needed in future versions
        ],
    },
    
    # Additional metadata
    license="MIT",
    zip_safe=False,
    
    # Optional extras
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
    },
)