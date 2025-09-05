
"""
Test script for portfolio-lib 1.0.1
Tests all inbuilt functions and technical indicators using the actual API.
"""

import subprocess
import sys
import numpy as np
import pandas as pd
from portfolio_lib.indicators import TechnicalIndicators
from portfolio_lib.portfolio import AdvancedPortfolioAnalytics, PositionSizing, PerformanceAttribution, RiskMetrics

# Ensure the latest version is installed
subprocess.check_call([sys.executable, "-m", "pip", "install", "portfolio-lib==1.0.1"])

def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=200)
    prices = np.cumsum(np.random.randn(200)) + 100
    high = prices + np.random.rand(200)
    low = prices - np.random.rand(200)
    close = prices
    volume = np.random.randint(100, 1000, size=200)
    return pd.DataFrame({"Date": dates, "High": high, "Low": low, "Close": close, "Volume": volume}).set_index("Date")

def test_indicators(df):
    print("Testing technical indicators...")
    ti = TechnicalIndicators
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    volume = df["Volume"].values
    print("SMA:", ti.sma(close, 20)[-5:])
    print("EMA:", ti.ema(close, 20)[-5:])
    print("RSI:", ti.rsi(close, 14)[-5:])
    macd_line, signal_line, hist = ti.macd(close)
    print("MACD:", macd_line[-5:], signal_line[-5:], hist[-5:])
    upper, sma, lower = ti.bollinger_bands(close)
    print("Bollinger Bands:", upper[-1], sma[-1], lower[-1])
    k, d = ti.stochastic_oscillator(high, low, close)
    print("Stochastic Oscillator:", k[-5:], d[-5:])
    print("Williams %R:", ti.williams_r(high, low, close)[-5:])
    print("Momentum:", ti.momentum(close)[-5:])
    print("ROC:", ti.roc(close)[-5:])
    print("ATR:", ti.atr(high, low, close)[-5:])
    adx, plus_di, minus_di = ti.adx(high, low, close)
    print("ADX:", adx[-5:], plus_di[-5:], minus_di[-5:])
    print("CCI:", ti.cci(high, low, close)[-5:])
    print("OBV:", ti.obv(close, volume)[-5:])
    print("MFI:", ti.mfi(high, low, close, volume)[-5:])
    ichimoku = ti.ichimoku(high, low, close)
    print("Ichimoku Cloud:", {k: v[-1] for k, v in ichimoku.items()})
    print("Parabolic SAR:", ti.parabolic_sar(high, low)[-5:])
    print("All technical indicators tested!\n")

def test_portfolio_analytics(df):
    print("Testing portfolio analytics...")
    returns = df["Close"].pct_change().dropna().values
    benchmark = np.random.normal(0, 0.01, size=len(returns))
    analytics = AdvancedPortfolioAnalytics(returns, benchmark)
    equity_curve = np.cumprod(1 + returns)
    print("VaR:", analytics.calculate_var())
    print("CVaR:", analytics.calculate_cvar())
    print("Max Drawdown:", analytics.calculate_maximum_drawdown(equity_curve))
    print("Ulcer Index:", analytics.calculate_ulcer_index(equity_curve))
    print("Burke Ratio:", analytics.calculate_burke_ratio(equity_curve))
    print("Sterling Ratio:", analytics.calculate_sterling_ratio(equity_curve))
    print("Tracking Error:", analytics.calculate_tracking_error())
    print("Information Ratio:", analytics.calculate_information_ratio())
    print("Beta:", analytics.calculate_beta())
    print("Alpha:", analytics.calculate_alpha())
    print("Treynor Ratio:", analytics.calculate_treynor_ratio())
    print("Modigliani Ratio:", analytics.calculate_modigliani_ratio())
    print("Omega Ratio:", analytics.calculate_omega_ratio())
    print("Kappa Ratio:", analytics.calculate_kappa_ratio())
    print("Gain Pain Ratio:", analytics.calculate_gain_pain_ratio())
    print("Comprehensive Risk Metrics:", analytics.calculate_comprehensive_risk_metrics(equity_curve))
    print("All portfolio analytics tested!\n")

def test_position_sizing():
    print("Testing position sizing...")
    print("Kelly Criterion:", PositionSizing.kelly_criterion(0.55, 100, 50))
    print("Fixed Fractional:", PositionSizing.fixed_fractional(10000, 0.02, 0.01))
    print("Volatility Position Sizing:", PositionSizing.volatility_position_sizing(10000, 0.1, 0.02))
    cov = np.array([[0.01, 0.002], [0.002, 0.015]])
    print("Risk Parity Weights:", PositionSizing.risk_parity_weights(cov))
    print("All position sizing methods tested!\n")

def test_performance_attribution():
    print("Testing performance attribution...")
    portfolio_returns = np.random.normal(0, 0.01, 100)
    benchmark_returns = np.random.normal(0, 0.01, 100)
    pa = PerformanceAttribution(portfolio_returns, benchmark_returns, np.array([0.5, 0.5]), np.array([0.5, 0.5]))
    print("Brinson Attribution:", pa.brinson_attribution(np.array([0.5, 0.5])))
    print("Sector Attribution:", pa.calculate_sector_attribution({"AAPL": "Tech", "GOOG": "Tech"}))
    print("All performance attribution methods tested!\n")

def main():
    df = generate_sample_data()
    test_indicators(df)
    test_portfolio_analytics(df)
    test_position_sizing()
    test_performance_attribution()
    print("All tests completed successfully!")

if __name__ == "__main__":
    main()
