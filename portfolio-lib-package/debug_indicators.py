#!/usr/bin/env python3
"""
Debug script to test new indicators one by one
"""
import numpy as np
import pandas as pd
from portfolio_lib.indicators import TechnicalIndicators

def create_test_data():
    """Create sample price data for testing"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    # Generate realistic price data
    base_price = 100
    returns = np.random.normal(0.001, 0.02, 100)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    high = np.array(prices) * (1 + np.random.uniform(0, 0.02, 100))
    low = np.array(prices) * (1 - np.random.uniform(0, 0.02, 100))
    close = np.array(prices)
    volume = np.random.randint(1000, 10000, 100)
    
    return pd.DataFrame({
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

def test_indicator(indicator_name, data, *args, **kwargs):
    """Test a single indicator"""
    try:
        indicators = TechnicalIndicators()
        method = getattr(indicators, indicator_name)
        result = method(data, *args, **kwargs)
        print(f"✓ {indicator_name}: Success - Shape: {np.array(result).shape}")
        return True
    except Exception as e:
        print(f"✗ {indicator_name}: Failed - {str(e)}")
        return False

def main():
    """Main test function"""
    print("Creating test data...")
    data = create_test_data()
    
    # Test existing indicators first
    print("\n=== Testing Existing Indicators ===")
    existing_indicators = [
        ('sma', data['close'], 20),
        ('ema', data['close'], 20),
        ('rsi', data['close'], 14),
        ('macd', data['close']),
        ('bollinger_bands', data['close'], 20, 2),
    ]
    
    for indicator_name, *args in existing_indicators:
        test_indicator(indicator_name, *args)
    
    # Test new indicators
    print("\n=== Testing New Indicators ===")
    new_indicators = [
        ('vwap', data),
        ('supertrend', data),
        ('keltner_channels', data),
        ('donchian_channels', data),
        ('aroon', data),
        ('chande_momentum_oscillator', data['close']),
        ('detrended_price_oscillator', data['close']),
        ('ease_of_movement', data['high'], data['low'], data['volume']),
        ('force_index', data),
        ('mass_index', data['high'], data['low']),
        ('negative_volume_index', data['close'], data['volume']),
        ('positive_volume_index', data['close'], data['volume']),
        ('price_volume_trend', data['close'], data['volume']),
        ('trix', data['close']),
        ('ultimate_oscillator', data['high'], data['low'], data['close']),
        ('vortex_indicator', data['high'], data['low'], data['close']),
        ('williams_accumulation_distribution', data),
        ('chaikin_oscillator', data),
        ('elder_ray_index', data),
        ('klinger_oscillator', data['high'], data['low'], data['close'], data['volume']),
    ]
    
    success_count = 0
    for indicator_name, *args in new_indicators:
        if test_indicator(indicator_name, *args):
            success_count += 1
    
    print(f"\n=== Results ===")
    print(f"Successfully tested: {success_count}/{len(new_indicators)} new indicators")

if __name__ == "__main__":
    main()
