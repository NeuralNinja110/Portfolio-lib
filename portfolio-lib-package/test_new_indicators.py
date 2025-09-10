"""
Comprehensive test suite for new technical indicators
Tests all 20 newly added indicators for correctness and edge cases
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from portfolio_lib.indicators import TechnicalIndicators

class TestNewIndicators:
    """Test class for all 20 new technical indicators"""
    
    @classmethod
    def setup_class(cls):
        """Setup test data"""
        np.random.seed(42)  # For reproducible tests
        cls.length = 100
        
        # Generate realistic OHLCV data
        cls.close = np.cumsum(np.random.randn(cls.length) * 0.02) + 100
        cls.high = cls.close + np.random.rand(cls.length) * 2
        cls.low = cls.close - np.random.rand(cls.length) * 2
        cls.open = cls.close + np.random.randn(cls.length) * 0.5
        cls.volume = np.random.randint(1000, 10000, cls.length)
        
        # Ensure high >= low
        for i in range(cls.length):
            if cls.high[i] < cls.low[i]:
                cls.high[i], cls.low[i] = cls.low[i], cls.high[i]
            if cls.close[i] > cls.high[i]:
                cls.high[i] = cls.close[i]
            if cls.close[i] < cls.low[i]:
                cls.low[i] = cls.close[i]
    
    def test_klinger_oscillator(self):
        """Test Klinger Oscillator"""
        kvo, signal = TechnicalIndicators.klinger_oscillator(
            self.high, self.low, self.close, self.volume
        )
        
        assert len(kvo) == self.length
        assert len(signal) == self.length
        assert not np.all(np.isnan(kvo))
        assert not np.all(np.isnan(signal))
        
        # Test with different parameters
        kvo2, signal2 = TechnicalIndicators.klinger_oscillator(
            self.high, self.low, self.close, self.volume, 20, 35, 10
        )
        assert not np.array_equal(kvo, kvo2)
        
        print("✓ Klinger Oscillator test passed")
    
    def test_price_channel(self):
        """Test Price Channel"""
        upper, middle, lower = TechnicalIndicators.price_channel(self.high, self.low, 20)
        
        assert len(upper) == self.length
        assert len(middle) == self.length
        assert len(lower) == self.length
        
        # Test that upper >= middle >= lower where defined
        valid_mask = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
        assert np.all(upper[valid_mask] >= middle[valid_mask])
        assert np.all(middle[valid_mask] >= lower[valid_mask])
        
        print("✓ Price Channel test passed")
    
    def test_donchian_channel(self):
        """Test Donchian Channel"""
        upper, middle, lower = TechnicalIndicators.donchian_channel(self.high, self.low, 15)
        
        assert len(upper) == self.length
        assert len(middle) == self.length
        assert len(lower) == self.length
        assert not np.all(np.isnan(upper))
        
        print("✓ Donchian Channel test passed")
    
    def test_elder_force_index(self):
        """Test Elder's Force Index"""
        efi = TechnicalIndicators.elder_force_index(self.close, self.volume, 13)
        
        assert len(efi) == self.length
        assert not np.all(np.isnan(efi))
        
        # Test period = 1 (raw force)
        efi_raw = TechnicalIndicators.elder_force_index(self.close, self.volume, 1)
        assert not np.array_equal(efi, efi_raw)
        
        print("✓ Elder's Force Index test passed")
    
    def test_ease_of_movement(self):
        """Test Ease of Movement"""
        eom = TechnicalIndicators.ease_of_movement(self.high, self.low, self.volume, 14)
        
        assert len(eom) == self.length
        assert not np.all(np.isnan(eom))
        
        print("✓ Ease of Movement test passed")
    
    def test_mass_index(self):
        """Test Mass Index"""
        mi = TechnicalIndicators.mass_index(self.high, self.low, 25, 9)
        
        assert len(mi) == self.length
        assert not np.all(np.isnan(mi))
        
        # Mass index should be positive
        valid_values = mi[~np.isnan(mi)]
        assert np.all(valid_values > 0)
        
        print("✓ Mass Index test passed")
    
    def test_negative_volume_index(self):
        """Test Negative Volume Index"""
        nvi = TechnicalIndicators.negative_volume_index(self.close, self.volume)
        
        assert len(nvi) == self.length
        assert not np.all(np.isnan(nvi))
        assert nvi[0] == 1000  # Starting value
        
        print("✓ Negative Volume Index test passed")
    
    def test_positive_volume_index(self):
        """Test Positive Volume Index"""
        pvi = TechnicalIndicators.positive_volume_index(self.close, self.volume)
        
        assert len(pvi) == self.length
        assert not np.all(np.isnan(pvi))
        assert pvi[0] == 1000  # Starting value
        
        print("✓ Positive Volume Index test passed")
    
    def test_price_volume_trend(self):
        """Test Price Volume Trend"""
        pvt = TechnicalIndicators.price_volume_trend(self.close, self.volume)
        
        assert len(pvt) == self.length
        assert not np.all(np.isnan(pvt))
        assert pvt[0] == self.volume[0]  # Starting value
        
        print("✓ Price Volume Trend test passed")
    
    def test_volume_accumulation(self):
        """Test Volume Accumulation"""
        va = TechnicalIndicators.volume_accumulation(self.close, self.volume)
        
        assert len(va) == self.length
        assert not np.all(np.isnan(va))
        assert va[0] == self.volume[0]  # Starting value
        
        print("✓ Volume Accumulation test passed")
    
    def test_williams_ad(self):
        """Test Williams Accumulation/Distribution"""
        wad = TechnicalIndicators.williams_ad(self.high, self.low, self.close, self.volume)
        
        assert len(wad) == self.length
        assert not np.all(np.isnan(wad))
        assert wad[0] == 0  # Starting value
        
        print("✓ Williams A/D test passed")
    
    def test_coppock_curve(self):
        """Test Coppock Curve"""
        cc = TechnicalIndicators.coppock_curve(self.close, 14, 11, 10)
        
        assert len(cc) == self.length
        assert not np.all(np.isnan(cc))
        
        print("✓ Coppock Curve test passed")
    
    def test_know_sure_thing(self):
        """Test Know Sure Thing"""
        kst, signal = TechnicalIndicators.know_sure_thing(self.close)
        
        assert len(kst) == self.length
        assert len(signal) == self.length
        assert not np.all(np.isnan(kst))
        
        print("✓ Know Sure Thing test passed")
    
    def test_price_oscillator(self):
        """Test Price Oscillator"""
        po = TechnicalIndicators.price_oscillator(self.close, 12, 26)
        
        assert len(po) == self.length
        assert not np.all(np.isnan(po))
        
        # Test bounds (should be percentage)
        valid_values = po[~np.isnan(po)]
        assert np.all(np.abs(valid_values) < 100)  # Reasonable bounds
        
        print("✓ Price Oscillator test passed")
    
    def test_ultimate_oscillator(self):
        """Test Ultimate Oscillator"""
        uo = TechnicalIndicators.ultimate_oscillator(self.high, self.low, self.close, 7, 14, 28)
        
        assert len(uo) == self.length
        assert not np.all(np.isnan(uo))
        
        # Ultimate Oscillator should be between 0 and 100
        valid_values = uo[~np.isnan(uo)]
        assert np.all(valid_values >= 0)
        assert np.all(valid_values <= 100)
        
        print("✓ Ultimate Oscillator test passed")
    
    def test_triple_ema(self):
        """Test Triple EMA"""
        tema = TechnicalIndicators.triple_ema(self.close, 14)
        
        assert len(tema) == self.length
        assert not np.all(np.isnan(tema))
        
        print("✓ Triple EMA test passed")
    
    def test_relative_vigor_index(self):
        """Test Relative Vigor Index"""
        rvi, signal = TechnicalIndicators.relative_vigor_index(
            self.open, self.high, self.low, self.close, 10
        )
        
        assert len(rvi) == self.length
        assert len(signal) == self.length
        assert not np.all(np.isnan(rvi))
        
        print("✓ Relative Vigor Index test passed")
    
    def test_schaff_trend_cycle(self):
        """Test Schaff Trend Cycle"""
        stc = TechnicalIndicators.schaff_trend_cycle(self.close, 23, 50, 10)
        
        assert len(stc) == self.length
        assert not np.all(np.isnan(stc))
        
        # STC should be between 0 and 100
        valid_values = stc[~np.isnan(stc)]
        if len(valid_values) > 0:
            assert np.all(valid_values >= 0)
            assert np.all(valid_values <= 100)
        
        print("✓ Schaff Trend Cycle test passed")
    
    def test_stochastic_rsi(self):
        """Test Stochastic RSI"""
        k, d = TechnicalIndicators.stochastic_rsi(self.close, 14, 14, 3, 3)
        
        assert len(k) == self.length
        assert len(d) == self.length
        assert not np.all(np.isnan(k))
        
        # Stochastic RSI should be between 0 and 100
        valid_k = k[~np.isnan(k)]
        if len(valid_k) > 0:
            assert np.all(valid_k >= 0)
            assert np.all(valid_k <= 100)
        
        print("✓ Stochastic RSI test passed")
    
    def test_vortex_indicator(self):
        """Test Vortex Indicator"""
        vi_plus, vi_minus = TechnicalIndicators.vortex_indicator(
            self.high, self.low, self.close, 14
        )
        
        assert len(vi_plus) == self.length
        assert len(vi_minus) == self.length
        assert not np.all(np.isnan(vi_plus))
        assert not np.all(np.isnan(vi_minus))
        
        # VI values should be positive
        valid_plus = vi_plus[~np.isnan(vi_plus)]
        valid_minus = vi_minus[~np.isnan(vi_minus)]
        assert np.all(valid_plus >= 0)
        assert np.all(valid_minus >= 0)
        
        print("✓ Vortex Indicator test passed")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        
        # Test with small arrays
        small_data = np.array([1, 2, 3, 4, 5])
        small_volume = np.array([100, 200, 150, 180, 160])
        
        try:
            result = TechnicalIndicators.price_oscillator(small_data, 3, 4)
            assert len(result) == 5
        except Exception as e:
            pytest.fail(f"Failed on small data: {e}")
        
        # Test with constant values
        constant_data = np.ones(50) * 100
        constant_volume = np.ones(50) * 1000
        
        try:
            result = TechnicalIndicators.ease_of_movement(constant_data, constant_data, constant_volume)
            assert len(result) == 50
        except Exception as e:
            pytest.fail(f"Failed on constant data: {e}")
        
        # Test with zeros in volume
        zero_volume = np.zeros(20)
        try:
            result = TechnicalIndicators.elder_force_index(self.close[:20], zero_volume)
            assert len(result) == 20
        except Exception as e:
            pytest.fail(f"Failed with zero volume: {e}")
        
        print("✓ Edge cases test passed")
    
    def run_performance_test(self):
        """Test performance with larger datasets"""
        import time
        
        large_length = 10000
        large_close = np.random.randn(large_length).cumsum() + 100
        large_high = large_close + np.random.rand(large_length)
        large_low = large_close - np.random.rand(large_length)
        large_volume = np.random.randint(1000, 10000, large_length)
        
        start_time = time.time()
        
        # Test a few indicators for performance
        TechnicalIndicators.klinger_oscillator(large_high, large_low, large_close, large_volume)
        TechnicalIndicators.ultimate_oscillator(large_high, large_low, large_close)
        TechnicalIndicators.vortex_indicator(large_high, large_low, large_close)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"✓ Performance test passed - {execution_time:.3f}s for {large_length} data points")
        
        # Should complete in reasonable time (< 5 seconds for 10k points)
        assert execution_time < 5.0


def run_all_tests():
    """Run all tests"""
    print("Running comprehensive tests for 20 new technical indicators...")
    print("=" * 60)
    
    test_instance = TestNewIndicators()
    test_instance.setup_class()
    
    # Run all indicator tests
    test_methods = [
        'test_klinger_oscillator',
        'test_price_channel', 
        'test_donchian_channel',
        'test_elder_force_index',
        'test_ease_of_movement',
        'test_mass_index',
        'test_negative_volume_index',
        'test_positive_volume_index',
        'test_price_volume_trend',
        'test_volume_accumulation',
        'test_williams_ad',
        'test_coppock_curve',
        'test_know_sure_thing',
        'test_price_oscillator',
        'test_ultimate_oscillator',
        'test_triple_ema',
        'test_relative_vigor_index',
        'test_schaff_trend_cycle',
        'test_stochastic_rsi',
        'test_vortex_indicator',
        'test_edge_cases'
    ]
    
    passed_tests = 0
    total_tests = len(test_methods)
    
    for test_method in test_methods:
        try:
            method = getattr(test_instance, test_method)
            method()
            passed_tests += 1
        except Exception as e:
            print(f"✗ {test_method} failed: {e}")
    
    # Performance test
    try:
        test_instance.run_performance_test()
        passed_tests += 1
        total_tests += 1
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
    
    print("=" * 60)
    print(f"Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! All 20 indicators are working correctly.")
    else:
        print(f"⚠️  {total_tests - passed_tests} test(s) failed.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        exit(1)
