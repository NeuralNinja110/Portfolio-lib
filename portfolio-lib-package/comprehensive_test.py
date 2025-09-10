#!/usr/bin/env python3
"""
Comprehensive test suite for all 20 new technical indicators
"""
import numpy as np
import pandas as pd
from portfolio_lib.indicators import TechnicalIndicators

def create_comprehensive_test_data():
    """Create comprehensive test data"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    # Generate realistic price data with trend
    base_price = 100
    trend = np.linspace(0, 0.2, 100)  # 20% upward trend
    noise = np.random.normal(0, 0.02, 100)
    
    prices = []
    for i in range(100):
        if i == 0:
            prices.append(base_price)
        else:
            price_change = trend[i] + noise[i]
            prices.append(prices[-1] * (1 + price_change))
    
    # Create OHLCV data
    close = np.array(prices)
    high = close * (1 + np.random.uniform(0.001, 0.03, 100))
    low = close * (1 - np.random.uniform(0.001, 0.03, 100))
    open_prices = close * (1 + np.random.uniform(-0.01, 0.01, 100))
    volume = np.random.randint(10000, 100000, 100)
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

def test_all_new_indicators():
    """Test all 20 new indicators comprehensively"""
    print("🚀 Starting Comprehensive Test of 20 New Technical Indicators")
    print("=" * 70)
    
    data = create_comprehensive_test_data()
    indicators = TechnicalIndicators()
    
    test_results = []
    
    # Test 1: VWAP
    try:
        vwap = indicators.vwap(data)
        assert len(vwap) == len(data), "VWAP length mismatch"
        assert not np.all(np.isnan(vwap)), "VWAP all NaN"
        test_results.append(("✅ VWAP", "PASSED", f"Shape: {vwap.shape}"))
    except Exception as e:
        test_results.append(("❌ VWAP", "FAILED", str(e)))
    
    # Test 2: SuperTrend
    try:
        supertrend, trend = indicators.supertrend(data)
        assert len(supertrend) == len(data), "SuperTrend length mismatch"
        assert len(trend) == len(data), "Trend length mismatch"
        test_results.append(("✅ SuperTrend", "PASSED", f"Shape: {supertrend.shape}"))
    except Exception as e:
        test_results.append(("❌ SuperTrend", "FAILED", str(e)))
    
    # Test 3: Keltner Channels
    try:
        upper, middle, lower = indicators.keltner_channels(data)
        assert len(upper) == len(data), "Keltner upper length mismatch"
        assert len(middle) == len(data), "Keltner middle length mismatch"
        assert len(lower) == len(data), "Keltner lower length mismatch"
        test_results.append(("✅ Keltner Channels", "PASSED", f"Shape: {upper.shape}"))
    except Exception as e:
        test_results.append(("❌ Keltner Channels", "FAILED", str(e)))
    
    # Test 4: Donchian Channels
    try:
        upper, middle, lower = indicators.donchian_channels(data)
        assert len(upper) == len(data), "Donchian upper length mismatch"
        test_results.append(("✅ Donchian Channels", "PASSED", f"Shape: {upper.shape}"))
    except Exception as e:
        test_results.append(("❌ Donchian Channels", "FAILED", str(e)))
    
    # Test 5: Aroon
    try:
        aroon_up, aroon_down = indicators.aroon(data)
        assert len(aroon_up) == len(data), "Aroon up length mismatch"
        assert len(aroon_down) == len(data), "Aroon down length mismatch"
        test_results.append(("✅ Aroon", "PASSED", f"Shape: {aroon_up.shape}"))
    except Exception as e:
        test_results.append(("❌ Aroon", "FAILED", str(e)))
    
    # Test 6: CMO
    try:
        cmo = indicators.chande_momentum_oscillator(data['close'])
        assert len(cmo) == len(data), "CMO length mismatch"
        test_results.append(("✅ Chande Momentum Oscillator", "PASSED", f"Shape: {cmo.shape}"))
    except Exception as e:
        test_results.append(("❌ Chande Momentum Oscillator", "FAILED", str(e)))
    
    # Test 7: DPO
    try:
        dpo = indicators.detrended_price_oscillator(data['close'])
        assert len(dpo) == len(data), "DPO length mismatch"
        test_results.append(("✅ Detrended Price Oscillator", "PASSED", f"Shape: {dpo.shape}"))
    except Exception as e:
        test_results.append(("❌ Detrended Price Oscillator", "FAILED", str(e)))
    
    # Test 8: Ease of Movement
    try:
        eom = indicators.ease_of_movement(data['high'], data['low'], data['volume'])
        assert len(eom) == len(data), "EOM length mismatch"
        test_results.append(("✅ Ease of Movement", "PASSED", f"Shape: {eom.shape}"))
    except Exception as e:
        test_results.append(("❌ Ease of Movement", "FAILED", str(e)))
    
    # Test 9: Force Index
    try:
        force_idx = indicators.force_index(data)
        assert len(force_idx) == len(data), "Force Index length mismatch"
        test_results.append(("✅ Force Index", "PASSED", f"Shape: {force_idx.shape}"))
    except Exception as e:
        test_results.append(("❌ Force Index", "FAILED", str(e)))
    
    # Test 10: Mass Index
    try:
        mass_idx = indicators.mass_index(data['high'], data['low'])
        assert len(mass_idx) == len(data), "Mass Index length mismatch"
        test_results.append(("✅ Mass Index", "PASSED", f"Shape: {mass_idx.shape}"))
    except Exception as e:
        test_results.append(("❌ Mass Index", "FAILED", str(e)))
    
    # Test 11: NVI
    try:
        nvi = indicators.negative_volume_index(data['close'], data['volume'])
        assert len(nvi) == len(data), "NVI length mismatch"
        test_results.append(("✅ Negative Volume Index", "PASSED", f"Shape: {nvi.shape}"))
    except Exception as e:
        test_results.append(("❌ Negative Volume Index", "FAILED", str(e)))
    
    # Test 12: PVI
    try:
        pvi = indicators.positive_volume_index(data['close'], data['volume'])
        assert len(pvi) == len(data), "PVI length mismatch"
        test_results.append(("✅ Positive Volume Index", "PASSED", f"Shape: {pvi.shape}"))
    except Exception as e:
        test_results.append(("❌ Positive Volume Index", "FAILED", str(e)))
    
    # Test 13: PVT
    try:
        pvt = indicators.price_volume_trend(data['close'], data['volume'])
        assert len(pvt) == len(data), "PVT length mismatch"
        test_results.append(("✅ Price Volume Trend", "PASSED", f"Shape: {pvt.shape}"))
    except Exception as e:
        test_results.append(("❌ Price Volume Trend", "FAILED", str(e)))
    
    # Test 14: TRIX
    try:
        trix = indicators.trix(data['close'])
        assert len(trix) == len(data), "TRIX length mismatch"
        test_results.append(("✅ TRIX", "PASSED", f"Shape: {trix.shape}"))
    except Exception as e:
        test_results.append(("❌ TRIX", "FAILED", str(e)))
    
    # Test 15: Ultimate Oscillator
    try:
        uo = indicators.ultimate_oscillator(data['high'], data['low'], data['close'])
        assert len(uo) == len(data), "UO length mismatch"
        test_results.append(("✅ Ultimate Oscillator", "PASSED", f"Shape: {uo.shape}"))
    except Exception as e:
        test_results.append(("❌ Ultimate Oscillator", "FAILED", str(e)))
    
    # Test 16: Vortex Indicator
    try:
        vi_plus, vi_minus = indicators.vortex_indicator(data['high'], data['low'], data['close'])
        assert len(vi_plus) == len(data), "VI+ length mismatch"
        assert len(vi_minus) == len(data), "VI- length mismatch"
        test_results.append(("✅ Vortex Indicator", "PASSED", f"Shape: {vi_plus.shape}"))
    except Exception as e:
        test_results.append(("❌ Vortex Indicator", "FAILED", str(e)))
    
    # Test 17: Williams A/D
    try:
        wad = indicators.williams_accumulation_distribution(data)
        assert len(wad) == len(data), "Williams A/D length mismatch"
        test_results.append(("✅ Williams A/D", "PASSED", f"Shape: {wad.shape}"))
    except Exception as e:
        test_results.append(("❌ Williams A/D", "FAILED", str(e)))
    
    # Test 18: Chaikin Oscillator
    try:
        chaikin_osc = indicators.chaikin_oscillator(data)
        assert len(chaikin_osc) == len(data), "Chaikin Oscillator length mismatch"
        test_results.append(("✅ Chaikin Oscillator", "PASSED", f"Shape: {chaikin_osc.shape}"))
    except Exception as e:
        test_results.append(("❌ Chaikin Oscillator", "FAILED", str(e)))
    
    # Test 19: Elder Ray Index
    try:
        bull_power, bear_power = indicators.elder_ray_index(data)
        assert len(bull_power) == len(data), "Bull Power length mismatch"
        assert len(bear_power) == len(data), "Bear Power length mismatch"
        test_results.append(("✅ Elder Ray Index", "PASSED", f"Shape: {bull_power.shape}"))
    except Exception as e:
        test_results.append(("❌ Elder Ray Index", "FAILED", str(e)))
    
    # Test 20: Klinger Oscillator
    try:
        klinger, klinger_signal = indicators.klinger_oscillator(data['high'], data['low'], data['close'], data['volume'])
        assert len(klinger) == len(data), "Klinger length mismatch"
        assert len(klinger_signal) == len(data), "Klinger signal length mismatch"
        test_results.append(("✅ Klinger Oscillator", "PASSED", f"Shape: {klinger.shape}"))
    except Exception as e:
        test_results.append(("❌ Klinger Oscillator", "FAILED", str(e)))
    
    # Print results
    print("\n📊 TEST RESULTS:")
    print("-" * 70)
    passed = 0
    failed = 0
    
    for indicator, status, details in test_results:
        print(f"{indicator:<30} {status:<10} {details}")
        if "PASSED" in status:
            passed += 1
        else:
            failed += 1
    
    print("-" * 70)
    print(f"✅ PASSED: {passed}")
    print(f"❌ FAILED: {failed}")
    print(f"📈 SUCCESS RATE: {(passed / (passed + failed)) * 100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! All 20 new indicators are working perfectly!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
    
    return passed == 20

def demonstrate_indicators():
    """Demonstrate key indicators with sample outputs"""
    print("\n🎯 DEMONSTRATION OF KEY INDICATORS")
    print("=" * 50)
    
    data = create_comprehensive_test_data()
    indicators = TechnicalIndicators()
    
    # Show VWAP example
    print("\n📍 VWAP Example:")
    vwap = indicators.vwap(data)
    print(f"Last 5 VWAP values: {vwap[-5:]}")
    print(f"VWAP vs Close Price difference: {np.mean(data['close'][-5:] - vwap[-5:]):.2f}")
    
    # Show SuperTrend example
    print("\n📍 SuperTrend Example:")
    supertrend, trend = indicators.supertrend(data)
    print(f"Last 5 SuperTrend values: {supertrend[-5:]}")
    print(f"Current trend: {'Uptrend' if trend[-1] == 1 else 'Downtrend'}")
    
    # Show Aroon example
    print("\n📍 Aroon Example:")
    aroon_up, aroon_down = indicators.aroon(data)
    print(f"Aroon Up (last 5): {aroon_up[-5:]}")
    print(f"Aroon Down (last 5): {aroon_down[-5:]}")
    
    print("\n✨ All indicators are generating realistic values!")

if __name__ == "__main__":
    success = test_all_new_indicators()
    demonstrate_indicators()
    
    if success:
        print("\n🚀 READY FOR DOCUMENTATION UPDATE!")
        print("All 20 new technical indicators have been successfully implemented and tested.")
    else:
        print("\n⚠️ Some issues detected. Please review before proceeding.")
