"""
Technical Analysis Indicators Library
Comprehensive collection of technical indicators for quantitative analysis
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union

class TechnicalIndicators:
    """Collection of technical analysis indicators"""
    
    @staticmethod
    def sma(data: Union[List[float], np.ndarray, pd.Series], period: int) -> np.ndarray:
        """Simple Moving Average"""
        data = np.array(data)
        result = np.full(len(data), np.nan)
        for i in range(period - 1, len(data)):
            result[i] = np.mean(data[i - period + 1:i + 1])
        return result
    
    @staticmethod
    def ema(data: Union[List[float], np.ndarray, pd.Series], period: int) -> np.ndarray:
        """Exponential Moving Average"""
        data = np.array(data)
        alpha = 2 / (period + 1)
        result = np.full(len(data), np.nan)
        result[0] = data[0]
        
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        
        return result
    
    @staticmethod
    def rsi(data: Union[List[float], np.ndarray, pd.Series], period: int = 14) -> np.ndarray:
        """Relative Strength Index"""
        data = np.array(data)
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.full(len(data), np.nan)
        avg_losses = np.full(len(data), np.nan)
        
        # Initial averages
        if len(gains) >= period:
            avg_gains[period] = np.mean(gains[:period])
            avg_losses[period] = np.mean(losses[:period])
            
            # Exponential moving averages
            for i in range(period + 1, len(data)):
                avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
                avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: Union[List[float], np.ndarray, pd.Series], 
             fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD (Moving Average Convergence Divergence)"""
        data = np.array(data)
        ema_fast = TechnicalIndicators.ema(data, fast_period)
        ema_slow = TechnicalIndicators.ema(data, slow_period)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line[~np.isnan(macd_line)], signal_period)
        
        # Align signal line with macd line
        signal_aligned = np.full(len(macd_line), np.nan)
        valid_start = slow_period - 1
        signal_end = min(valid_start + len(signal_line), len(signal_aligned))
        actual_signal_length = signal_end - valid_start
        signal_aligned[valid_start:signal_end] = signal_line[:actual_signal_length]
        
        histogram = macd_line - signal_aligned
        
        return macd_line, signal_aligned, histogram
    
    @staticmethod
    def bollinger_bands(data: Union[List[float], np.ndarray, pd.Series], 
                       period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands"""
        data = np.array(data)
        sma = TechnicalIndicators.sma(data, period)
        
        rolling_std = np.full(len(data), np.nan)
        for i in range(period - 1, len(data)):
            rolling_std[i] = np.std(data[i - period + 1:i + 1])
        
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def stochastic_oscillator(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                            k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Oscillator"""
        highest_high = np.full(len(close), np.nan)
        lowest_low = np.full(len(close), np.nan)
        
        for i in range(k_period - 1, len(close)):
            highest_high[i] = np.max(high[i - k_period + 1:i + 1])
            lowest_low[i] = np.min(low[i - k_period + 1:i + 1])
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = TechnicalIndicators.sma(k_percent[~np.isnan(k_percent)], d_period)
        
        # Align D% with K%
        d_aligned = np.full(len(k_percent), np.nan)
        valid_start = k_period - 1
        d_aligned[valid_start:valid_start + len(d_percent)] = d_percent
        
        return k_percent, d_aligned
    
    @staticmethod
    def williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Williams %R"""
        highest_high = np.full(len(close), np.nan)
        lowest_low = np.full(len(close), np.nan)
        
        for i in range(period - 1, len(close)):
            highest_high[i] = np.max(high[i - period + 1:i + 1])
            lowest_low[i] = np.min(low[i - period + 1:i + 1])
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    @staticmethod
    def momentum(data: Union[List[float], np.ndarray, pd.Series], period: int = 10) -> np.ndarray:
        """Momentum Indicator"""
        data = np.array(data)
        momentum = np.full(len(data), np.nan)
        
        for i in range(period, len(data)):
            momentum[i] = data[i] - data[i - period]
        
        return momentum
    
    @staticmethod
    def roc(data: Union[List[float], np.ndarray, pd.Series], period: int = 10) -> np.ndarray:
        """Rate of Change"""
        data = np.array(data)
        roc = np.full(len(data), np.nan)
        
        for i in range(period, len(data)):
            if data[i - period] != 0:
                roc[i] = ((data[i] - data[i - period]) / data[i - period]) * 100
        
        return roc
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Average True Range"""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        atr = TechnicalIndicators.sma(tr, period)
        return atr
    
    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Average Directional Index"""
        tr = TechnicalIndicators.atr(high, low, close, 1)
        
        plus_dm = np.full(len(high), 0.0)
        minus_dm = np.full(len(high), 0.0)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            if high_diff > low_diff and high_diff > 0:
                plus_dm[i] = high_diff
            if low_diff > high_diff and low_diff > 0:
                minus_dm[i] = low_diff
        
        plus_di = 100 * TechnicalIndicators.sma(plus_dm, period) / TechnicalIndicators.sma(tr, period)
        minus_di = 100 * TechnicalIndicators.sma(minus_dm, period) / TechnicalIndicators.sma(tr, period)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = TechnicalIndicators.sma(dx, period)
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> np.ndarray:
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = TechnicalIndicators.sma(typical_price, period)
        
        mean_deviation = np.full(len(typical_price), np.nan)
        for i in range(period - 1, len(typical_price)):
            mean_deviation[i] = np.mean(np.abs(typical_price[i - period + 1:i + 1] - sma_tp[i]))
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci
    
    @staticmethod
    def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """On Balance Volume"""
        obv = np.zeros(len(close))
        obv[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        return obv
    
    @staticmethod
    def mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 14) -> np.ndarray:
        """Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = np.zeros(len(close))
        negative_flow = np.zeros(len(close))
        
        for i in range(1, len(close)):
            if typical_price[i] > typical_price[i-1]:
                positive_flow[i] = money_flow[i]
            elif typical_price[i] < typical_price[i-1]:
                negative_flow[i] = money_flow[i]
        
        mfi = np.full(len(close), np.nan)
        for i in range(period, len(close)):
            pos_sum = np.sum(positive_flow[i - period + 1:i + 1])
            neg_sum = np.sum(negative_flow[i - period + 1:i + 1])
            
            if neg_sum != 0:
                money_ratio = pos_sum / neg_sum
                mfi[i] = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    @staticmethod
    def ichimoku(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52) -> dict:
        """Ichimoku Cloud"""
        
        def calculate_line(high, low, period):
            result = np.full(len(high), np.nan)
            for i in range(period - 1, len(high)):
                period_high = np.max(high[i - period + 1:i + 1])
                period_low = np.min(low[i - period + 1:i + 1])
                result[i] = (period_high + period_low) / 2
            return result
        
        tenkan_sen = calculate_line(high, low, tenkan_period)
        kijun_sen = calculate_line(high, low, kijun_period)
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        senkou_span_b = calculate_line(high, low, senkou_b_period)
        
        # Chikou span is close shifted back 26 periods
        chikou_span = np.roll(close, kijun_period)
        chikou_span[:kijun_period] = np.nan
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def parabolic_sar(high: np.ndarray, low: np.ndarray, af_start: float = 0.02, af_max: float = 0.2) -> np.ndarray:
        """Parabolic SAR"""
        sar = np.full(len(high), np.nan)
        trend = np.ones(len(high))  # 1 for uptrend, -1 for downtrend
        af = af_start
        ep = high[0] if high[0] > low[0] else low[0]
        
        sar[0] = low[0]
        
        for i in range(1, len(high)):
            if trend[i-1] == 1:  # Uptrend
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_start, af_max)
                
                if low[i] <= sar[i]:
                    trend[i] = -1
                    sar[i] = ep
                    ep = low[i]
                    af = af_start
                else:
                    trend[i] = 1
            else:  # Downtrend
                sar[i] = sar[i-1] - af * (sar[i-1] - ep)
                
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_start, af_max)
                
                if high[i] >= sar[i]:
                    trend[i] = 1
                    sar[i] = ep
                    ep = high[i]
                    af = af_start
                else:
                    trend[i] = -1
        
        return sar

# Export for easy import
__all__ = ['TechnicalIndicators']
