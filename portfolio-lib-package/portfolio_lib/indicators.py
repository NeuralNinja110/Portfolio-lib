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
    
    @staticmethod
    def klinger_oscillator(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
                          fast_period: int = 34, slow_period: int = 55, signal_period: int = 13) -> Tuple[np.ndarray, np.ndarray]:
        """Klinger Oscillator - measures the difference between money flow volume and cumulative volume"""
        typical_price = (high + low + close) / 3
        dm = np.where(typical_price > np.roll(typical_price, 1), 1, -1)
        dm[0] = 1  # First value default
        
        cm = np.zeros(len(volume))
        cm[0] = volume[0]
        for i in range(1, len(volume)):
            if dm[i] == dm[i-1]:
                cm[i] = cm[i-1] + volume[i]
            else:
                cm[i] = dm[i-1] * cm[i-1] + volume[i]
        
        vf = volume * np.abs(2 * ((dm * cm) / volume) - 1) * dm * 100
        
        kvo_fast = TechnicalIndicators.ema(vf, fast_period)
        kvo_slow = TechnicalIndicators.ema(vf, slow_period)
        kvo = kvo_fast - kvo_slow
        signal = TechnicalIndicators.ema(kvo[~np.isnan(kvo)], signal_period)
        
        # Align signal
        signal_aligned = np.full(len(kvo), np.nan)
        valid_start = max(fast_period, slow_period) - 1
        signal_end = min(valid_start + len(signal), len(signal_aligned))
        actual_length = signal_end - valid_start
        signal_aligned[valid_start:signal_end] = signal[:actual_length]
        
        return kvo, signal_aligned
    
    @staticmethod
    def price_channel(high: np.ndarray, low: np.ndarray, period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Price Channel - highest high and lowest low over a period"""
        upper = np.full(len(high), np.nan)
        lower = np.full(len(low), np.nan)
        
        for i in range(period - 1, len(high)):
            upper[i] = np.max(high[i - period + 1:i + 1])
            lower[i] = np.min(low[i - period + 1:i + 1])
        
        middle = (upper + lower) / 2
        return upper, middle, lower
    
    @staticmethod
    def donchian_channel(high: np.ndarray, low: np.ndarray, period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Donchian Channel - same as price channel but different name/usage"""
        return TechnicalIndicators.price_channel(high, low, period)
    
    @staticmethod
    def elder_force_index(close: np.ndarray, volume: np.ndarray, period: int = 13) -> np.ndarray:
        """Elder's Force Index - volume and price change momentum"""
        price_change = close - np.roll(close, 1)
        price_change[0] = 0
        raw_force = price_change * volume
        
        if period == 1:
            return raw_force
        else:
            return TechnicalIndicators.ema(raw_force, period)
    
    @staticmethod
    def ease_of_movement(high: np.ndarray, low: np.ndarray, volume: np.ndarray, period: int = 14) -> np.ndarray:
        """Ease of Movement - price movement relative to volume"""
        distance_moved = ((high + low) / 2) - np.roll(((high + low) / 2), 1)
        distance_moved[0] = 0
        
        high_low_range = high - low
        box_ratio = np.where(high_low_range != 0, volume / high_low_range, 0)
        
        one_period_evm = np.where(box_ratio != 0, distance_moved / box_ratio, 0)
        
        return TechnicalIndicators.sma(one_period_evm, period)
    
    @staticmethod
    def mass_index(high: np.ndarray, low: np.ndarray, period: int = 25, ema_period: int = 9) -> np.ndarray:
        """Mass Index - volatility indicator based on range expansion"""
        hl_ratio = high - low
        ema1 = TechnicalIndicators.ema(hl_ratio, ema_period)
        ema2 = TechnicalIndicators.ema(ema1, ema_period)
        
        mass_ratio = np.where(ema2 != 0, ema1 / ema2, 1)
        
        mass_index = np.full(len(high), np.nan)
        for i in range(period - 1, len(high)):
            mass_index[i] = np.sum(mass_ratio[i - period + 1:i + 1])
        
        return mass_index
    
    @staticmethod
    def negative_volume_index(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Negative Volume Index - cumulative indicator for down volume days"""
        nvi = np.zeros(len(close))
        nvi[0] = 1000  # Starting value
        
        for i in range(1, len(close)):
            if volume[i] < volume[i-1]:
                nvi[i] = nvi[i-1] * (close[i] / close[i-1])
            else:
                nvi[i] = nvi[i-1]
        
        return nvi
    
    @staticmethod
    def positive_volume_index(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Positive Volume Index - cumulative indicator for up volume days"""
        pvi = np.zeros(len(close))
        pvi[0] = 1000  # Starting value
        
        for i in range(1, len(close)):
            if volume[i] > volume[i-1]:
                pvi[i] = pvi[i-1] * (close[i] / close[i-1])
            else:
                pvi[i] = pvi[i-1]
        
        return pvi
    
    @staticmethod
    def price_volume_trend(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Price Volume Trend - volume-weighted momentum indicator"""
        price_change_pct = np.zeros(len(close))
        for i in range(1, len(close)):
            if close[i-1] != 0:
                price_change_pct[i] = (close[i] - close[i-1]) / close[i-1]
        
        pvt = np.zeros(len(close))
        pvt[0] = volume[0]
        
        for i in range(1, len(close)):
            pvt[i] = pvt[i-1] + (price_change_pct[i] * volume[i])
        
        return pvt
    
    @staticmethod
    def volume_accumulation(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Volume Accumulation - simplified A/D line using close only"""
        va = np.zeros(len(close))
        va[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                va[i] = va[i-1] + volume[i]
            elif close[i] < close[i-1]:
                va[i] = va[i-1] - volume[i]
            else:
                va[i] = va[i-1]
        
        return va
    
    @staticmethod
    def williams_ad(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Williams Accumulation/Distribution"""
        wad = np.zeros(len(close))
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                wad[i] = wad[i-1] + (close[i] - min(close[i-1], low[i])) * volume[i]
            elif close[i] < close[i-1]:
                wad[i] = wad[i-1] + (close[i] - max(close[i-1], high[i])) * volume[i]
            else:
                wad[i] = wad[i-1]
        
        return wad
    
    @staticmethod
    def coppock_curve(close: np.ndarray, roc1_period: int = 14, roc2_period: int = 11, wma_period: int = 10) -> np.ndarray:
        """Coppock Curve - long-term momentum indicator"""
        roc1 = TechnicalIndicators.roc(close, roc1_period)
        roc2 = TechnicalIndicators.roc(close, roc2_period)
        
        roc_sum = roc1 + roc2
        
        # Weighted Moving Average
        weights = np.arange(1, wma_period + 1)
        coppock = np.full(len(close), np.nan)
        
        for i in range(wma_period - 1, len(roc_sum)):
            if not np.isnan(roc_sum[i - wma_period + 1:i + 1]).any():
                values = roc_sum[i - wma_period + 1:i + 1]
                coppock[i] = np.sum(values * weights) / np.sum(weights)
        
        return coppock
    
    @staticmethod
    def know_sure_thing(close: np.ndarray, 
                       roc1_period: int = 10, roc1_ma: int = 10,
                       roc2_period: int = 15, roc2_ma: int = 10,
                       roc3_period: int = 20, roc3_ma: int = 10,
                       roc4_period: int = 30, roc4_ma: int = 15,
                       signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """Know Sure Thing (KST) - momentum oscillator"""
        roc1 = TechnicalIndicators.sma(TechnicalIndicators.roc(close, roc1_period), roc1_ma)
        roc2 = TechnicalIndicators.sma(TechnicalIndicators.roc(close, roc2_period), roc2_ma)
        roc3 = TechnicalIndicators.sma(TechnicalIndicators.roc(close, roc3_period), roc3_ma)
        roc4 = TechnicalIndicators.sma(TechnicalIndicators.roc(close, roc4_period), roc4_ma)
        
        kst = (roc1 * 1) + (roc2 * 2) + (roc3 * 3) + (roc4 * 4)
        signal = TechnicalIndicators.sma(kst[~np.isnan(kst)], signal_period)
        
        # Align signal
        signal_aligned = np.full(len(kst), np.nan)
        valid_start = max(roc1_period + roc1_ma, roc4_period + roc4_ma) - 1
        signal_end = min(valid_start + len(signal), len(signal_aligned))
        actual_length = signal_end - valid_start
        signal_aligned[valid_start:signal_end] = signal[:actual_length]
        
        return kst, signal_aligned
    
    @staticmethod
    def price_oscillator(close: np.ndarray, fast_period: int = 12, slow_period: int = 26) -> np.ndarray:
        """Price Oscillator - percentage difference between two moving averages"""
        fast_ma = TechnicalIndicators.ema(close, fast_period)
        slow_ma = TechnicalIndicators.ema(close, slow_period)
        
        po = np.where(slow_ma != 0, ((fast_ma - slow_ma) / slow_ma) * 100, 0)
        return po
    
    @staticmethod
    def ultimate_oscillator(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                           period1: int = 7, period2: int = 14, period3: int = 28) -> np.ndarray:
        """Ultimate Oscillator - momentum oscillator using three timeframes"""
        prior_close = np.roll(close, 1)
        prior_close[0] = close[0]
        
        bp = close - np.minimum(low, prior_close)  # Buying pressure
        tr = np.maximum(high - low, np.maximum(np.abs(high - prior_close), np.abs(low - prior_close)))
        
        avg1 = np.full(len(close), np.nan)
        avg2 = np.full(len(close), np.nan)
        avg3 = np.full(len(close), np.nan)
        
        for i in range(max(period1, period2, period3) - 1, len(close)):
            if i >= period1 - 1:
                bp_sum1 = np.sum(bp[i - period1 + 1:i + 1])
                tr_sum1 = np.sum(tr[i - period1 + 1:i + 1])
                avg1[i] = bp_sum1 / tr_sum1 if tr_sum1 != 0 else 0
            
            if i >= period2 - 1:
                bp_sum2 = np.sum(bp[i - period2 + 1:i + 1])
                tr_sum2 = np.sum(tr[i - period2 + 1:i + 1])
                avg2[i] = bp_sum2 / tr_sum2 if tr_sum2 != 0 else 0
            
            if i >= period3 - 1:
                bp_sum3 = np.sum(bp[i - period3 + 1:i + 1])
                tr_sum3 = np.sum(tr[i - period3 + 1:i + 1])
                avg3[i] = bp_sum3 / tr_sum3 if tr_sum3 != 0 else 0
        
        uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
        return uo
    
    @staticmethod
    def triple_ema(data: Union[List[float], np.ndarray, pd.Series], period: int) -> np.ndarray:
        """Triple Exponential Moving Average (TEMA)"""
        ema1 = TechnicalIndicators.ema(data, period)
        ema2 = TechnicalIndicators.ema(ema1[~np.isnan(ema1)], period)
        ema3 = TechnicalIndicators.ema(ema2[~np.isnan(ema2)], period)
        
        # Align all EMAs
        tema = np.full(len(data), np.nan)
        valid_start = (period - 1) * 3  # Triple lag
        
        if len(ema3) > 0:
            tema[valid_start:valid_start + len(ema3)] = (3 * ema1[valid_start:valid_start + len(ema3)]) - \
                                                        (3 * ema2[:len(ema3)]) + ema3
        
        return tema
    
    @staticmethod
    def relative_vigor_index(open_price: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                           period: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Relative Vigor Index - momentum indicator comparing closing to opening"""
        numerator = close - open_price
        denominator = high - low
        
        # Simple moving averages
        num_ma = TechnicalIndicators.sma(numerator, period)
        den_ma = TechnicalIndicators.sma(denominator, period)
        
        rvi = np.where(den_ma != 0, num_ma / den_ma, 0)
        rvi_signal = TechnicalIndicators.sma(rvi[~np.isnan(rvi)], 4)
        
        # Align signal
        signal_aligned = np.full(len(rvi), np.nan)
        valid_start = period - 1 + 3  # Period + signal period - 1
        signal_end = min(valid_start + len(rvi_signal), len(signal_aligned))
        actual_length = signal_end - valid_start
        signal_aligned[valid_start:signal_end] = rvi_signal[:actual_length]
        
        return rvi, signal_aligned
    
    @staticmethod
    def schaff_trend_cycle(close: np.ndarray, fast_period: int = 23, slow_period: int = 50, cycle_period: int = 10) -> np.ndarray:
        """Schaff Trend Cycle - combines MACD with Stochastic"""
        # Calculate MACD
        macd_line, _, _ = TechnicalIndicators.macd(close, fast_period, slow_period, 1)
        
        # Calculate Stochastic of MACD
        stoch_k = np.full(len(close), np.nan)
        
        for i in range(cycle_period - 1, len(macd_line)):
            if not np.isnan(macd_line[i - cycle_period + 1:i + 1]).any():
                macd_window = macd_line[i - cycle_period + 1:i + 1]
                highest = np.max(macd_window)
                lowest = np.min(macd_window)
                
                if highest != lowest:
                    stoch_k[i] = 100 * (macd_line[i] - lowest) / (highest - lowest)
                else:
                    stoch_k[i] = 50
        
        # Smooth the result
        stc = TechnicalIndicators.ema(stoch_k[~np.isnan(stoch_k)], 3)
        
        # Align result
        result = np.full(len(close), np.nan)
        valid_start = slow_period - 1 + cycle_period - 1
        result[valid_start:valid_start + len(stc)] = stc
        
        return result
    
    @staticmethod
    def stochastic_rsi(close: np.ndarray, period: int = 14, stoch_period: int = 14, k_period: int = 3, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic RSI - Stochastic applied to RSI"""
        rsi = TechnicalIndicators.rsi(close, period)
        
        stoch_rsi = np.full(len(close), np.nan)
        
        for i in range(stoch_period - 1, len(rsi)):
            if not np.isnan(rsi[i - stoch_period + 1:i + 1]).any():
                rsi_window = rsi[i - stoch_period + 1:i + 1]
                highest_rsi = np.max(rsi_window)
                lowest_rsi = np.min(rsi_window)
                
                if highest_rsi != lowest_rsi:
                    stoch_rsi[i] = 100 * (rsi[i] - lowest_rsi) / (highest_rsi - lowest_rsi)
                else:
                    stoch_rsi[i] = 50
        
        # Calculate %K and %D
        k_line = TechnicalIndicators.sma(stoch_rsi[~np.isnan(stoch_rsi)], k_period)
        d_line = TechnicalIndicators.sma(k_line[~np.isnan(k_line)], d_period)
        
        # Align results
        k_aligned = np.full(len(close), np.nan)
        d_aligned = np.full(len(close), np.nan)
        
        valid_start_k = period + stoch_period + k_period - 3
        valid_start_d = valid_start_k + d_period - 1
        
        k_aligned[valid_start_k:valid_start_k + len(k_line)] = k_line
        d_aligned[valid_start_d:valid_start_d + len(d_line)] = d_line
        
        return k_aligned, d_aligned
    
    @staticmethod
    def vortex_indicator(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        """Vortex Indicator - trend indicator based on vortex movement"""
        prior_close = np.roll(close, 1)
        prior_close[0] = close[0]
        
        vm_plus = np.abs(high - prior_close)
        vm_minus = np.abs(low - prior_close)
        
        tr = np.maximum(high - low, np.maximum(np.abs(high - prior_close), np.abs(low - prior_close)))
        
        vi_plus = np.full(len(close), np.nan)
        vi_minus = np.full(len(close), np.nan)
        
        for i in range(period - 1, len(close)):
            sum_vm_plus = np.sum(vm_plus[i - period + 1:i + 1])
            sum_vm_minus = np.sum(vm_minus[i - period + 1:i + 1])
            sum_tr = np.sum(tr[i - period + 1:i + 1])
            
            if sum_tr != 0:
                vi_plus[i] = sum_vm_plus / sum_tr
                vi_minus[i] = sum_vm_minus / sum_tr
        
        return vi_plus, vi_minus

    @staticmethod
    def vwap(data: pd.DataFrame) -> np.ndarray:
        """Volume Weighted Average Price (VWAP)"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        return vwap.values

    @staticmethod
    def supertrend(data: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """SuperTrend indicator"""
        high, low, close = data['high'].values, data['low'].values, data['close'].values
        atr = TechnicalIndicators.atr(high, low, close, period)
        
        hl_avg = (high + low) / 2
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        supertrend = np.full(len(close), np.nan)
        trend = np.ones(len(close))  # 1 for uptrend, -1 for downtrend
        
        for i in range(1, len(close)):
            # Calculate final upper and lower bands
            if upper_band[i] < upper_band[i-1] or close[i-1] > upper_band[i-1]:
                final_upper = upper_band[i]
            else:
                final_upper = upper_band[i-1]
                
            if lower_band[i] > lower_band[i-1] or close[i-1] < lower_band[i-1]:
                final_lower = lower_band[i]
            else:
                final_lower = lower_band[i-1]
            
            # Determine trend
            if close[i] <= final_lower:
                trend[i] = -1
                supertrend[i] = final_upper
            elif close[i] >= final_upper:
                trend[i] = 1
                supertrend[i] = final_lower
            else:
                trend[i] = trend[i-1]
                supertrend[i] = final_upper if trend[i] == -1 else final_lower
        
        return supertrend, trend

    @staticmethod
    def keltner_channels(data: pd.DataFrame, period: int = 20, multiplier: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Keltner Channels"""
        high, low, close = data['high'].values, data['low'].values, data['close'].values
        middle = TechnicalIndicators.ema(close, period)
        atr = TechnicalIndicators.atr(high, low, close, period)
        
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        
        return upper, middle, lower

    @staticmethod
    def donchian_channels(data: pd.DataFrame, period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Donchian Channels"""
        high, low = data['high'].values, data['low'].values
        
        upper = np.full(len(high), np.nan)
        lower = np.full(len(low), np.nan)
        
        for i in range(period - 1, len(high)):
            upper[i] = np.max(high[i - period + 1:i + 1])
            lower[i] = np.min(low[i - period + 1:i + 1])
        
        middle = (upper + lower) / 2
        return upper, middle, lower

    @staticmethod
    def aroon(data: pd.DataFrame, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        """Aroon Up and Aroon Down"""
        high, low = data['high'].values, data['low'].values
        
        aroon_up = np.full(len(high), np.nan)
        aroon_down = np.full(len(low), np.nan)
        
        for i in range(period - 1, len(high)):
            high_period = high[i - period + 1:i + 1]
            low_period = low[i - period + 1:i + 1]
            
            periods_since_high = period - 1 - np.argmax(high_period)
            periods_since_low = period - 1 - np.argmax(low_period[::-1])
            
            aroon_up[i] = ((period - periods_since_high) / period) * 100
            aroon_down[i] = ((period - periods_since_low) / period) * 100
        
        return aroon_up, aroon_down

    @staticmethod
    def chande_momentum_oscillator(close: np.ndarray, period: int = 14) -> np.ndarray:
        """Chande Momentum Oscillator (CMO)"""
        momentum = np.diff(close)
        momentum = np.concatenate([[0], momentum])  # Prepend 0 for length consistency
        
        gains = np.where(momentum > 0, momentum, 0)
        losses = np.where(momentum < 0, -momentum, 0)
        
        cmo = np.full(len(close), np.nan)
        
        for i in range(period, len(close)):
            sum_gains = np.sum(gains[i - period + 1:i + 1])
            sum_losses = np.sum(losses[i - period + 1:i + 1])
            
            if sum_gains + sum_losses != 0:
                cmo[i] = ((sum_gains - sum_losses) / (sum_gains + sum_losses)) * 100
        
        return cmo

    @staticmethod
    def detrended_price_oscillator(close: np.ndarray, period: int = 14) -> np.ndarray:
        """Detrended Price Oscillator (DPO)"""
        sma = TechnicalIndicators.sma(close, period)
        shift = period // 2 + 1
        
        dpo = np.full(len(close), np.nan)
        for i in range(shift, len(close)):
            if i - shift < len(sma) and not np.isnan(sma[i - shift]):
                dpo[i] = close[i] - sma[i - shift]
        
        return dpo

    @staticmethod
    def force_index(data: pd.DataFrame, period: int = 13) -> np.ndarray:
        """Force Index"""
        close, volume = data['close'].values, data['volume'].values
        price_change = np.diff(close)
        price_change = np.concatenate([[0], price_change])
        
        raw_force = price_change * volume
        force_index = TechnicalIndicators.ema(raw_force, period)
        
        return force_index

    @staticmethod
    def trix(close: np.ndarray, period: int = 14) -> np.ndarray:
        """TRIX - Rate of change of triple smoothed EMA"""
        ema1 = TechnicalIndicators.ema(close, period)
        ema2 = TechnicalIndicators.ema(ema1[~np.isnan(ema1)], period)
        ema3 = TechnicalIndicators.ema(ema2[~np.isnan(ema2)], period)
        
        trix = np.full(len(close), np.nan)
        ema3_extended = np.full(len(close), np.nan)
        
        # Calculate starting position for ema3 in the original array
        start_pos = len(close) - len(ema3)
        ema3_extended[start_pos:] = ema3
        
        for i in range(1, len(ema3_extended)):
            if not np.isnan(ema3_extended[i]) and not np.isnan(ema3_extended[i-1]) and ema3_extended[i-1] != 0:
                trix[i] = ((ema3_extended[i] - ema3_extended[i-1]) / ema3_extended[i-1]) * 10000
        
        return trix

    @staticmethod
    def williams_accumulation_distribution(data: pd.DataFrame) -> np.ndarray:
        """Williams Accumulation/Distribution Line"""
        high, low, close = data['high'].values, data['low'].values, data['close'].values
        
        wad = np.zeros(len(close))
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                wad[i] = wad[i-1] + (close[i] - np.minimum(close[i-1], low[i]))
            elif close[i] < close[i-1]:
                wad[i] = wad[i-1] + (close[i] - np.maximum(close[i-1], high[i]))
            else:
                wad[i] = wad[i-1]
        
        return wad

    @staticmethod
    def chaikin_oscillator(data: pd.DataFrame, fast_period: int = 3, slow_period: int = 10) -> np.ndarray:
        """Chaikin Oscillator"""
        high, low, close, volume = data['high'].values, data['low'].values, data['close'].values, data['volume'].values
        
        # Calculate Accumulation/Distribution Line
        mfv = ((close - low) - (high - close)) / (high - low)
        mfv = np.where(high == low, 0, mfv)  # Handle division by zero
        ad_line = np.cumsum(mfv * volume)
        
        # Calculate oscillator
        fast_ema = TechnicalIndicators.ema(ad_line, fast_period)
        slow_ema = TechnicalIndicators.ema(ad_line, slow_period)
        
        oscillator = fast_ema - slow_ema
        return oscillator

    @staticmethod
    def elder_ray_index(data: pd.DataFrame, period: int = 13) -> Tuple[np.ndarray, np.ndarray]:
        """Elder Ray Index (Bull Power and Bear Power)"""
        high, low, close = data['high'].values, data['low'].values, data['close'].values
        ema = TechnicalIndicators.ema(close, period)
        
        bull_power = high - ema
        bear_power = low - ema
        
        return bull_power, bear_power

# Export for easy import
__all__ = ['TechnicalIndicators']
