"""
Portfolio-lib - Lightweight Python Backtesting Library
A comprehensive backtesting framework for algorithmic trading strategies
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

class Position:
    """Represents a trading position"""
    def __init__(self, symbol: str, quantity: float, entry_price: float, timestamp: datetime = None):
        self.symbol = symbol
        self.quantity = quantity
        self.shares = quantity  # Alias for compatibility
        self.entry_price = entry_price
        self.price = entry_price  # Alias for compatibility
        self.timestamp = timestamp or datetime.now()
        self.current_price = entry_price
        
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
        
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity
        
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price * 100
        
    @property
    def value(self) -> float:
        """Alias for market_value for compatibility"""
        return self.market_value

class Trade:
    """Represents a completed trade"""
    def __init__(self, symbol: str, quantity: float, price: float, timestamp: datetime, action: str = None, side: str = None, commission: float = 0.0):
        self.symbol = symbol
        # Support both action (BUY/SELL) and side (buy/sell) parameters
        if action is not None:
            self.action = action.upper()
            self.side = action.lower()
        elif side is not None:
            self.side = side.lower()
            self.action = side.upper()
        else:
            self.side = 'buy'
            self.action = 'BUY'
        self.quantity = quantity
        self.shares = quantity  # Alias for compatibility
        self.price = price
        self.timestamp = timestamp
        self.commission = commission
        
    @property
    def gross_value(self) -> float:
        return self.quantity * self.price
        
    @property
    def net_value(self) -> float:
        return self.gross_value - self.commission

class Portfolio:
    """Portfolio management class"""
    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []
        
    def add_trade(self, trade: Trade):
        """Add a trade to the portfolio"""
        self.trades.append(trade)
        
        # Check if it's a buy trade (handle both side and action attributes)
        is_buy = (hasattr(trade, 'side') and trade.side == 'buy') or (hasattr(trade, 'action') and trade.action == 'BUY')
        
        if is_buy:
            self.cash -= trade.net_value
            if trade.symbol in self.positions:
                # Average cost basis for additional shares
                existing = self.positions[trade.symbol]
                total_quantity = existing.quantity + trade.quantity
                avg_price = ((existing.quantity * existing.entry_price) + 
                           (trade.quantity * trade.price)) / total_quantity
                existing.quantity = total_quantity
                existing.entry_price = avg_price
            else:
                self.positions[trade.symbol] = Position(
                    trade.symbol, trade.quantity, trade.price, trade.timestamp
                )
        else:  # sell
            self.cash += trade.net_value
            if trade.symbol in self.positions:
                position = self.positions[trade.symbol]
                position.quantity -= trade.quantity
                if position.quantity <= 0:
                    del self.positions[trade.symbol]
    
    def update_prices(self, prices: Dict[str, float], timestamp: datetime):
        """Update current prices for all positions"""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]
        
        # Record equity curve
        total_equity = self.total_equity
        self.equity_curve.append(total_equity)
        self.timestamps.append(timestamp)
    
    @property
    def total_equity(self) -> float:
        """Calculate total portfolio equity"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    @property
    def total_return(self) -> float:
        """Calculate total return percentage"""
        if self.initial_cash == 0:
            return 0.0
        return (self.total_equity - self.initial_cash) / self.initial_cash * 100
        
    @property
    def total_value(self) -> float:
        """Alias for total_equity for compatibility"""
        return self.total_equity

class DataFeed:
    """Base class for data feeds"""
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.data: Dict[str, pd.DataFrame] = {}
    
    def load_data(self, start_date: str, end_date: str):
        """Load data for the specified date range"""
        raise NotImplementedError

class YFinanceDataFeed(DataFeed):
    """Yahoo Finance data feed"""
    def load_data(self, start_date: str, end_date: str):
        """Load data from Yahoo Finance"""
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                if not data.empty:
                    self.data[symbol] = data
                else:
                    print(f"Warning: No data found for {symbol}")
            except Exception as e:
                print(f"Error loading data for {symbol}: {e}")
    
    def fetch_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Alias for load_data that returns the data directly"""
        self.load_data(start_date, end_date)
        return self.data

class BaseIndicator:
    """Base class for technical indicators"""
    def __init__(self, period: int):
        self.period = period
        self.values: List[float] = []
        
    def update(self, value: float):
        """Update indicator with new value"""
        self.values.append(value)
        if len(self.values) > self.period * 2:  # Keep some history
            self.values = self.values[-self.period * 2:]
    
    @property
    def value(self) -> Optional[float]:
        """Get current indicator value"""
        if len(self.values) < self.period:
            return None
        return self._calculate()
    
    def _calculate(self) -> float:
        """Calculate indicator value - to be implemented by subclasses"""
        raise NotImplementedError

class SMA(BaseIndicator):
    """Simple Moving Average"""
    def _calculate(self) -> float:
        return np.mean(self.values[-self.period:])

class EMA(BaseIndicator):
    """Exponential Moving Average"""
    def __init__(self, period: int):
        super().__init__(period)
        self.alpha = 2 / (period + 1)
        self.ema_value = None
    
    def _calculate(self) -> float:
        if self.ema_value is None:
            self.ema_value = np.mean(self.values[-self.period:])
        else:
            self.ema_value = self.alpha * self.values[-1] + (1 - self.alpha) * self.ema_value
        return self.ema_value

class RSI(BaseIndicator):
    """Relative Strength Index"""
    def _calculate(self) -> float:
        if len(self.values) < self.period + 1:
            return 50.0
        
        deltas = np.diff(self.values[-self.period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class MACD:
    """Moving Average Convergence Divergence"""
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_ema = EMA(fast_period)
        self.slow_ema = EMA(slow_period)
        self.signal_ema = EMA(signal_period)
        self.macd_values: List[float] = []
    
    def update(self, value: float):
        self.fast_ema.update(value)
        self.slow_ema.update(value)
        
        if self.fast_ema.value is not None and self.slow_ema.value is not None:
            macd_value = self.fast_ema.value - self.slow_ema.value
            self.macd_values.append(macd_value)
            self.signal_ema.update(macd_value)
    
    @property
    def macd(self) -> Optional[float]:
        return self.macd_values[-1] if self.macd_values else None
    
    @property
    def signal(self) -> Optional[float]:
        return self.signal_ema.value
    
    @property
    def histogram(self) -> Optional[float]:
        if self.macd is not None and self.signal is not None:
            return self.macd - self.signal
        return None

class BollingerBands:
    """Bollinger Bands indicator"""
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
        self.values: List[float] = []
    
    def update(self, value: float):
        self.values.append(value)
        if len(self.values) > self.period * 2:
            self.values = self.values[-self.period * 2:]
    
    @property
    def middle_band(self) -> Optional[float]:
        if len(self.values) < self.period:
            return None
        return np.mean(self.values[-self.period:])
    
    @property
    def upper_band(self) -> Optional[float]:
        middle = self.middle_band
        if middle is None or len(self.values) < self.period:
            return None
        std = np.std(self.values[-self.period:])
        return middle + (self.std_dev * std)
    
    @property
    def lower_band(self) -> Optional[float]:
        middle = self.middle_band
        if middle is None or len(self.values) < self.period:
            return None
        std = np.std(self.values[-self.period:])
        return middle - (self.std_dev * std)

# ============================================================================
# TREND INDICATORS
# ============================================================================

class WMA(BaseIndicator):
    """Weighted Moving Average"""
    def _calculate(self) -> float:
        weights = np.arange(1, self.period + 1)
        return np.average(self.values[-self.period:], weights=weights)

class DEMA(BaseIndicator):
    """Double Exponential Moving Average"""
    def __init__(self, period: int):
        super().__init__(period)
        self.ema1 = EMA(period)
        self.ema2 = EMA(period)
    
    def update(self, value: float):
        super().update(value)
        self.ema1.update(value)
        if self.ema1.value is not None:
            self.ema2.update(self.ema1.value)
    
    def _calculate(self) -> float:
        if self.ema1.value is None or self.ema2.value is None:
            return np.mean(self.values[-self.period:])
        return 2 * self.ema1.value - self.ema2.value

class TEMA(BaseIndicator):
    """Triple Exponential Moving Average"""
    def __init__(self, period: int):
        super().__init__(period)
        self.ema1 = EMA(period)
        self.ema2 = EMA(period)
        self.ema3 = EMA(period)
    
    def update(self, value: float):
        super().update(value)
        self.ema1.update(value)
        if self.ema1.value is not None:
            self.ema2.update(self.ema1.value)
        if self.ema2.value is not None:
            self.ema3.update(self.ema2.value)
    
    def _calculate(self) -> float:
        if None in [self.ema1.value, self.ema2.value, self.ema3.value]:
            return np.mean(self.values[-self.period:])
        return 3 * self.ema1.value - 3 * self.ema2.value + self.ema3.value

class KAMA(BaseIndicator):
    """Kaufman Adaptive Moving Average"""
    def __init__(self, period: int = 10, fast_sc: int = 2, slow_sc: int = 30):
        super().__init__(period)
        self.fast_sc = 2 / (fast_sc + 1)
        self.slow_sc = 2 / (slow_sc + 1)
        self.kama_value = None
    
    def _calculate(self) -> float:
        if len(self.values) < self.period + 1:
            return np.mean(self.values)
        
        change = abs(self.values[-1] - self.values[-self.period])
        volatility = sum(abs(self.values[i] - self.values[i-1]) for i in range(-self.period + 1, 0))
        
        if volatility == 0:
            er = 1
        else:
            er = change / volatility
        
        sc = (er * (self.fast_sc - self.slow_sc) + self.slow_sc) ** 2
        
        if self.kama_value is None:
            self.kama_value = self.values[-1]
        else:
            self.kama_value = self.kama_value + sc * (self.values[-1] - self.kama_value)
        
        return self.kama_value

class HullMA(BaseIndicator):
    """Hull Moving Average"""
    def __init__(self, period: int):
        super().__init__(period)
        self.wma_half = WMA(period // 2)
        self.wma_full = WMA(period)
        self.wma_sqrt = WMA(int(np.sqrt(period)))
        self.hull_values = []
    
    def update(self, value: float):
        super().update(value)
        self.wma_half.update(value)
        self.wma_full.update(value)
        
        if self.wma_half.value is not None and self.wma_full.value is not None:
            hull_value = 2 * self.wma_half.value - self.wma_full.value
            self.hull_values.append(hull_value)
            if len(self.hull_values) > int(np.sqrt(self.period)) * 2:
                self.hull_values = self.hull_values[-int(np.sqrt(self.period)) * 2:]
            self.wma_sqrt.update(hull_value)
    
    def _calculate(self) -> float:
        return self.wma_sqrt.value if self.wma_sqrt.value is not None else np.mean(self.values[-self.period:])

class VWAP:
    """Volume Weighted Average Price"""
    def __init__(self):
        self.price_volume = 0
        self.volume_sum = 0
        self.values = []
    
    def update(self, price: float, volume: float):
        self.price_volume += price * volume
        self.volume_sum += volume
        if self.volume_sum > 0:
            self.values.append(self.price_volume / self.volume_sum)
        else:
            self.values.append(price)
    
    @property
    def value(self) -> Optional[float]:
        return self.values[-1] if self.values else None

class ParabolicSAR:
    """Parabolic Stop and Reverse"""
    def __init__(self, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2):
        self.af_start = af_start
        self.af_increment = af_increment
        self.af_max = af_max
        self.sar = None
        self.trend = 1  # 1 for up, -1 for down
        self.af = af_start
        self.ep = None  # Extreme Point
        self.highs = []
        self.lows = []
    
    def update(self, high: float, low: float, close: float):
        self.highs.append(high)
        self.lows.append(low)
        
        if len(self.highs) < 2:
            self.sar = low
            self.ep = high
            return
        
        prev_sar = self.sar
        
        if self.trend == 1:  # Uptrend
            self.sar = prev_sar + self.af * (self.ep - prev_sar)
            if low <= self.sar:
                self.trend = -1
                self.sar = self.ep
                self.ep = low
                self.af = self.af_start
            else:
                if high > self.ep:
                    self.ep = high
                    self.af = min(self.af + self.af_increment, self.af_max)
        else:  # Downtrend
            self.sar = prev_sar - self.af * (prev_sar - self.ep)
            if high >= self.sar:
                self.trend = 1
                self.sar = self.ep
                self.ep = high
                self.af = self.af_start
            else:
                if low < self.ep:
                    self.ep = low
                    self.af = min(self.af + self.af_increment, self.af_max)
    
    @property
    def value(self) -> Optional[float]:
        return self.sar

# ============================================================================
# MOMENTUM INDICATORS
# ============================================================================

class Stochastic:
    """Stochastic Oscillator"""
    def __init__(self, k_period: int = 14, d_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period
        self.highs = []
        self.lows = []
        self.closes = []
        self.k_values = []
        self.d_sma = SMA(d_period)
    
    def update(self, high: float, low: float, close: float):
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        
        if len(self.highs) > self.k_period * 2:
            self.highs = self.highs[-self.k_period * 2:]
            self.lows = self.lows[-self.k_period * 2:]
            self.closes = self.closes[-self.k_period * 2:]
        
        if len(self.closes) >= self.k_period:
            highest_high = max(self.highs[-self.k_period:])
            lowest_low = min(self.lows[-self.k_period:])
            
            if highest_high - lowest_low != 0:
                k_value = 100 * (close - lowest_low) / (highest_high - lowest_low)
            else:
                k_value = 50
            
            self.k_values.append(k_value)
            self.d_sma.update(k_value)
    
    @property
    def k(self) -> Optional[float]:
        return self.k_values[-1] if self.k_values else None
    
    @property
    def d(self) -> Optional[float]:
        return self.d_sma.value

class WilliamsR(BaseIndicator):
    """Williams %R"""
    def __init__(self, period: int = 14):
        super().__init__(period)
        self.highs = []
        self.lows = []
    
    def update_hlc(self, high: float, low: float, close: float):
        self.highs.append(high)
        self.lows.append(low)
        self.values.append(close)
        
        if len(self.values) > self.period * 2:
            self.highs = self.highs[-self.period * 2:]
            self.lows = self.lows[-self.period * 2:]
            self.values = self.values[-self.period * 2:]
    
    def _calculate(self) -> float:
        if len(self.values) < self.period:
            return -50
        
        highest_high = max(self.highs[-self.period:])
        lowest_low = min(self.lows[-self.period:])
        close = self.values[-1]
        
        if highest_high - lowest_low != 0:
            return -100 * (highest_high - close) / (highest_high - lowest_low)
        return -50

class CCI(BaseIndicator):
    """Commodity Channel Index"""
    def __init__(self, period: int = 20):
        super().__init__(period)
        self.highs = []
        self.lows = []
        self.closes = []
        self.typical_prices = []
    
    def update_hlc(self, high: float, low: float, close: float):
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        
        typical_price = (high + low + close) / 3
        self.typical_prices.append(typical_price)
        
        if len(self.typical_prices) > self.period * 2:
            self.typical_prices = self.typical_prices[-self.period * 2:]
    
    def _calculate(self) -> float:
        if len(self.typical_prices) < self.period:
            return 0
        
        tp_sma = np.mean(self.typical_prices[-self.period:])
        mean_deviation = np.mean([abs(tp - tp_sma) for tp in self.typical_prices[-self.period:]])
        
        if mean_deviation == 0:
            return 0
        
        return (self.typical_prices[-1] - tp_sma) / (0.015 * mean_deviation)

class ROC(BaseIndicator):
    """Rate of Change"""
    def _calculate(self) -> float:
        if len(self.values) < self.period + 1:
            return 0
        return ((self.values[-1] - self.values[-self.period - 1]) / self.values[-self.period - 1]) * 100

class Momentum(BaseIndicator):
    """Momentum"""
    def _calculate(self) -> float:
        if len(self.values) < self.period + 1:
            return 0
        return self.values[-1] - self.values[-self.period - 1]

class StochasticRSI:
    """Stochastic RSI"""
    def __init__(self, rsi_period: int = 14, stoch_period: int = 14):
        self.rsi = RSI(rsi_period)
        self.stoch_period = stoch_period
        self.rsi_values = []
    
    def update(self, value: float):
        self.rsi.update(value)
        if self.rsi.value is not None:
            self.rsi_values.append(self.rsi.value)
            if len(self.rsi_values) > self.stoch_period * 2:
                self.rsi_values = self.rsi_values[-self.stoch_period * 2:]
    
    @property
    def value(self) -> Optional[float]:
        if len(self.rsi_values) < self.stoch_period:
            return None
        
        rsi_high = max(self.rsi_values[-self.stoch_period:])
        rsi_low = min(self.rsi_values[-self.stoch_period:])
        current_rsi = self.rsi_values[-1]
        
        if rsi_high - rsi_low != 0:
            return 100 * (current_rsi - rsi_low) / (rsi_high - rsi_low)
        return 50

class TRIX(BaseIndicator):
    """TRIX"""
    def __init__(self, period: int = 14):
        super().__init__(period)
        self.ema1 = EMA(period)
        self.ema2 = EMA(period)
        self.ema3 = EMA(period)
        self.trix_values = []
    
    def update(self, value: float):
        super().update(value)
        self.ema1.update(value)
        if self.ema1.value is not None:
            self.ema2.update(self.ema1.value)
        if self.ema2.value is not None:
            self.ema3.update(self.ema2.value)
            if len(self.trix_values) > 0:
                trix = 10000 * (self.ema3.value - self.trix_values[-1]) / self.trix_values[-1]
                self.trix_values.append(trix)
            else:
                self.trix_values.append(0)
    
    def _calculate(self) -> float:
        return self.trix_values[-1] if self.trix_values else 0

# ============================================================================
# VOLATILITY INDICATORS
# ============================================================================

class ATR:
    """Average True Range"""
    def __init__(self, period: int = 14):
        self.period = period
        self.tr_values = []
        self.atr_sma = SMA(period)
        self.prev_close = None
    
    def update(self, high: float, low: float, close: float):
        if self.prev_close is not None:
            tr1 = high - low
            tr2 = abs(high - self.prev_close)
            tr3 = abs(low - self.prev_close)
            tr = max(tr1, tr2, tr3)
            
            self.tr_values.append(tr)
            self.atr_sma.update(tr)
        
        self.prev_close = close
    
    @property
    def value(self) -> Optional[float]:
        return self.atr_sma.value

class TrueRange:
    """True Range"""
    def __init__(self):
        self.prev_close = None
        self.tr_value = None
    
    def update(self, high: float, low: float, close: float):
        if self.prev_close is not None:
            tr1 = high - low
            tr2 = abs(high - self.prev_close)
            tr3 = abs(low - self.prev_close)
            self.tr_value = max(tr1, tr2, tr3)
        else:
            self.tr_value = high - low
        
        self.prev_close = close
    
    @property
    def value(self) -> Optional[float]:
        return self.tr_value

class KeltnerChannels:
    """Keltner Channels"""
    def __init__(self, period: int = 20, multiplier: float = 2.0):
        self.ema = EMA(period)
        self.atr = ATR(period)
        self.multiplier = multiplier
    
    def update(self, high: float, low: float, close: float):
        self.ema.update(close)
        self.atr.update(high, low, close)
    
    @property
    def middle(self) -> Optional[float]:
        return self.ema.value
    
    @property
    def upper(self) -> Optional[float]:
        if self.ema.value is None or self.atr.value is None:
            return None
        return self.ema.value + (self.multiplier * self.atr.value)
    
    @property
    def lower(self) -> Optional[float]:
        if self.ema.value is None or self.atr.value is None:
            return None
        return self.ema.value - (self.multiplier * self.atr.value)

class DonchianChannels:
    """Donchian Channels"""
    def __init__(self, period: int = 20):
        self.period = period
        self.highs = []
        self.lows = []
    
    def update(self, high: float, low: float):
        self.highs.append(high)
        self.lows.append(low)
        
        if len(self.highs) > self.period * 2:
            self.highs = self.highs[-self.period * 2:]
            self.lows = self.lows[-self.period * 2:]
    
    @property
    def upper(self) -> Optional[float]:
        if len(self.highs) < self.period:
            return None
        return max(self.highs[-self.period:])
    
    @property
    def lower(self) -> Optional[float]:
        if len(self.lows) < self.period:
            return None
        return min(self.lows[-self.period:])
    
    @property
    def middle(self) -> Optional[float]:
        if self.upper is None or self.lower is None:
            return None
        return (self.upper + self.lower) / 2

class ADX:
    """Average Directional Index"""
    def __init__(self, period: int = 14):
        self.period = period
        self.atr = ATR(period)
        self.plus_dm_sma = SMA(period)
        self.minus_dm_sma = SMA(period)
        self.dx_values = []
        self.adx_sma = SMA(period)
        self.prev_high = None
        self.prev_low = None
    
    def update(self, high: float, low: float, close: float):
        self.atr.update(high, low, close)
        
        if self.prev_high is not None and self.prev_low is not None:
            plus_dm = max(high - self.prev_high, 0) if high - self.prev_high > self.prev_low - low else 0
            minus_dm = max(self.prev_low - low, 0) if self.prev_low - low > high - self.prev_high else 0
            
            self.plus_dm_sma.update(plus_dm)
            self.minus_dm_sma.update(minus_dm)
            
            if self.atr.value and self.atr.value > 0:
                plus_di = 100 * self.plus_dm_sma.value / self.atr.value
                minus_di = 100 * self.minus_dm_sma.value / self.atr.value
                
                if plus_di + minus_di > 0:
                    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                    self.dx_values.append(dx)
                    self.adx_sma.update(dx)
        
        self.prev_high = high
        self.prev_low = low
    
    @property
    def value(self) -> Optional[float]:
        return self.adx_sma.value

# ============================================================================
# VOLUME INDICATORS
# ============================================================================

class OBV:
    """On-Balance Volume"""
    def __init__(self):
        self.obv_value = 0
        self.prev_close = None
    
    def update(self, close: float, volume: float):
        if self.prev_close is not None:
            if close > self.prev_close:
                self.obv_value += volume
            elif close < self.prev_close:
                self.obv_value -= volume
        
        self.prev_close = close
    
    @property
    def value(self) -> float:
        return self.obv_value

class AccumulationDistribution:
    """Accumulation/Distribution Line"""
    def __init__(self):
        self.ad_value = 0
    
    def update(self, high: float, low: float, close: float, volume: float):
        if high != low:
            ad_multiplier = ((close - low) - (high - close)) / (high - low)
            self.ad_value += ad_multiplier * volume
    
    @property
    def value(self) -> float:
        return self.ad_value

class ChaikinMoneyFlow:
    """Chaikin Money Flow"""
    def __init__(self, period: int = 20):
        self.period = period
        self.money_flow_multipliers = []
        self.volumes = []
    
    def update(self, high: float, low: float, close: float, volume: float):
        if high != low:
            mf_multiplier = ((close - low) - (high - close)) / (high - low)
        else:
            mf_multiplier = 0
        
        self.money_flow_multipliers.append(mf_multiplier * volume)
        self.volumes.append(volume)
        
        if len(self.money_flow_multipliers) > self.period * 2:
            self.money_flow_multipliers = self.money_flow_multipliers[-self.period * 2:]
            self.volumes = self.volumes[-self.period * 2:]
    
    @property
    def value(self) -> Optional[float]:
        if len(self.money_flow_multipliers) < self.period:
            return None
        
        sum_mf = sum(self.money_flow_multipliers[-self.period:])
        sum_volume = sum(self.volumes[-self.period:])
        
        return sum_mf / sum_volume if sum_volume > 0 else 0

class VROC(BaseIndicator):
    """Volume Rate of Change"""
    def __init__(self, period: int = 25):
        super().__init__(period)
        self.volumes = []
    
    def update_volume(self, volume: float):
        self.volumes.append(volume)
        if len(self.volumes) > self.period * 2:
            self.volumes = self.volumes[-self.period * 2:]
    
    def _calculate(self) -> float:
        if len(self.volumes) < self.period + 1:
            return 0
        
        current_volume = self.volumes[-1]
        past_volume = self.volumes[-self.period - 1]
        
        if past_volume > 0:
            return ((current_volume - past_volume) / past_volume) * 100
        return 0

class ForceIndex:
    """Force Index"""
    def __init__(self, period: int = 13):
        self.ema = EMA(period)
        self.prev_close = None
    
    def update(self, close: float, volume: float):
        if self.prev_close is not None:
            fi = volume * (close - self.prev_close)
            self.ema.update(fi)
        
        self.prev_close = close
    
    @property
    def value(self) -> Optional[float]:
        return self.ema.value

class VWMA(BaseIndicator):
    """Volume Weighted Moving Average"""
    def __init__(self, period: int):
        super().__init__(period)
        self.volumes = []
    
    def update_with_volume(self, price: float, volume: float):
        self.values.append(price)
        self.volumes.append(volume)
        
        if len(self.values) > self.period * 2:
            self.values = self.values[-self.period * 2:]
            self.volumes = self.volumes[-self.period * 2:]
    
    def _calculate(self) -> float:
        if len(self.values) < self.period:
            return np.mean(self.values)
        
        prices = self.values[-self.period:]
        volumes = self.volumes[-self.period:]
        
        total_volume = sum(volumes)
        if total_volume == 0:
            return np.mean(prices)
        
        return sum(p * v for p, v in zip(prices, volumes)) / total_volume

# ============================================================================
# STATISTICAL INDICATORS
# ============================================================================

class StandardDeviation(BaseIndicator):
    """Standard Deviation"""
    def _calculate(self) -> float:
        return np.std(self.values[-self.period:])

class Variance(BaseIndicator):
    """Variance"""
    def _calculate(self) -> float:
        return np.var(self.values[-self.period:])

class ZScore(BaseIndicator):
    """Z-Score"""
    def _calculate(self) -> float:
        if len(self.values) < self.period:
            return 0
        
        mean = np.mean(self.values[-self.period:])
        std = np.std(self.values[-self.period:])
        
        if std == 0:
            return 0
        
        return (self.values[-1] - mean) / std

class LinearRegression:
    """Linear Regression"""
    def __init__(self, period: int):
        self.period = period
        self.values = []
    
    def update(self, value: float):
        self.values.append(value)
        if len(self.values) > self.period * 2:
            self.values = self.values[-self.period * 2:]
    
    @property
    def slope(self) -> Optional[float]:
        if len(self.values) < self.period:
            return None
        
        y = np.array(self.values[-self.period:])
        x = np.arange(len(y))
        
        return np.polyfit(x, y, 1)[0]
    
    @property
    def intercept(self) -> Optional[float]:
        if len(self.values) < self.period:
            return None
        
        y = np.array(self.values[-self.period:])
        x = np.arange(len(y))
        
        return np.polyfit(x, y, 1)[1]
    
    @property
    def value(self) -> Optional[float]:
        if self.slope is None or self.intercept is None:
            return None
        
        return self.slope * (self.period - 1) + self.intercept

class Correlation:
    """Correlation between two series"""
    def __init__(self, period: int):
        self.period = period
        self.x_values = []
        self.y_values = []
    
    def update(self, x: float, y: float):
        self.x_values.append(x)
        self.y_values.append(y)
        
        if len(self.x_values) > self.period * 2:
            self.x_values = self.x_values[-self.period * 2:]
            self.y_values = self.y_values[-self.period * 2:]
    
    @property
    def value(self) -> Optional[float]:
        if len(self.x_values) < self.period:
            return None
        
        x = np.array(self.x_values[-self.period:])
        y = np.array(self.y_values[-self.period:])
        
        return np.corrcoef(x, y)[0, 1]

# ============================================================================
# PATTERN INDICATORS
# ============================================================================

class PivotPoints:
    """Pivot Points"""
    def __init__(self):
        self.pivot = None
        self.r1 = None
        self.r2 = None
        self.r3 = None
        self.s1 = None
        self.s2 = None
        self.s3 = None
    
    def update(self, high: float, low: float, close: float):
        self.pivot = (high + low + close) / 3
        
        self.r1 = 2 * self.pivot - low
        self.s1 = 2 * self.pivot - high
        
        self.r2 = self.pivot + (high - low)
        self.s2 = self.pivot - (high - low)
        
        self.r3 = high + 2 * (self.pivot - low)
        self.s3 = low - 2 * (high - self.pivot)

class FibonacciRetracement:
    """Fibonacci Retracement"""
    def __init__(self):
        self.high = None
        self.low = None
        self.levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    
    def update(self, high: float, low: float):
        if self.high is None or high > self.high:
            self.high = high
        if self.low is None or low < self.low:
            self.low = low
    
    def get_retracement_levels(self) -> Dict[str, float]:
        if self.high is None or self.low is None:
            return {}
        
        diff = self.high - self.low
        levels = {}
        
        for level in self.levels:
            levels[f"{level:.1%}"] = self.high - (diff * level)
        
        return levels

class ZigZag:
    """ZigZag Indicator"""
    def __init__(self, deviation: float = 5.0):
        self.deviation = deviation / 100
        self.highs = []
        self.lows = []
        self.last_pivot = None
        self.last_pivot_type = None  # 'high' or 'low'
        self.current_trend = None
    
    def update(self, high: float, low: float, close: float):
        self.highs.append(high)
        self.lows.append(low)
        
        if self.last_pivot is None:
            self.last_pivot = close
            return
        
        if self.last_pivot_type == 'high' or self.last_pivot_type is None:
            if low < self.last_pivot * (1 - self.deviation):
                self.last_pivot = low
                self.last_pivot_type = 'low'
                self.current_trend = 'down'
        
        if self.last_pivot_type == 'low' or self.last_pivot_type is None:
            if high > self.last_pivot * (1 + self.deviation):
                self.last_pivot = high
                self.last_pivot_type = 'high'
                self.current_trend = 'up'
    
    @property
    def trend(self) -> Optional[str]:
        return self.current_trend

# ============================================================================
# ADVANCED MOMENTUM INDICATORS
# ============================================================================

class UltimateOscillator:
    """Ultimate Oscillator"""
    def __init__(self, period1: int = 7, period2: int = 14, period3: int = 28):
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        self.bp_values = []  # Buying Pressure
        self.tr_values = []  # True Range
        self.prev_close = None
    
    def update(self, high: float, low: float, close: float):
        if self.prev_close is not None:
            bp = close - min(low, self.prev_close)
            tr = max(high, self.prev_close) - min(low, self.prev_close)
            
            self.bp_values.append(bp)
            self.tr_values.append(tr)
            
            if len(self.bp_values) > self.period3 * 2:
                self.bp_values = self.bp_values[-self.period3 * 2:]
                self.tr_values = self.tr_values[-self.period3 * 2:]
        
        self.prev_close = close
    
    @property
    def value(self) -> Optional[float]:
        if len(self.bp_values) < self.period3:
            return None
        
        avg1 = sum(self.bp_values[-self.period1:]) / sum(self.tr_values[-self.period1:]) if sum(self.tr_values[-self.period1:]) > 0 else 0
        avg2 = sum(self.bp_values[-self.period2:]) / sum(self.tr_values[-self.period2:]) if sum(self.tr_values[-self.period2:]) > 0 else 0
        avg3 = sum(self.bp_values[-self.period3:]) / sum(self.tr_values[-self.period3:]) if sum(self.tr_values[-self.period3:]) > 0 else 0
        
        return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

class AwesomeOscillator:
    """Awesome Oscillator"""
    def __init__(self, fast_period: int = 5, slow_period: int = 34):
        self.fast_sma = SMA(fast_period)
        self.slow_sma = SMA(slow_period)
        self.hl2_values = []
    
    def update(self, high: float, low: float):
        hl2 = (high + low) / 2
        self.hl2_values.append(hl2)
        self.fast_sma.update(hl2)
        self.slow_sma.update(hl2)
    
    @property
    def value(self) -> Optional[float]:
        if self.fast_sma.value is None or self.slow_sma.value is None:
            return None
        return self.fast_sma.value - self.slow_sma.value

class WavesTrend:
    """WavesTrend Oscillator"""
    def __init__(self, period1: int = 10, period2: int = 21):
        self.period1 = period1
        self.period2 = period2
        self.esa = EMA(period1)
        self.d_ema = EMA(period1)
        self.ci_ema = EMA(period2)
        self.tci_values = []
    
    def update(self, high: float, low: float, close: float):
        hlc3 = (high + low + close) / 3
        self.esa.update(hlc3)
        
        if self.esa.value is not None:
            d = abs(hlc3 - self.esa.value)
            self.d_ema.update(d)
            
            if self.d_ema.value is not None and self.d_ema.value > 0:
                ci = (hlc3 - self.esa.value) / (0.015 * self.d_ema.value)
                self.ci_ema.update(ci)
                self.tci_values.append(ci)
    
    @property
    def value(self) -> Optional[float]:
        return self.ci_ema.value

class DeMarker:
    """DeMarker Indicator"""
    def __init__(self, period: int = 14):
        self.period = period
        self.demax_values = []
        self.demin_values = []
        self.prev_high = None
        self.prev_low = None
    
    def update(self, high: float, low: float):
        if self.prev_high is not None and self.prev_low is not None:
            demax = max(high - self.prev_high, 0)
            demin = max(self.prev_low - low, 0)
            
            self.demax_values.append(demax)
            self.demin_values.append(demin)
            
            if len(self.demax_values) > self.period * 2:
                self.demax_values = self.demax_values[-self.period * 2:]
                self.demin_values = self.demin_values[-self.period * 2:]
        
        self.prev_high = high
        self.prev_low = low
    
    @property
    def value(self) -> Optional[float]:
        if len(self.demax_values) < self.period:
            return None
        
        avg_demax = np.mean(self.demax_values[-self.period:])
        avg_demin = np.mean(self.demin_values[-self.period:])
        
        if avg_demax + avg_demin == 0:
            return 0.5
        
        return avg_demax / (avg_demax + avg_demin)

class AroonIndicator:
    """Aroon Indicator"""
    def __init__(self, period: int = 25):
        self.period = period
        self.highs = []
        self.lows = []
    
    def update(self, high: float, low: float):
        self.highs.append(high)
        self.lows.append(low)
        
        if len(self.highs) > self.period * 2:
            self.highs = self.highs[-self.period * 2:]
            self.lows = self.lows[-self.period * 2:]
    
    @property
    def aroon_up(self) -> Optional[float]:
        if len(self.highs) < self.period:
            return None
        
        max_idx = np.argmax(self.highs[-self.period:])
        return ((self.period - 1 - max_idx) / (self.period - 1)) * 100
    
    @property
    def aroon_down(self) -> Optional[float]:
        if len(self.lows) < self.period:
            return None
        
        min_idx = np.argmin(self.lows[-self.period:])
        return ((self.period - 1 - min_idx) / (self.period - 1)) * 100
    
    @property
    def aroon_oscillator(self) -> Optional[float]:
        if self.aroon_up is None or self.aroon_down is None:
            return None
        return self.aroon_up - self.aroon_down

# ============================================================================
# ADDITIONAL TREND INDICATORS
# ============================================================================

class McGinleyDynamic(BaseIndicator):
    """McGinley Dynamic"""
    def __init__(self, period: int = 10):
        super().__init__(period)
        self.md_value = None
        self.k = 0.6
    
    def _calculate(self) -> float:
        if self.md_value is None:
            self.md_value = np.mean(self.values)
        
        if len(self.values) >= 2:
            md_factor = self.values[-1] / self.md_value
            n = self.period * (md_factor ** 4)
            alpha = 2 / (n + 1)
            self.md_value = self.md_value + alpha * (self.values[-1] - self.md_value)
        
        return self.md_value

class VerticalHorizontalFilter(BaseIndicator):
    """Vertical Horizontal Filter"""
    def _calculate(self) -> float:
        if len(self.values) < self.period:
            return 0
        
        numerator = abs(self.values[-1] - self.values[-self.period])
        denominator = sum(abs(self.values[i] - self.values[i-1]) 
                         for i in range(-self.period + 1, 0))
        
        return numerator / denominator if denominator > 0 else 0

class SuperTrend:
    """SuperTrend Indicator"""
    def __init__(self, period: int = 10, multiplier: float = 3.0):
        self.period = period
        self.multiplier = multiplier
        self.atr = ATR(period)
        self.basic_ub = []
        self.basic_lb = []
        self.final_ub = []
        self.final_lb = []
        self.supertrend = []
        self.trend = []
    
    def update(self, high: float, low: float, close: float):
        hl2 = (high + low) / 2
        self.atr.update(high, low, close)
        
        if self.atr.value is not None:
            basic_ub = hl2 + (self.multiplier * self.atr.value)
            basic_lb = hl2 - (self.multiplier * self.atr.value)
            
            self.basic_ub.append(basic_ub)
            self.basic_lb.append(basic_lb)
            
            # Calculate final bands
            if len(self.final_ub) == 0:
                final_ub = basic_ub
                final_lb = basic_lb
            else:
                final_ub = basic_ub if basic_ub < self.final_ub[-1] or close > self.final_ub[-1] else self.final_ub[-1]
                final_lb = basic_lb if basic_lb > self.final_lb[-1] or close < self.final_lb[-1] else self.final_lb[-1]
            
            self.final_ub.append(final_ub)
            self.final_lb.append(final_lb)
            
            # Determine trend and SuperTrend
            if len(self.supertrend) == 0:
                if close <= final_ub:
                    supertrend = final_ub
                    trend = -1
                else:
                    supertrend = final_lb
                    trend = 1
            else:
                prev_supertrend = self.supertrend[-1]
                prev_trend = self.trend[-1]
                
                if prev_trend == 1 and close > final_lb:
                    supertrend = final_lb
                    trend = 1
                elif prev_trend == 1 and close <= final_lb:
                    supertrend = final_ub
                    trend = -1
                elif prev_trend == -1 and close < final_ub:
                    supertrend = final_ub
                    trend = -1
                else:
                    supertrend = final_lb
                    trend = 1
            
            self.supertrend.append(supertrend)
            self.trend.append(trend)
    
    @property
    def value(self) -> Optional[float]:
        return self.supertrend[-1] if self.supertrend else None
    
    @property
    def trend_direction(self) -> Optional[int]:
        return self.trend[-1] if self.trend else None

class AlmaIndicator(BaseIndicator):
    """Arnaud Legoux Moving Average (ALMA)"""
    def __init__(self, period: int = 9, offset: float = 0.85, sigma: float = 6):
        super().__init__(period)
        self.offset = offset
        self.sigma = sigma
        self.weights = self._calculate_weights()
    
    def _calculate_weights(self):
        m = self.offset * (self.period - 1)
        s = self.period / self.sigma
        
        weights = []
        for i in range(self.period):
            weights.append(np.exp(-((i - m) ** 2) / (2 * s * s)))
        
        weight_sum = sum(weights)
        return [w / weight_sum for w in weights]
    
    def _calculate(self) -> float:
        if len(self.values) < self.period:
            return np.mean(self.values)
        
        return sum(self.values[i] * self.weights[i] 
                  for i in range(-self.period, 0))

# ============================================================================
# VOLUME-PRICE INDICATORS
# ============================================================================

class MoneyFlowIndex:
    """Money Flow Index"""
    def __init__(self, period: int = 14):
        self.period = period
        self.typical_prices = []
        self.raw_money_flows = []
        self.prev_tp = None
    
    def update(self, high: float, low: float, close: float, volume: float):
        tp = (high + low + close) / 3
        raw_money_flow = tp * volume
        
        self.typical_prices.append(tp)
        self.raw_money_flows.append(raw_money_flow)
        
        if len(self.typical_prices) > self.period * 2:
            self.typical_prices = self.typical_prices[-self.period * 2:]
            self.raw_money_flows = self.raw_money_flows[-self.period * 2:]
        
        self.prev_tp = tp
    
    @property
    def value(self) -> Optional[float]:
        if len(self.typical_prices) < self.period + 1:
            return None
        
        positive_flow = 0
        negative_flow = 0
        
        for i in range(-self.period, 0):
            if i == -self.period:
                continue
            if self.typical_prices[i] > self.typical_prices[i-1]:
                positive_flow += self.raw_money_flows[i]
            elif self.typical_prices[i] < self.typical_prices[i-1]:
                negative_flow += self.raw_money_flows[i]
        
        if negative_flow == 0:
            return 100
        
        money_ratio = positive_flow / negative_flow
        return 100 - (100 / (1 + money_ratio))

class VolumeOscillator:
    """Volume Oscillator"""
    def __init__(self, fast_period: int = 5, slow_period: int = 10):
        self.fast_sma = SMA(fast_period)
        self.slow_sma = SMA(slow_period)
    
    def update(self, volume: float):
        self.fast_sma.update(volume)
        self.slow_sma.update(volume)
    
    @property
    def value(self) -> Optional[float]:
        if self.fast_sma.value is None or self.slow_sma.value is None or self.slow_sma.value == 0:
            return None
        return ((self.fast_sma.value - self.slow_sma.value) / self.slow_sma.value) * 100

class EaseOfMovement:
    """Ease of Movement"""
    def __init__(self, period: int = 14):
        self.period = period
        self.eom_values = []
        self.sma = SMA(period)
        self.prev_high = None
        self.prev_low = None
    
    def update(self, high: float, low: float, volume: float):
        if self.prev_high is not None and self.prev_low is not None:
            distance_moved = ((high + low) / 2) - ((self.prev_high + self.prev_low) / 2)
            scale = volume / (high - low) if high != low else 0
            
            eom = distance_moved / scale if scale != 0 else 0
            self.eom_values.append(eom)
            self.sma.update(eom)
        
        self.prev_high = high
        self.prev_low = low
    
    @property
    def value(self) -> Optional[float]:
        return self.sma.value

class NegativeVolumeIndex:
    """Negative Volume Index"""
    def __init__(self):
        self.nvi = 100
        self.prev_close = None
        self.prev_volume = None
    
    def update(self, close: float, volume: float):
        if self.prev_close is not None and self.prev_volume is not None:
            if volume < self.prev_volume:
                self.nvi = self.nvi * (close / self.prev_close)
        
        self.prev_close = close
        self.prev_volume = volume
    
    @property
    def value(self) -> float:
        return self.nvi

class PositiveVolumeIndex:
    """Positive Volume Index"""
    def __init__(self):
        self.pvi = 100
        self.prev_close = None
        self.prev_volume = None
    
    def update(self, close: float, volume: float):
        if self.prev_close is not None and self.prev_volume is not None:
            if volume > self.prev_volume:
                self.pvi = self.pvi * (close / self.prev_close)
        
        self.prev_close = close
        self.prev_volume = volume
    
    @property
    def value(self) -> float:
        return self.pvi

# ============================================================================
# VOLATILITY AND BANDS
# ============================================================================

class StandardError(BaseIndicator):
    """Standard Error"""
    def _calculate(self) -> float:
        if len(self.values) < 2:
            return 0
        return np.std(self.values[-self.period:]) / np.sqrt(self.period)

class MeanDeviation(BaseIndicator):
    """Mean Deviation"""
    def _calculate(self) -> float:
        mean = np.mean(self.values[-self.period:])
        return np.mean([abs(x - mean) for x in self.values[-self.period:]])

class ChoppinessIndex(BaseIndicator):
    """Choppiness Index"""
    def __init__(self, period: int = 14):
        super().__init__(period)
        self.highs = []
        self.lows = []
        self.atr_values = []
    
    def update_hlc(self, high: float, low: float, close: float):
        self.highs.append(high)
        self.lows.append(low)
        self.values.append(close)
        
        if len(self.values) >= 2:
            tr = max(high - low, 
                    abs(high - self.values[-2]), 
                    abs(low - self.values[-2]))
            self.atr_values.append(tr)
        
        if len(self.highs) > self.period * 2:
            self.highs = self.highs[-self.period * 2:]
            self.lows = self.lows[-self.period * 2:]
            self.atr_values = self.atr_values[-self.period * 2:]
    
    def _calculate(self) -> float:
        if len(self.highs) < self.period or len(self.atr_values) < self.period:
            return 50
        
        highest_high = max(self.highs[-self.period:])
        lowest_low = min(self.lows[-self.period:])
        atr_sum = sum(self.atr_values[-self.period:])
        
        if highest_high - lowest_low == 0:
            return 50
        
        ci = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(self.period)
        return ci

class RelativeVolatilityIndex:
    """Relative Volatility Index"""
    def __init__(self, period: int = 10):
        self.period = period
        self.price_changes = []
        self.up_changes = []
        self.down_changes = []
        self.up_rsi = RSI(period)
        self.down_rsi = RSI(period)
        self.prev_price = None
    
    def update(self, price: float):
        if self.prev_price is not None:
            std_up = 0
            std_down = 0
            
            if len(self.price_changes) >= 9:  # Need at least 10 prices for std
                recent_prices = [self.prev_price] + self.price_changes[-8:]
                std_dev = np.std(recent_prices)
                
                if price > self.prev_price:
                    std_up = std_dev
                else:
                    std_down = std_dev
            
            self.up_rsi.update(std_up)
            self.down_rsi.update(std_down)
            self.price_changes.append(price)
        
        self.prev_price = price
    
    @property
    def value(self) -> Optional[float]:
        if self.up_rsi.value is None or self.down_rsi.value is None:
            return None
        return (self.up_rsi.value + self.down_rsi.value) / 2

# ============================================================================
# MARKET STRUCTURE INDICATORS
# ============================================================================

class SwingIndex:
    """Swing Index"""
    def __init__(self):
        self.prev_ohlc = None
        self.si_values = []
    
    def update(self, open_price: float, high: float, low: float, close: float):
        if self.prev_ohlc is not None:
            po, ph, pl, pc = self.prev_ohlc
            
            # Calculate A, B, C components
            a = abs(high - pc)
            b = abs(low - pc)
            c = abs(high - low)
            
            # Calculate K
            if a > b and a > c:
                k = a - 0.5 * b + 0.25 * (pc - po)
            elif b > a and b > c:
                k = b - 0.5 * a + 0.25 * (pc - po)
            else:
                k = c + 0.25 * (pc - po)
            
            # Calculate R
            r = max(abs(high - pc), abs(low - pc))
            
            # Calculate Swing Index
            if k != 0 and r != 0:
                si = 50 * ((close - pc + 0.5 * (close - open_price) + 0.25 * (pc - po)) / k) * (200 / r)
            else:
                si = 0
            
            self.si_values.append(si)
        
        self.prev_ohlc = (open_price, high, low, close)
    
    @property
    def value(self) -> Optional[float]:
        return self.si_values[-1] if self.si_values else None

class AccumulativeSwingIndex:
    """Accumulative Swing Index"""
    def __init__(self):
        self.swing_index = SwingIndex()
        self.asi_value = 0
    
    def update(self, open_price: float, high: float, low: float, close: float):
        self.swing_index.update(open_price, high, low, close)
        if self.swing_index.value is not None:
            self.asi_value += self.swing_index.value
    
    @property
    def value(self) -> float:
        return self.asi_value

class FractalIndicator:
    """Fractal Indicator"""
    def __init__(self, period: int = 5):
        self.period = period
        self.highs = []
        self.lows = []
        self.bull_fractals = []
        self.bear_fractals = []
    
    def update(self, high: float, low: float):
        self.highs.append(high)
        self.lows.append(low)
        
        if len(self.highs) >= self.period:
            middle = self.period // 2
            
            # Check for bullish fractal (high point)
            is_bull_fractal = True
            for i in range(self.period):
                if i != middle and self.highs[-self.period + i] >= self.highs[-self.period + middle]:
                    is_bull_fractal = False
                    break
            
            # Check for bearish fractal (low point)
            is_bear_fractal = True
            for i in range(self.period):
                if i != middle and self.lows[-self.period + i] <= self.lows[-self.period + middle]:
                    is_bear_fractal = False
                    break
            
            self.bull_fractals.append(is_bull_fractal)
            self.bear_fractals.append(is_bear_fractal)
        else:
            self.bull_fractals.append(False)
            self.bear_fractals.append(False)
    
    @property
    def is_bull_fractal(self) -> bool:
        return self.bull_fractals[-1] if self.bull_fractals else False
    
    @property
    def is_bear_fractal(self) -> bool:
        return self.bear_fractals[-1] if self.bear_fractals else False

# ============================================================================
# ADVANCED MATHEMATICAL INDICATORS
# ============================================================================

class HilbertTransform:
    """Hilbert Transform - Dominant Cycle Period"""
    def __init__(self):
        self.prices = []
        self.smooth_prices = []
        self.detrender = []
        self.i1 = []
        self.q1 = []
        self.ji = []
        self.jq = []
        self.i2 = []
        self.q2 = []
        self.re = []
        self.im = []
        self.period = []
        self.smooth_period = []
    
    def update(self, price: float):
        self.prices.append(price)
        
        if len(self.prices) < 7:
            self.smooth_prices.append(price)
            self.detrender.append(0)
            self.i1.append(0)
            self.q1.append(0)
            self.ji.append(0)
            self.jq.append(0)
            self.i2.append(0)
            self.q2.append(0)
            self.re.append(0)
            self.im.append(0)
            self.period.append(15)
            self.smooth_period.append(15)
            return
        
        # Smooth the price
        smooth = (4*price + 3*self.prices[-2] + 2*self.prices[-3] + self.prices[-4]) / 10
        self.smooth_prices.append(smooth)
        
        # Detrender
        detrender_val = (0.0962*smooth + 0.5769*self.smooth_prices[-2] 
                        - 0.5769*self.smooth_prices[-4] - 0.0962*self.smooth_prices[-6])
        self.detrender.append(detrender_val)
        
        # Compute InPhase and Quadrature components
        i1_val = self.detrender[-4] if len(self.detrender) >= 4 else 0
        q1_val = (0.0962*detrender_val + 0.5769*self.detrender[-2] 
                 - 0.5769*self.detrender[-4] - 0.0962*self.detrender[-6]) if len(self.detrender) >= 6 else 0
        
        self.i1.append(i1_val)
        self.q1.append(q1_val)
        
        # Advance the phase by 90 degrees
        ji_val = (0.0962*i1_val + 0.5769*self.i1[-2] 
                 - 0.5769*self.i1[-4] - 0.0962*self.i1[-6]) if len(self.i1) >= 6 else 0
        jq_val = (0.0962*q1_val + 0.5769*self.q1[-2] 
                 - 0.5769*self.q1[-4] - 0.0962*self.q1[-6]) if len(self.q1) >= 6 else 0
        
        self.ji.append(ji_val)
        self.jq.append(jq_val)
        
        # Phasor addition
        i2_val = i1_val - jq_val
        q2_val = q1_val + ji_val
        
        # Smooth the I and Q components
        i2_val = 0.2*i2_val + 0.8*self.i2[-1] if self.i2 else i2_val
        q2_val = 0.2*q2_val + 0.8*self.q2[-1] if self.q2 else q2_val
        
        self.i2.append(i2_val)
        self.q2.append(q2_val)
        
        # Homodyne Discriminator
        re_val = i2_val*self.i2[-1] + q2_val*self.q2[-1] if len(self.i2) >= 2 else 0
        im_val = i2_val*self.q2[-1] - q2_val*self.i2[-1] if len(self.i2) >= 2 else 0
        
        re_val = 0.2*re_val + 0.8*self.re[-1] if self.re else re_val
        im_val = 0.2*im_val + 0.8*self.im[-1] if self.im else im_val
        
        self.re.append(re_val)
        self.im.append(im_val)
        
        # Compute the period
        if im_val != 0 and re_val != 0:
            period_val = 2*np.pi / np.arctan(im_val/re_val)
        else:
            period_val = self.period[-1] if self.period else 15
        
        if period_val > 1.5*self.period[-1] if self.period else False:
            period_val = 1.5*self.period[-1]
        elif period_val < 0.67*self.period[-1] if self.period else False:
            period_val = 0.67*self.period[-1]
        
        if period_val < 6:
            period_val = 6
        elif period_val > 50:
            period_val = 50
        
        period_val = 0.2*period_val + 0.8*self.period[-1] if self.period else period_val
        
        self.period.append(period_val)
        self.smooth_period.append(0.33*period_val + 0.67*self.smooth_period[-1] if self.smooth_period else period_val)
    
    @property
    def dominant_cycle(self) -> Optional[float]:
        return self.smooth_period[-1] if self.smooth_period else None

class ChandeMomentumOscillator(BaseIndicator):
    """Chande Momentum Oscillator"""
    def _calculate(self) -> float:
        if len(self.values) < 2:
            return 0
        
        gains = 0
        losses = 0
        
        for i in range(-self.period, 0):
            if i >= -len(self.values) + 1:
                change = self.values[i] - self.values[i-1]
                if change > 0:
                    gains += change
                else:
                    losses -= change
        
        if gains + losses == 0:
            return 0
        
        return 100 * (gains - losses) / (gains + losses)

class KnowSureThing:
    """Know Sure Thing (KST)"""
    def __init__(self, roc1: int = 10, roc2: int = 15, roc3: int = 20, roc4: int = 30,
                 sma1: int = 10, sma2: int = 10, sma3: int = 10, sma4: int = 15, signal: int = 9):
        self.roc1 = ROC(roc1)
        self.roc2 = ROC(roc2)
        self.roc3 = ROC(roc3)
        self.roc4 = ROC(roc4)
        self.sma1 = SMA(sma1)
        self.sma2 = SMA(sma2)
        self.sma3 = SMA(sma3)
        self.sma4 = SMA(sma4)
        self.signal_sma = SMA(signal)
        self.kst_values = []
    
    def update(self, price: float):
        self.roc1.update(price)
        self.roc2.update(price)
        self.roc3.update(price)
        self.roc4.update(price)
        
        if self.roc1.value is not None:
            self.sma1.update(self.roc1.value)
        if self.roc2.value is not None:
            self.sma2.update(self.roc2.value)
        if self.roc3.value is not None:
            self.sma3.update(self.roc3.value)
        if self.roc4.value is not None:
            self.sma4.update(self.roc4.value)
        
        if all(sma.value is not None for sma in [self.sma1, self.sma2, self.sma3, self.sma4]):
            kst = (1*self.sma1.value + 2*self.sma2.value + 3*self.sma3.value + 4*self.sma4.value)
            self.kst_values.append(kst)
            self.signal_sma.update(kst)
    
    @property
    def kst(self) -> Optional[float]:
        return self.kst_values[-1] if self.kst_values else None
    
    @property
    def signal(self) -> Optional[float]:
        return self.signal_sma.value

# ============================================================================
# ADDITIONAL OSCILLATORS AND INDICATORS
# ============================================================================

class PercentagePriceOscillator:
    """Percentage Price Oscillator (PPO)"""
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_ema = EMA(fast_period)
        self.slow_ema = EMA(slow_period)
        self.signal_ema = EMA(signal_period)
        self.ppo_values = []
    
    def update(self, price: float):
        self.fast_ema.update(price)
        self.slow_ema.update(price)
        
        if self.fast_ema.value is not None and self.slow_ema.value is not None and self.slow_ema.value != 0:
            ppo = ((self.fast_ema.value - self.slow_ema.value) / self.slow_ema.value) * 100
            self.ppo_values.append(ppo)
            self.signal_ema.update(ppo)
    
    @property
    def ppo(self) -> Optional[float]:
        return self.ppo_values[-1] if self.ppo_values else None
    
    @property
    def signal(self) -> Optional[float]:
        return self.signal_ema.value
    
    @property
    def histogram(self) -> Optional[float]:
        if self.ppo is not None and self.signal is not None:
            return self.ppo - self.signal
        return None

class DetrendedPriceOscillator(BaseIndicator):
    """Detrended Price Oscillator (DPO)"""
    def _calculate(self) -> float:
        if len(self.values) < self.period:
            return 0
        
        sma = np.mean(self.values[-self.period:])
        lookback_index = self.period // 2 + 1
        
        if len(self.values) >= lookback_index:
            return self.values[-lookback_index] - sma
        return 0

class PriceOscillator:
    """Price Oscillator"""
    def __init__(self, fast_period: int = 10, slow_period: int = 20):
        self.fast_sma = SMA(fast_period)
        self.slow_sma = SMA(slow_period)
    
    def update(self, price: float):
        self.fast_sma.update(price)
        self.slow_sma.update(price)
    
    @property
    def value(self) -> Optional[float]:
        if self.fast_sma.value is None or self.slow_sma.value is None:
            return None
        return self.fast_sma.value - self.slow_sma.value

class SchaffTrendCycle:
    """Schaff Trend Cycle"""
    def __init__(self, cycle_period: int = 10, fast_period: int = 23, slow_period: int = 50):
        self.cycle_period = cycle_period
        self.fast_ema = EMA(fast_period)
        self.slow_ema = EMA(slow_period)
        self.macd_values = []
        self.k_values = []
        self.d_values = []
        self.stc_values = []
    
    def update(self, price: float):
        self.fast_ema.update(price)
        self.slow_ema.update(price)
        
        if self.fast_ema.value is not None and self.slow_ema.value is not None:
            macd = self.fast_ema.value - self.slow_ema.value
            self.macd_values.append(macd)
            
            if len(self.macd_values) >= self.cycle_period:
                # First stochastic calculation on MACD
                macd_low = min(self.macd_values[-self.cycle_period:])
                macd_high = max(self.macd_values[-self.cycle_period:])
                
                if macd_high - macd_low != 0:
                    k = 100 * (macd - macd_low) / (macd_high - macd_low)
                else:
                    k = 50
                
                self.k_values.append(k)
                
                if len(self.k_values) >= 3:
                    d = np.mean(self.k_values[-3:])
                    self.d_values.append(d)
                    
                    if len(self.d_values) >= self.cycle_period:
                        # Second stochastic calculation on %D
                        d_low = min(self.d_values[-self.cycle_period:])
                        d_high = max(self.d_values[-self.cycle_period:])
                        
                        if d_high - d_low != 0:
                            stc = 100 * (d - d_low) / (d_high - d_low)
                        else:
                            stc = 50
                        
                        self.stc_values.append(stc)
    
    @property
    def value(self) -> Optional[float]:
        return self.stc_values[-1] if self.stc_values else None

class ElderRayIndex:
    """Elder Ray Index (Bull Power and Bear Power)"""
    def __init__(self, period: int = 13):
        self.ema = EMA(period)
        self.bull_power = []
        self.bear_power = []
    
    def update(self, high: float, low: float, close: float):
        self.ema.update(close)
        
        if self.ema.value is not None:
            bull = high - self.ema.value
            bear = low - self.ema.value
            
            self.bull_power.append(bull)
            self.bear_power.append(bear)
    
    @property
    def bull_power_value(self) -> Optional[float]:
        return self.bull_power[-1] if self.bull_power else None
    
    @property
    def bear_power_value(self) -> Optional[float]:
        return self.bear_power[-1] if self.bear_power else None

class KaufmanEfficiencyRatio(BaseIndicator):
    """Kaufman Efficiency Ratio"""
    def _calculate(self) -> float:
        if len(self.values) < self.period + 1:
            return 0
        
        direction = abs(self.values[-1] - self.values[-self.period-1])
        volatility = sum(abs(self.values[i] - self.values[i-1]) 
                        for i in range(-self.period, 0))
        
        return direction / volatility if volatility > 0 else 0

class RelativeVigorIndex:
    """Relative Vigor Index"""
    def __init__(self, period: int = 10):
        self.period = period
        self.closes = []
        self.opens = []
        self.highs = []
        self.lows = []
        self.rvi_values = []
        self.signal_values = []
    
    def update(self, open_price: float, high: float, low: float, close: float):
        self.opens.append(open_price)
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        
        if len(self.closes) > self.period * 2:
            self.opens = self.opens[-self.period * 2:]
            self.highs = self.highs[-self.period * 2:]
            self.lows = self.lows[-self.period * 2:]
            self.closes = self.closes[-self.period * 2:]
        
        if len(self.closes) >= self.period:
            numerator = sum((self.closes[i] - self.opens[i]) 
                           for i in range(-self.period, 0))
            denominator = sum((self.highs[i] - self.lows[i]) 
                             for i in range(-self.period, 0))
            
            if denominator != 0:
                rvi = numerator / denominator
                self.rvi_values.append(rvi)
                
                # Signal line is 4-period SMA of RVI
                if len(self.rvi_values) >= 4:
                    signal = np.mean(self.rvi_values[-4:])
                    self.signal_values.append(signal)
    
    @property
    def rvi(self) -> Optional[float]:
        return self.rvi_values[-1] if self.rvi_values else None
    
    @property
    def signal(self) -> Optional[float]:
        return self.signal_values[-1] if self.signal_values else None

class MarketFacilitationIndex:
    """Market Facilitation Index"""
    def __init__(self):
        self.mfi_values = []
    
    def update(self, high: float, low: float, volume: float):
        if volume > 0:
            mfi = (high - low) / volume
        else:
            mfi = 0
        
        self.mfi_values.append(mfi)
    
    @property
    def value(self) -> Optional[float]:
        return self.mfi_values[-1] if self.mfi_values else None

class IchimokuKinkoHyo:
    """Ichimoku Kinko Hyo"""
    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26, 
                 senkou_b_period: int = 52, displacement: int = 26):
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement
        self.highs = []
        self.lows = []
        self.closes = []
        self.tenkan_sen = []
        self.kijun_sen = []
        self.senkou_span_a = []
        self.senkou_span_b = []
        self.chikou_span = []
    
    def update(self, high: float, low: float, close: float):
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        
        # Tenkan-sen (Conversion Line)
        if len(self.highs) >= self.tenkan_period:
            tenkan_high = max(self.highs[-self.tenkan_period:])
            tenkan_low = min(self.lows[-self.tenkan_period:])
            tenkan = (tenkan_high + tenkan_low) / 2
            self.tenkan_sen.append(tenkan)
        else:
            self.tenkan_sen.append(None)
        
        # Kijun-sen (Base Line)
        if len(self.highs) >= self.kijun_period:
            kijun_high = max(self.highs[-self.kijun_period:])
            kijun_low = min(self.lows[-self.kijun_period:])
            kijun = (kijun_high + kijun_low) / 2
            self.kijun_sen.append(kijun)
        else:
            self.kijun_sen.append(None)
        
        # Senkou Span A (Leading Span A)
        if len(self.tenkan_sen) > 0 and len(self.kijun_sen) > 0:
            if self.tenkan_sen[-1] is not None and self.kijun_sen[-1] is not None:
                senkou_a = (self.tenkan_sen[-1] + self.kijun_sen[-1]) / 2
                self.senkou_span_a.append(senkou_a)
            else:
                self.senkou_span_a.append(None)
        else:
            self.senkou_span_a.append(None)
        
        # Senkou Span B (Leading Span B)
        if len(self.highs) >= self.senkou_b_period:
            senkou_b_high = max(self.highs[-self.senkou_b_period:])
            senkou_b_low = min(self.lows[-self.senkou_b_period:])
            senkou_b = (senkou_b_high + senkou_b_low) / 2
            self.senkou_span_b.append(senkou_b)
        else:
            self.senkou_span_b.append(None)
        
        # Chikou Span (Lagging Span)
        self.chikou_span.append(close)
    
    @property
    def tenkan(self) -> Optional[float]:
        return self.tenkan_sen[-1] if self.tenkan_sen else None
    
    @property
    def kijun(self) -> Optional[float]:
        return self.kijun_sen[-1] if self.kijun_sen else None
    
    @property
    def senkou_a(self) -> Optional[float]:
        return self.senkou_span_a[-1] if self.senkou_span_a else None
    
    @property
    def senkou_b(self) -> Optional[float]:
        return self.senkou_span_b[-1] if self.senkou_span_b else None
    
    @property
    def chikou(self) -> Optional[float]:
        return self.chikou_span[-1] if self.chikou_span else None

class PVT:
    """Price Volume Trend"""
    def __init__(self):
        self.pvt_value = 0
        self.prev_close = None
    
    def update(self, close: float, volume: float):
        if self.prev_close is not None and self.prev_close != 0:
            pvt_change = volume * ((close - self.prev_close) / self.prev_close)
            self.pvt_value += pvt_change
        
        self.prev_close = close
    
    @property
    def value(self) -> float:
        return self.pvt_value

class TypicalPrice(BaseIndicator):
    """Typical Price (HLC/3)"""
    def __init__(self):
        super().__init__(1)
        self.highs = []
        self.lows = []
    
    def update_hlc(self, high: float, low: float, close: float):
        self.highs.append(high)
        self.lows.append(low)
        self.values.append(close)
    
    def _calculate(self) -> float:
        if not self.highs or not self.lows or not self.values:
            return 0
        return (self.highs[-1] + self.lows[-1] + self.values[-1]) / 3

class WeightedClose(BaseIndicator):
    """Weighted Close (HLCC/4)"""
    def __init__(self):
        super().__init__(1)
        self.highs = []
        self.lows = []
    
    def update_hlc(self, high: float, low: float, close: float):
        self.highs.append(high)
        self.lows.append(low)
        self.values.append(close)
    
    def _calculate(self) -> float:
        if not self.highs or not self.lows or not self.values:
            return 0
        return (self.highs[-1] + self.lows[-1] + 2 * self.values[-1]) / 4

class MedianPrice(BaseIndicator):
    """Median Price (HL/2)"""
    def __init__(self):
        super().__init__(1)
        self.highs = []
        self.lows = []
    
    def update_hl(self, high: float, low: float):
        self.highs.append(high)
        self.lows.append(low)
    
    @property
    def value(self) -> Optional[float]:
        if not self.highs or not self.lows:
            return None
        return (self.highs[-1] + self.lows[-1]) / 2

class AbsolutePriceOscillator:
    """Absolute Price Oscillator (APO)"""
    def __init__(self, fast_period: int = 12, slow_period: int = 26):
        self.fast_ema = EMA(fast_period)
        self.slow_ema = EMA(slow_period)
    
    def update(self, price: float):
        self.fast_ema.update(price)
        self.slow_ema.update(price)
    
    @property
    def value(self) -> Optional[float]:
        if self.fast_ema.value is None or self.slow_ema.value is None:
            return None
        return self.fast_ema.value - self.slow_ema.value

class BalanceOfPower:
    """Balance of Power"""
    def __init__(self):
        self.bop_values = []
    
    def update(self, open_price: float, high: float, low: float, close: float):
        if high != low:
            bop = (close - open_price) / (high - low)
        else:
            bop = 0
        
        self.bop_values.append(bop)
    
    @property
    def value(self) -> Optional[float]:
        return self.bop_values[-1] if self.bop_values else None

class CooppockCurve:
    """Coppock Curve"""
    def __init__(self, wma_period: int = 10, roc1_period: int = 14, roc2_period: int = 11):
        self.roc1 = ROC(roc1_period)
        self.roc2 = ROC(roc2_period)
        self.wma = WMA(wma_period)
        self.roc_sum_values = []
    
    def update(self, price: float):
        self.roc1.update(price)
        self.roc2.update(price)
        
        if self.roc1.value is not None and self.roc2.value is not None:
            roc_sum = self.roc1.value + self.roc2.value
            self.roc_sum_values.append(roc_sum)
            self.wma.update(roc_sum)
    
    @property
    def value(self) -> Optional[float]:
        return self.wma.value

class RainbowOscillator:
    """Rainbow Oscillator"""
    def __init__(self, period: int = 2):
        self.period = period
        self.sma_levels = [SMA(period) for _ in range(10)]
        self.rb_values = []
    
    def update(self, price: float):
        # First level
        self.sma_levels[0].update(price)
        
        # Subsequent levels
        for i in range(1, 10):
            if self.sma_levels[i-1].value is not None:
                self.sma_levels[i].update(self.sma_levels[i-1].value)
        
        # Calculate Rainbow Oscillator
        if self.sma_levels[0].value is not None and self.sma_levels[9].value is not None:
            if self.sma_levels[9].value != 0:
                rb = 100 * (price - self.sma_levels[9].value) / self.sma_levels[9].value
                self.rb_values.append(rb)
    
    @property
    def value(self) -> Optional[float]:
        return self.rb_values[-1] if self.rb_values else None

class KeltnerBands:
    """Keltner Bands (alternative implementation)"""
    def __init__(self, period: int = 20, multiplier: float = 1.5):
        self.ema = EMA(period)
        self.sma_tr = SMA(period)
        self.multiplier = multiplier
        self.prev_close = None
    
    def update(self, high: float, low: float, close: float):
        self.ema.update(close)
        
        if self.prev_close is not None:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
            self.sma_tr.update(tr)
        
        self.prev_close = close
    
    @property
    def middle(self) -> Optional[float]:
        return self.ema.value
    
    @property
    def upper(self) -> Optional[float]:
        if self.ema.value is None or self.sma_tr.value is None:
            return None
        return self.ema.value + (self.multiplier * self.sma_tr.value)
    
    @property
    def lower(self) -> Optional[float]:
        if self.ema.value is None or self.sma_tr.value is None:
            return None
        return self.ema.value - (self.multiplier * self.sma_tr.value)

class StochasticMomentumIndex:
    """Stochastic Momentum Index"""
    def __init__(self, k_period: int = 10, d_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period
        self.highs = []
        self.lows = []
        self.closes = []
        self.smi_values = []
        self.signal_sma = SMA(d_period)
    
    def update(self, high: float, low: float, close: float):
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        
        if len(self.closes) > self.k_period * 2:
            self.highs = self.highs[-self.k_period * 2:]
            self.lows = self.lows[-self.k_period * 2:]
            self.closes = self.closes[-self.k_period * 2:]
        
        if len(self.closes) >= self.k_period:
            highest_high = max(self.highs[-self.k_period:])
            lowest_low = min(self.lows[-self.k_period:])
            
            if highest_high != lowest_low:
                hl_range = highest_high - lowest_low
                close_position = close - (highest_high + lowest_low) / 2
                smi = 100 * (close_position / (hl_range / 2))
            else:
                smi = 0
            
            self.smi_values.append(smi)
            self.signal_sma.update(smi)
    
    @property
    def smi(self) -> Optional[float]:
        return self.smi_values[-1] if self.smi_values else None
    
    @property
    def signal(self) -> Optional[float]:
        return self.signal_sma.value

# Indicators module
class indicators:
    # Basic indicators (5)
    SMA = SMA
    EMA = EMA
    RSI = RSI
    MACD = MACD
    BollingerBands = BollingerBands
    
    # Trend indicators (11)
    WMA = WMA
    DEMA = DEMA
    TEMA = TEMA
    KAMA = KAMA
    HullMA = HullMA
    VWAP = VWAP
    ParabolicSAR = ParabolicSAR
    McGinleyDynamic = McGinleyDynamic
    VerticalHorizontalFilter = VerticalHorizontalFilter
    SuperTrend = SuperTrend
    AlmaIndicator = AlmaIndicator
    
    # Momentum indicators (25)
    Stochastic = Stochastic
    WilliamsR = WilliamsR
    CCI = CCI
    ROC = ROC
    Momentum = Momentum
    StochasticRSI = StochasticRSI
    TRIX = TRIX
    UltimateOscillator = UltimateOscillator
    AwesomeOscillator = AwesomeOscillator
    WavesTrend = WavesTrend
    DeMarker = DeMarker
    AroonIndicator = AroonIndicator
    ChandeMomentumOscillator = ChandeMomentumOscillator
    KnowSureThing = KnowSureThing
    PercentagePriceOscillator = PercentagePriceOscillator
    DetrendedPriceOscillator = DetrendedPriceOscillator
    PriceOscillator = PriceOscillator
    SchaffTrendCycle = SchaffTrendCycle
    ElderRayIndex = ElderRayIndex
    KaufmanEfficiencyRatio = KaufmanEfficiencyRatio
    RelativeVigorIndex = RelativeVigorIndex
    AbsolutePriceOscillator = AbsolutePriceOscillator
    CooppockCurve = CooppockCurve
    RainbowOscillator = RainbowOscillator
    StochasticMomentumIndex = StochasticMomentumIndex
    
    # Volatility indicators (11)
    ATR = ATR
    TrueRange = TrueRange
    KeltnerChannels = KeltnerChannels
    DonchianChannels = DonchianChannels
    ADX = ADX
    StandardError = StandardError
    MeanDeviation = MeanDeviation
    ChoppinessIndex = ChoppinessIndex
    RelativeVolatilityIndex = RelativeVolatilityIndex
    KeltnerBands = KeltnerBands
    
    # Volume indicators (13)
    OBV = OBV
    AccumulationDistribution = AccumulationDistribution
    ChaikinMoneyFlow = ChaikinMoneyFlow
    VROC = VROC
    ForceIndex = ForceIndex
    VWMA = VWMA
    MoneyFlowIndex = MoneyFlowIndex
    VolumeOscillator = VolumeOscillator
    EaseOfMovement = EaseOfMovement
    NegativeVolumeIndex = NegativeVolumeIndex
    PositiveVolumeIndex = PositiveVolumeIndex
    MarketFacilitationIndex = MarketFacilitationIndex
    PVT = PVT
    
    # Statistical indicators (5)
    StandardDeviation = StandardDeviation
    Variance = Variance
    ZScore = ZScore
    LinearRegression = LinearRegression
    Correlation = Correlation
    
    # Pattern indicators (3)
    PivotPoints = PivotPoints
    FibonacciRetracement = FibonacciRetracement
    ZigZag = ZigZag
    
    # Market Structure indicators (3)
    SwingIndex = SwingIndex
    AccumulativeSwingIndex = AccumulativeSwingIndex
    FractalIndicator = FractalIndicator
    
    # Advanced Mathematical indicators (1)
    HilbertTransform = HilbertTransform
    
    # Price indicators (4)
    TypicalPrice = TypicalPrice
    WeightedClose = WeightedClose
    MedianPrice = MedianPrice
    BalanceOfPower = BalanceOfPower
    
    # Complex indicators (1)
    IchimokuKinkoHyo = IchimokuKinkoHyo

class BaseStrategy:
    """Base strategy class"""
    def __init__(self):
        self.data: Dict[str, pd.DataFrame] = {}
        self.current_index = 0
        self.portfolio: Optional[Portfolio] = None
        self.indicators: Dict[str, Dict[str, BaseIndicator]] = {}
        
    def init_indicators(self):
        """Initialize technical indicators - to be implemented by subclasses"""
        pass
    
    def next(self):
        """Strategy logic for each bar - to be implemented by subclasses"""
        raise NotImplementedError
    
    def buy(self, symbol: str, size: float = 1.0, price: Optional[float] = None):
        """Place a buy order"""
        if self.portfolio is None:
            return
        
        current_price = price or self.data[symbol].iloc[self.current_index]['Close']
        current_time = self.data[symbol].index[self.current_index]
        
        # Calculate quantity based on size (as percentage of portfolio)
        available_cash = self.portfolio.cash
        order_value = available_cash * size
        quantity = order_value / current_price
        
        if available_cash >= order_value:
            trade = Trade(symbol, quantity, current_price, current_time, action='BUY')
            self.portfolio.add_trade(trade)
    
    def sell(self, symbol: str, size: Optional[float] = None, price: Optional[float] = None):
        """Place a sell order"""
        if self.portfolio is None or symbol not in self.portfolio.positions:
            return
        
        current_price = price or self.data[symbol].iloc[self.current_index]['Close']
        current_time = self.data[symbol].index[self.current_index]
        
        position = self.portfolio.positions[symbol]
        quantity = size or position.quantity
        
        trade = Trade(symbol, quantity, current_price, current_time, action='SELL')
        self.portfolio.add_trade(trade)
    
    def position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol"""
        if self.portfolio is None:
            return None
        return self.portfolio.positions.get(symbol)

class PerformanceMetrics:
    """Calculate performance metrics"""
    def __init__(self, equity_curve: List[float], timestamps: List[datetime], trades: List[Trade], initial_cash: float):
        self.equity_curve = np.array(equity_curve)
        self.timestamps = timestamps
        self.trades = trades
        self.initial_cash = initial_cash
        self.returns = self._calculate_returns()
    
    def _calculate_returns(self) -> np.ndarray:
        """Calculate daily returns"""
        if len(self.equity_curve) < 2:
            return np.array([])
        return np.diff(self.equity_curve) / self.equity_curve[:-1]
    
    @property
    def total_return(self) -> float:
        """Total return percentage"""
        if len(self.equity_curve) == 0:
            return 0.0
        return (self.equity_curve[-1] - self.initial_cash) / self.initial_cash * 100
    
    @property
    def annualized_return(self) -> float:
        """Annualized return"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        days = (self.timestamps[-1] - self.timestamps[0]).days
        if days == 0:
            return 0.0
        
        years = days / 365.25
        total_ret = self.equity_curve[-1] / self.initial_cash
        return (total_ret ** (1/years) - 1) * 100
    
    @property
    def volatility(self) -> float:
        """Annualized volatility"""
        if len(self.returns) == 0:
            return 0.0
        return np.std(self.returns) * np.sqrt(252) * 100
    
    @property
    def sharpe_ratio(self) -> float:
        """Sharpe ratio (assuming 0% risk-free rate)"""
        if len(self.returns) == 0 or np.std(self.returns) == 0:
            return 0.0
        return np.mean(self.returns) / np.std(self.returns) * np.sqrt(252)
    
    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown percentage"""
        if len(self.equity_curve) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (self.equity_curve - peak) / peak * 100
        return np.min(drawdown)
    
    @property
    def sortino_ratio(self) -> float:
        """Sortino ratio"""
        if len(self.returns) == 0:
            return 0.0
        
        negative_returns = self.returns[self.returns < 0]
        if len(negative_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(negative_returns)
        if downside_deviation == 0:
            return 0.0
        
        return np.mean(self.returns) / downside_deviation * np.sqrt(252)
    
    @property
    def calmar_ratio(self) -> float:
        """Calmar ratio"""
        max_dd = abs(self.max_drawdown)
        if max_dd == 0:
            return 0.0
        return self.annualized_return / max_dd
    
    @property
    def win_rate(self) -> float:
        """Win rate percentage"""
        if not self.trades:
            return 0.0
        
        # Group trades by symbol to calculate P&L per round trip
        positions = {}
        winning_trades = 0
        total_trades = 0
        
        for trade in self.trades:
            if trade.symbol not in positions:
                positions[trade.symbol] = []
            
            # Check if it's a buy trade (handle both side and action attributes)
            is_buy = (hasattr(trade, 'side') and trade.side == 'buy') or (hasattr(trade, 'action') and trade.action == 'BUY')
            
            if is_buy:
                positions[trade.symbol].append(trade)
            else:  # sell
                if positions[trade.symbol]:
                    buy_trade = positions[trade.symbol].pop(0)
                    pnl = (trade.price - buy_trade.price) * trade.quantity
                    if pnl > 0:
                        winning_trades += 1
                    total_trades += 1
        
        return (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    
    @property
    def profit_factor(self) -> float:
        """Profit factor"""
        gross_profit = 0.0
        gross_loss = 0.0
        
        positions = {}
        for trade in self.trades:
            if trade.symbol not in positions:
                positions[trade.symbol] = []
            
            # Check if it's a buy trade (handle both side and action attributes)
            is_buy = (hasattr(trade, 'side') and trade.side == 'buy') or (hasattr(trade, 'action') and trade.action == 'BUY')
            
            if is_buy:
                positions[trade.symbol].append(trade)
            else:  # sell
                if positions[trade.symbol]:
                    buy_trade = positions[trade.symbol].pop(0)
                    pnl = (trade.price - buy_trade.price) * trade.quantity
                    if pnl > 0:
                        gross_profit += pnl
                    else:
                        gross_loss += abs(pnl)
        
        return gross_profit / gross_loss if gross_loss > 0 else 0.0

class BacktestResults:
    """Backtest results container"""
    def __init__(self, portfolio: Portfolio, metrics: PerformanceMetrics):
        self.portfolio = portfolio
        self.metrics = metrics
    
    def summary(self) -> str:
        """Generate summary report"""
        return f"""
BACKTEST RESULTS SUMMARY
=====================================
Initial Capital: ${self.portfolio.initial_cash:,.2f}
Final Portfolio Value: ${self.portfolio.total_equity:,.2f}
Total Return: {self.metrics.total_return:.2f}%
Annualized Return: {self.metrics.annualized_return:.2f}%
Max Drawdown: {self.metrics.max_drawdown:.2f}%
Sharpe Ratio: {self.metrics.sharpe_ratio:.2f}
Sortino Ratio: {self.metrics.sortino_ratio:.2f}
Calmar Ratio: {self.metrics.calmar_ratio:.2f}
Volatility: {self.metrics.volatility:.2f}%
Win Rate: {self.metrics.win_rate:.2f}%
Profit Factor: {self.metrics.profit_factor:.2f}
Total Trades: {len(self.metrics.trades)}
"""

class Backtest:
    """Main backtesting engine"""
    def __init__(self, strategy: BaseStrategy, initial_cash: float = 100000.0):
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.data_feed: Optional[DataFeed] = None
        
    def add_data_source(self, data_feed: DataFeed):
        """Add data source"""
        self.data_feed = data_feed
    
    def run(self, start_date: str = '2020-01-01', end_date: str = None) -> BacktestResults:
        """Run the backtest"""
        if self.data_feed is None:
            raise ValueError("No data feed configured")
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Load data
        self.data_feed.load_data(start_date, end_date)
        self.strategy.data = self.data_feed.data
        
        # Initialize portfolio
        portfolio = Portfolio(self.initial_cash)
        self.strategy.portfolio = portfolio
        
        # Initialize strategy indicators
        self.strategy.init_indicators()
        
        # Get the common date range across all symbols
        if not self.strategy.data:
            raise ValueError("No data loaded")
        
        # Find common date range
        date_ranges = []
        for symbol, data in self.strategy.data.items():
            if not data.empty:
                date_ranges.append((data.index[0], data.index[-1]))
        
        if not date_ranges:
            raise ValueError("No valid data found")
        
        start_common = max(start for start, _ in date_ranges)
        end_common = min(end for _, end in date_ranges)
        
        # Align all data to common date range
        aligned_data = {}
        for symbol, data in self.strategy.data.items():
            aligned_data[symbol] = data.loc[start_common:end_common]
        
        self.strategy.data = aligned_data
        
        if not aligned_data:
            raise ValueError("No overlapping data found")
        
        # Get the primary symbol for iteration
        primary_symbol = list(aligned_data.keys())[0]
        primary_data = aligned_data[primary_symbol]
        
        # Run backtest
        for i in range(len(primary_data)):
            self.strategy.current_index = i
            current_timestamp = primary_data.index[i]
            
            # Update indicator values
            current_prices = {}
            for symbol, data in aligned_data.items():
                if i < len(data):
                    current_price = data.iloc[i]['Close']
                    current_prices[symbol] = current_price
                    
                    # Update indicators for this symbol
                    if symbol in self.strategy.indicators:
                        for indicator in self.strategy.indicators[symbol].values():
                            indicator.update(current_price)
            
            # Update portfolio prices
            portfolio.update_prices(current_prices, current_timestamp)
            
            # Execute strategy logic
            try:
                self.strategy.next()
            except Exception as e:
                print(f"Strategy error at {current_timestamp}: {e}")
        
        # Calculate metrics
        metrics = PerformanceMetrics(
            portfolio.equity_curve,
            portfolio.timestamps,
            portfolio.trades,
            self.initial_cash
        )
        
        return BacktestResults(portfolio, metrics)

class TechnicalIndicators:
    """Convenience class for technical indicator calculations"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2):
        """Bollinger Bands - returns (upper, middle, lower)"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD - returns (macd_line, signal_line, histogram)"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram


# Export main classes and functions
__all__ = [
    'BaseStrategy', 'Backtest', 'YFinanceDataFeed', 'TechnicalIndicators', 'indicators',
    'Position', 'Trade', 'Portfolio', 'PerformanceMetrics', 'BacktestResults'
]
