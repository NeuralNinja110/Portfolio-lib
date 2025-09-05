"""
Portfolio Management and Risk Analytics Module
Comprehensive portfolio management with advanced risk metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import scipy.stats as stats
from dataclasses import dataclass

@dataclass
class RiskMetrics:
    """Risk metrics container"""
    var_95: float  # Value at Risk (95%)
    var_99: float  # Value at Risk (99%)
    cvar_95: float  # Conditional VaR (95%)
    cvar_99: float  # Conditional VaR (99%)
    skewness: float
    kurtosis: float
    maximum_drawdown: float
    calmar_ratio: float
    sterling_ratio: float
    burke_ratio: float
    
class AdvancedPortfolioAnalytics:
    """Advanced portfolio analytics and risk management"""
    
    def __init__(self, returns: np.ndarray, benchmark_returns: Optional[np.ndarray] = None):
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        
    def calculate_var(self, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk"""
        if len(self.returns) == 0:
            return 0.0
        return np.percentile(self.returns, confidence_level * 100)
    
    def calculate_cvar(self, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self.calculate_var(confidence_level)
        cvar_returns = self.returns[self.returns <= var]
        return np.mean(cvar_returns) if len(cvar_returns) > 0 else 0.0
    
    def calculate_maximum_drawdown(self, equity_curve: np.ndarray) -> Tuple[float, int, int]:
        """Calculate maximum drawdown and duration"""
        if len(equity_curve) == 0:
            return 0.0, 0, 0
            
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        
        # Find the peak before max drawdown
        peak_idx = np.argmax(equity_curve[:max_dd_idx+1])
        
        # Find recovery point
        recovery_idx = max_dd_idx
        for i in range(max_dd_idx, len(equity_curve)):
            if equity_curve[i] >= peak[max_dd_idx]:
                recovery_idx = i
                break
        
        duration = recovery_idx - peak_idx
        
        return max_dd, duration, max_dd_idx
    
    def calculate_ulcer_index(self, equity_curve: np.ndarray) -> float:
        """Calculate Ulcer Index - measure of downside risk"""
        if len(equity_curve) == 0:
            return 0.0
            
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak * 100
        
        squared_drawdowns = drawdown ** 2
        ulcer_index = np.sqrt(np.mean(squared_drawdowns))
        
        return ulcer_index
    
    def calculate_burke_ratio(self, equity_curve: np.ndarray) -> float:
        """Calculate Burke Ratio"""
        if len(self.returns) == 0 or len(equity_curve) == 0:
            return 0.0
            
        excess_return = np.mean(self.returns) * 252  # Annualized
        ulcer_index = self.calculate_ulcer_index(equity_curve)
        
        return excess_return / ulcer_index if ulcer_index != 0 else 0.0
    
    def calculate_sterling_ratio(self, equity_curve: np.ndarray) -> float:
        """Calculate Sterling Ratio"""
        if len(self.returns) == 0:
            return 0.0
            
        annual_return = np.mean(self.returns) * 252
        max_dd, _, _ = self.calculate_maximum_drawdown(equity_curve)
        
        # Sterling ratio uses average drawdown, approximated as max_dd * 0.7
        avg_drawdown = abs(max_dd) * 0.7
        
        return annual_return / avg_drawdown if avg_drawdown != 0 else 0.0
    
    def calculate_tracking_error(self) -> float:
        """Calculate tracking error vs benchmark"""
        if self.benchmark_returns is None or len(self.returns) != len(self.benchmark_returns):
            return 0.0
            
        excess_returns = self.returns - self.benchmark_returns
        return np.std(excess_returns) * np.sqrt(252)
    
    def calculate_information_ratio(self) -> float:
        """Calculate Information Ratio"""
        if self.benchmark_returns is None:
            return 0.0
            
        excess_returns = self.returns - self.benchmark_returns
        tracking_error = self.calculate_tracking_error()
        
        return (np.mean(excess_returns) * 252) / tracking_error if tracking_error != 0 else 0.0
    
    def calculate_beta(self) -> float:
        """Calculate Beta vs benchmark"""
        if self.benchmark_returns is None or len(self.returns) != len(self.benchmark_returns):
            return 1.0
            
        covariance = np.cov(self.returns, self.benchmark_returns)[0, 1]
        benchmark_variance = np.var(self.benchmark_returns)
        
        return covariance / benchmark_variance if benchmark_variance != 0 else 1.0
    
    def calculate_alpha(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Alpha vs benchmark"""
        if self.benchmark_returns is None:
            return 0.0
            
        beta = self.calculate_beta()
        portfolio_return = np.mean(self.returns) * 252
        benchmark_return = np.mean(self.benchmark_returns) * 252
        
        alpha = portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
        
        return alpha
    
    def calculate_treynor_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Treynor Ratio"""
        if self.benchmark_returns is None:
            return 0.0
            
        beta = self.calculate_beta()
        portfolio_return = np.mean(self.returns) * 252
        
        return (portfolio_return - risk_free_rate) / beta if beta != 0 else 0.0
    
    def calculate_modigliani_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Modigliani-Modigliani Ratio"""
        if self.benchmark_returns is None:
            return 0.0
            
        portfolio_return = np.mean(self.returns) * 252
        portfolio_vol = np.std(self.returns) * np.sqrt(252)
        benchmark_vol = np.std(self.benchmark_returns) * np.sqrt(252)
        
        if portfolio_vol == 0:
            return 0.0
            
        adjusted_return = risk_free_rate + (portfolio_return - risk_free_rate) * (benchmark_vol / portfolio_vol)
        benchmark_return = np.mean(self.benchmark_returns) * 252
        
        return adjusted_return - benchmark_return
    
    def calculate_omega_ratio(self, threshold: float = 0.0) -> float:
        """Calculate Omega Ratio"""
        if len(self.returns) == 0:
            return 1.0
            
        excess_returns = self.returns - threshold
        gains = excess_returns[excess_returns > 0]
        losses = excess_returns[excess_returns < 0]
        
        if len(losses) == 0:
            return float('inf')
        if len(gains) == 0:
            return 0.0
            
        return np.sum(gains) / abs(np.sum(losses))
    
    def calculate_kappa_ratio(self, order: int = 3, threshold: float = 0.0) -> float:
        """Calculate Kappa Ratio (generalized downside risk measure)"""
        if len(self.returns) == 0:
            return 0.0
            
        excess_returns = self.returns - threshold
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
            
        lower_partial_moment = np.mean(np.abs(downside_returns) ** order) ** (1/order)
        mean_excess_return = np.mean(excess_returns)
        
        return mean_excess_return / lower_partial_moment if lower_partial_moment != 0 else 0.0
    
    def calculate_gain_pain_ratio(self) -> float:
        """Calculate Gain-to-Pain Ratio"""
        if len(self.returns) == 0:
            return 0.0
            
        gains = np.sum(self.returns[self.returns > 0])
        losses = abs(np.sum(self.returns[self.returns < 0]))
        
        return gains / losses if losses != 0 else float('inf')
    
    def calculate_comprehensive_risk_metrics(self, equity_curve: np.ndarray) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        var_95 = self.calculate_var(0.05)
        var_99 = self.calculate_var(0.01)
        cvar_95 = self.calculate_cvar(0.05)
        cvar_99 = self.calculate_cvar(0.01)
        
        skewness = stats.skew(self.returns) if len(self.returns) > 0 else 0.0
        kurtosis = stats.kurtosis(self.returns) if len(self.returns) > 0 else 0.0
        
        max_dd, _, _ = self.calculate_maximum_drawdown(equity_curve)
        
        # Calmar ratio
        annual_return = np.mean(self.returns) * 252 if len(self.returns) > 0 else 0.0
        calmar_ratio = annual_return / abs(max_dd) if max_dd != 0 else 0.0
        
        sterling_ratio = self.calculate_sterling_ratio(equity_curve)
        burke_ratio = self.calculate_burke_ratio(equity_curve)
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            skewness=skewness,
            kurtosis=kurtosis,
            maximum_drawdown=max_dd,
            calmar_ratio=calmar_ratio,
            sterling_ratio=sterling_ratio,
            burke_ratio=burke_ratio
        )

class PositionSizing:
    """Position sizing and risk management"""
    
    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly Criterion optimal position size"""
        if avg_loss == 0 or win_rate == 0:
            return 0.0
            
        win_loss_ratio = avg_win / abs(avg_loss)
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Cap Kelly fraction at 25% for safety
        return max(0.0, min(kelly_fraction, 0.25))
    
    @staticmethod
    def fixed_fractional(account_equity: float, risk_per_trade: float, stop_loss_pct: float) -> float:
        """Calculate position size using fixed fractional method"""
        if stop_loss_pct == 0:
            return 0.0
            
        risk_amount = account_equity * risk_per_trade
        position_size = risk_amount / stop_loss_pct
        
        return position_size
    
    @staticmethod
    def volatility_position_sizing(account_equity: float, target_volatility: float, 
                                 asset_volatility: float, correlation_adjustment: float = 1.0) -> float:
        """Calculate position size based on volatility targeting"""
        if asset_volatility == 0:
            return 0.0
            
        leverage = target_volatility / (asset_volatility * correlation_adjustment)
        position_size = account_equity * leverage
        
        return position_size
    
    @staticmethod
    def risk_parity_weights(covariance_matrix: np.ndarray) -> np.ndarray:
        """Calculate risk parity portfolio weights"""
        n_assets = covariance_matrix.shape[0]
        
        # Start with equal weights
        weights = np.ones(n_assets) / n_assets
        
        # Iterative algorithm to achieve risk parity
        for _ in range(100):  # Max iterations
            portfolio_vol = np.sqrt(weights.T @ covariance_matrix @ weights)
            marginal_contrib = (covariance_matrix @ weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Target equal risk contribution
            target_contrib = portfolio_vol / n_assets
            
            # Update weights
            weights = weights * target_contrib / contrib
            weights = weights / np.sum(weights)  # Normalize
            
            # Check convergence
            if np.max(np.abs(contrib - target_contrib)) < 1e-6:
                break
        
        return weights

class PerformanceAttribution:
    """Performance attribution analysis"""
    
    def __init__(self, portfolio_returns: np.ndarray, benchmark_returns: np.ndarray, 
                 weights: np.ndarray, asset_returns: np.ndarray):
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        self.weights = weights  # Portfolio weights over time
        self.asset_returns = asset_returns  # Individual asset returns
    
    def brinson_attribution(self, benchmark_weights: np.ndarray) -> Dict[str, np.ndarray]:
        """Brinson-Fachler performance attribution"""
        # Allocation effect: (wp - wb) * rb
        allocation_effect = (self.weights - benchmark_weights) * self.asset_returns
        
        # Selection effect: wb * (rp - rb)
        selection_effect = benchmark_weights * (self.asset_returns - self.asset_returns)  # Simplified
        
        # Interaction effect: (wp - wb) * (rp - rb)
        interaction_effect = (self.weights - benchmark_weights) * (self.asset_returns - self.asset_returns)
        
        return {
            'allocation': allocation_effect,
            'selection': selection_effect,
            'interaction': interaction_effect,
            'total': allocation_effect + selection_effect + interaction_effect
        }
    
    def calculate_sector_attribution(self, sector_mapping: Dict[str, str]) -> Dict[str, float]:
        """Calculate performance attribution by sector"""
        # Simplified sector attribution
        sector_contributions = {}
        
        for asset, sector in sector_mapping.items():
            if sector not in sector_contributions:
                sector_contributions[sector] = 0.0
            
            # Add weighted contribution of each asset to its sector
            # This is a simplified version - full implementation would be more complex
            
        return sector_contributions

# Export main classes
__all__ = ['AdvancedPortfolioAnalytics', 'PositionSizing', 'PerformanceAttribution', 'RiskMetrics']
