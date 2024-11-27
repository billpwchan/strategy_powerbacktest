from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime

class ReturnMetrics:
    """
    Handles all return-related metric calculations for the backtest report.
    """
    
    @staticmethod
    def calculate_total_return(final_value: float, initial_capital: float) -> float:
        """Calculate total return percentage"""
        return (final_value - initial_capital) / initial_capital
    
    @staticmethod
    def calculate_annual_return(portfolio: pd.DataFrame, total_return: float) -> float:
        """Calculate annualized return"""
        days = (portfolio.index[-1] - portfolio.index[0]).days
        return (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
    
    @staticmethod
    def calculate_sharpe_ratio(portfolio: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        returns = portfolio['returns']
        excess_returns = returns - risk_free_rate / 252
        volatility = returns.std() * np.sqrt(252)
        return (excess_returns.mean() * 252) / volatility if volatility != 0 else 0
    
    @staticmethod
    def calculate_monthly_returns(portfolio: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly returns from portfolio data"""
        monthly_returns = portfolio['returns'].resample('ME').apply(
            lambda x: (1 + x).prod() - 1
        )
        return pd.DataFrame({'returns': monthly_returns})
    
    @staticmethod
    def calculate_realized_pnl(trades: List[Dict[str, Any]]) -> float:
        """Calculate realized PnL from trades"""
        return sum(t.get('pnl', 0) for t in trades if 'pnl' in t)
    
    @staticmethod
    def calculate_floating_pnl(portfolio: pd.DataFrame, trades: List[Dict[str, Any]]) -> float:
        """Calculate floating PnL for current position"""
        current_position = portfolio['position'].iloc[-1]
        if current_position <= 0:
            return 0.0
            
        last_buy_trade = next((t for t in reversed(trades) if t['type'] == 'buy'), None)
        if not last_buy_trade:
            return 0.0
            
        current_price = portfolio['close'].iloc[-1]
        return (current_price - last_buy_trade['price']) * current_position 