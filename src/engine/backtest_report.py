from dataclasses import dataclass
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

@dataclass
class BacktestReport:
    """Class for storing and formatting backtest results"""
    initial_capital: float
    final_portfolio_value: float
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades: pd.DataFrame
    equity_curve: pd.Series
    monthly_returns: pd.Series
    
    @classmethod
    def from_backtest_results(cls, portfolio: pd.DataFrame, trades: List[Dict], initial_capital: float):
        """Create a backtest report from raw backtest results"""
        equity_curve = portfolio['total']
        returns = portfolio['returns']
        
        # Ensure we have a datetime index
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise ValueError("Portfolio returns must have a DatetimeIndex")
        
        # Calculate metrics
        final_value = equity_curve.iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital * 100
        annual_return = (1 + total_return/100) ** (252/len(returns)) - 1
        
        # Risk metrics
        excess_returns = returns - 0.02/252  # Assuming 2% risk-free rate
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        # Trade statistics
        trades_df = pd.DataFrame(trades)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        losing_trades = trades_df[trades_df['pnl'] < 0]
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Monthly returns using 'ME' (month end) frequency
        monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        
        return cls(
            initial_capital=initial_capital,
            final_portfolio_value=final_value,
            total_return=total_return,
            annual_return=annual_return * 100,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trades=trades_df,
            equity_curve=equity_curve,
            monthly_returns=monthly_returns
        )
    
    def generate_report(self, output_dir: str = "reports") -> str:
        """Generate a formatted HTML report"""
        import jinja2
        import os
        
        # Create reports directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate plots
        self._generate_plots(output_dir)
        
        # Load template
        template_loader = jinja2.FileSystemLoader(searchpath="src/templates")
        template_env = jinja2.Environment(loader=template_loader)
        template = template_env.get_template("backtest_report.html")
        
        # Render report
        report_html = template.render(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metrics=self._get_metrics_dict(),
            monthly_returns=self.monthly_returns.round(4) * 100
        )
        
        # Save report
        report_path = os.path.join(output_dir, f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        with open(report_path, 'w') as f:
            f.write(report_html)
            
        return report_path
    
    def _get_metrics_dict(self) -> Dict[str, Any]:
        """Get formatted metrics dictionary"""
        return {
            "Initial Capital": f"${self.initial_capital:,.2f}",
            "Final Portfolio Value": f"${self.final_portfolio_value:,.2f}",
            "Total Return": f"{self.total_return:.2f}%",
            "Annual Return": f"{self.annual_return:.2f}%",
            "Sharpe Ratio": f"{self.sharpe_ratio:.2f}",
            "Maximum Drawdown": f"{self.max_drawdown:.2f}%",
            "Win Rate": f"{self.win_rate:.2f}%",
            "Profit Factor": f"{self.profit_factor:.2f}",
            "Number of Trades": f"{len(self.trades)}"
        }
    
    def _generate_plots(self, output_dir: str):
        """Generate and save analysis plots"""
        # Equity curve
        plt.figure(figsize=(12, 6))
        self.equity_curve.plot()
        plt.title('Portfolio Equity Curve')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'equity_curve.png'))
        plt.close()
        
        # Monthly returns heatmap
        monthly_returns_table = self.monthly_returns.round(4) * 100
        # Convert to a proper month/year format for the heatmap
        monthly_returns_df = pd.DataFrame({
            'Year': monthly_returns_table.index.year,
            'Month': monthly_returns_table.index.month,
            'Returns': monthly_returns_table.values
        })
        
        # Create pivot table for heatmap
        pivot_table = monthly_returns_df.pivot(
            index='Year',
            columns='Month',
            values='Returns'
        )
        
        # Plot heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Monthly Returns (%)'}
        )
        plt.title('Monthly Returns (%)')
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.savefig(os.path.join(output_dir, 'monthly_returns.png'))
        plt.close() 