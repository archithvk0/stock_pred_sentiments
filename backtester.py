"""
Backtesting Module for Stock Price Prediction
Simulates trading strategies and calculates performance metrics
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Represents a single trade"""
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    position: str  # 'long' or 'short'
    shares: int
    pnl: float
    pnl_pct: float
    signal: str  # 'buy' or 'sell'

@dataclass
class BacktestResult:
    """Contains backtesting results"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    trades: List[Trade]
    equity_curve: pd.Series
    daily_returns: pd.Series

class Backtester:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate per trade
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.reset()
    
    def reset(self):
        """Reset backtester state"""
        self.capital = self.initial_capital
        self.position = 0  # Current position (0 = no position, 1 = long, -1 = short)
        self.entry_price = 0
        self.entry_date = None
        self.trades = []
        self.equity_curve = []
        self.dates = []
    
    def run_backtest(self, data: pd.DataFrame, predictions: np.ndarray, 
                    confidence_threshold: float = 0.6) -> BacktestResult:
        """
        Run backtest on historical data with predictions
        
        Args:
            data: DataFrame with price data and features
            predictions: Model predictions (0 or 1)
            confidence_threshold: Minimum confidence to take a trade
        
        Returns:
            BacktestResult object
        """
        self.reset()
        
        # Ensure data has required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Add predictions to data
        data = data.copy()
        data['prediction'] = predictions
        
        # Run simulation
        for i, row in data.iterrows():
            current_date = row.name if hasattr(row, 'name') else i
            current_price = row['Close']
            
            # Update equity curve
            self._update_equity(current_price, current_date)
            
            # Check for exit signals
            if self.position != 0:
                self._check_exit_signal(row, current_date)
            
            # Check for entry signals
            if self.position == 0:
                self._check_entry_signal(row, current_date)
        
        # Close any open position at the end
        if self.position != 0:
            self._close_position(data.iloc[-1]['Close'], data.index[-1])
        
        # Calculate results
        return self._calculate_results(data)
    
    def _update_equity(self, current_price: float, current_date):
        """Update equity curve"""
        if self.position == 0:
            equity = self.capital
        elif self.position == 1:  # Long position
            equity = self.capital + (current_price - self.entry_price) * self.position_size
        else:  # Short position
            equity = self.capital + (self.entry_price - current_price) * self.position_size
        
        self.equity_curve.append(equity)
        self.dates.append(current_date)
    
    def _check_entry_signal(self, row: pd.Series, current_date):
        """Check for entry signals"""
        prediction = row['prediction']
        
        if prediction == 1:  # Buy signal
            self._open_long_position(row['Close'], current_date)
        elif prediction == 0:  # Sell signal (short)
            self._open_short_position(row['Close'], current_date)
    
    def _check_exit_signal(self, row: pd.Series, current_date):
        """Check for exit signals"""
        prediction = row['prediction']
        
        if self.position == 1 and prediction == 0:  # Exit long
            self._close_position(row['Close'], current_date)
        elif self.position == -1 and prediction == 1:  # Exit short
            self._close_position(row['Close'], current_date)
    
    def _open_long_position(self, price: float, date):
        """Open a long position"""
        self.position = 1
        self.entry_price = price
        self.entry_date = date
        self.position_size = int(self.capital * 0.95 / price)  # Use 95% of capital
        
        # Apply commission
        commission_cost = self.position_size * price * self.commission
        self.capital -= commission_cost
    
    def _open_short_position(self, price: float, date):
        """Open a short position"""
        self.position = -1
        self.entry_price = price
        self.entry_date = date
        self.position_size = int(self.capital * 0.95 / price)  # Use 95% of capital
        
        # Apply commission
        commission_cost = self.position_size * price * self.commission
        self.capital -= commission_cost
    
    def _close_position(self, price: float, date):
        """Close current position"""
        if self.position == 0:
            return
        
        # Calculate P&L
        if self.position == 1:  # Long position
            pnl = (price - self.entry_price) * self.position_size
        else:  # Short position
            pnl = (self.entry_price - price) * self.position_size
        
        # Apply commission
        commission_cost = self.position_size * price * self.commission
        pnl -= commission_cost
        
        # Update capital
        self.capital += pnl
        
        # Record trade
        trade = Trade(
            entry_date=self.entry_date,
            exit_date=date,
            entry_price=self.entry_price,
            exit_price=price,
            position='long' if self.position == 1 else 'short',
            shares=self.position_size,
            pnl=pnl,
            pnl_pct=pnl / (self.entry_price * self.position_size) * 100,
            signal='buy' if self.position == 1 else 'sell'
        )
        self.trades.append(trade)
        
        # Reset position
        self.position = 0
        self.entry_price = 0
        self.entry_date = None
    
    def _calculate_results(self, data: pd.DataFrame) -> BacktestResult:
        """Calculate backtesting results"""
        if not self.trades:
            return self._empty_results()
        
        # Calculate basic metrics
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        # Calculate annualized return
        start_date = data.index[0] if hasattr(data.index[0], 'year') else self.dates[0]
        end_date = data.index[-1] if hasattr(data.index[-1], 'year') else self.dates[-1]
        
        if hasattr(start_date, 'year'):
            years = (end_date - start_date).days / 365.25
        else:
            years = len(data) / 252  # Assuming trading days
        
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Calculate Sharpe ratio
        equity_series = pd.Series(self.equity_curve, index=self.dates)
        daily_returns = equity_series.pct_change().dropna()
        
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        max_drawdown = self._calculate_max_drawdown(equity_series)
        
        # Calculate trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        total_profit = sum(t.pnl for t in winning_trades)
        total_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            trades=self.trades,
            equity_curve=equity_series,
            daily_returns=daily_returns
        )
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return abs(drawdown.min())
    
    def _empty_results(self) -> BacktestResult:
        """Return empty results when no trades"""
        return BacktestResult(
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_win=0.0,
            avg_loss=0.0,
            trades=[],
            equity_curve=pd.Series(),
            daily_returns=pd.Series()
        )

class StrategyBacktester:
    """Advanced backtesting with multiple strategies"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.backtester = Backtester(initial_capital)
    
    def backtest_buy_and_hold(self, data: pd.DataFrame) -> BacktestResult:
        """Backtest buy and hold strategy"""
        # Buy at first day, hold until last day
        initial_price = data['Close'].iloc[0]
        final_price = data['Close'].iloc[-1]
        
        shares = int(self.initial_capital / initial_price)
        total_return = (final_price - initial_price) / initial_price
        
        # Create dummy trades
        trades = [
            Trade(
                entry_date=data.index[0],
                exit_date=data.index[-1],
                entry_price=initial_price,
                exit_price=final_price,
                position='long',
                shares=shares,
                pnl=(final_price - initial_price) * shares,
                pnl_pct=total_return * 100,
                signal='buy'
            )
        ]
        
        # Create equity curve
        equity_curve = data['Close'] / initial_price * self.initial_capital
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=total_return,  # Simplified
            sharpe_ratio=0.0,  # Would need to calculate properly
            max_drawdown=0.0,  # Would need to calculate properly
            win_rate=1.0 if total_return > 0 else 0.0,
            profit_factor=float('inf') if total_return > 0 else 0.0,
            total_trades=1,
            winning_trades=1 if total_return > 0 else 0,
            losing_trades=0 if total_return > 0 else 1,
            avg_win=total_return * self.initial_capital if total_return > 0 else 0.0,
            avg_loss=0.0 if total_return > 0 else abs(total_return * self.initial_capital),
            trades=trades,
            equity_curve=equity_curve,
            daily_returns=equity_curve.pct_change().dropna()
        )
    
    def backtest_momentum_strategy(self, data: pd.DataFrame, lookback: int = 20) -> BacktestResult:
        """Backtest momentum strategy"""
        # Calculate momentum (price change over lookback period)
        data = data.copy()
        data['momentum'] = data['Close'].pct_change(lookback)
        
        # Generate signals: buy when momentum > 0, sell when momentum < 0
        predictions = (data['momentum'] > 0).astype(int)
        predictions = predictions.fillna(0)
        
        return self.backtester.run_backtest(data, predictions.values)
    
    def backtest_mean_reversion_strategy(self, data: pd.DataFrame, lookback: int = 20) -> BacktestResult:
        """Backtest mean reversion strategy"""
        # Calculate moving average
        data = data.copy()
        data['ma'] = data['Close'].rolling(lookback).mean()
        data['deviation'] = (data['Close'] - data['ma']) / data['ma']
        
        # Generate signals: buy when price below MA, sell when above MA
        predictions = (data['deviation'] < -0.02).astype(int)  # 2% below MA
        predictions = predictions.fillna(0)
        
        return self.backtester.run_backtest(data, predictions.values)
    
    def compare_strategies(self, data: pd.DataFrame, predictions: np.ndarray) -> Dict[str, BacktestResult]:
        """Compare multiple strategies"""
        results = {}
        
        # ML strategy
        results['ml_strategy'] = self.backtester.run_backtest(data, predictions)
        
        # Buy and hold
        results['buy_and_hold'] = self.backtest_buy_and_hold(data)
        
        # Momentum strategy
        results['momentum'] = self.backtest_momentum_strategy(data)
        
        # Mean reversion strategy
        results['mean_reversion'] = self.backtest_mean_reversion_strategy(data)
        
        return results

class BacktestVisualizer:
    """Visualize backtesting results"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
    
    def plot_equity_curves(self, results: Dict[str, BacktestResult], title: str = "Equity Curves"):
        """Plot equity curves for multiple strategies"""
        plt.figure(figsize=(12, 8))
        
        for strategy_name, result in results.items():
            if not result.equity_curve.empty:
                plt.plot(result.equity_curve.index, result.equity_curve.values, 
                        label=f"{strategy_name} (Return: {result.total_return:.2%})")
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown(self, result: BacktestResult, title: str = "Drawdown Analysis"):
        """Plot drawdown analysis"""
        if result.equity_curve.empty:
            return
        
        equity = result.equity_curve
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        
        plt.figure(figsize=(12, 8))
        plt.plot(drawdown.index, drawdown.values * 100, color='red', alpha=0.7)
        plt.fill_between(drawdown.index, drawdown.values * 100, 0, color='red', alpha=0.3)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_trade_analysis(self, result: BacktestResult, title: str = "Trade Analysis"):
        """Plot trade analysis"""
        if not result.trades:
            return
        
        # Extract trade data
        trade_pnls = [t.pnl for t in result.trades]
        trade_dates = [t.exit_date for t in result.trades]
        
        plt.figure(figsize=(12, 8))
        
        # Plot cumulative P&L
        cumulative_pnl = np.cumsum(trade_pnls)
        plt.plot(trade_dates, cumulative_pnl, marker='o', alpha=0.7)
        
        # Color code winning vs losing trades
        for i, pnl in enumerate(trade_pnls):
            color = 'green' if pnl > 0 else 'red'
            plt.scatter(trade_dates[i], cumulative_pnl[i], color=color, s=50)
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Cumulative P&L')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def print_summary(self, results: Dict[str, BacktestResult]):
        """Print summary of backtesting results"""
        print("=" * 80)
        print("BACKTESTING RESULTS SUMMARY")
        print("=" * 80)
        
        for strategy_name, result in results.items():
            print(f"\n{strategy_name.upper()}:")
            print(f"  Total Return: {result.total_return:.2%}")
            print(f"  Annualized Return: {result.annualized_return:.2%}")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {result.max_drawdown:.2%}")
            print(f"  Win Rate: {result.win_rate:.2%}")
            print(f"  Profit Factor: {result.profit_factor:.2f}")
            print(f"  Total Trades: {result.total_trades}")
            print(f"  Winning Trades: {result.winning_trades}")
            print(f"  Losing Trades: {result.losing_trades}")
            if result.total_trades > 0:
                print(f"  Avg Win: ${result.avg_win:.2f}")
                print(f"  Avg Loss: ${result.avg_loss:.2f}")

# Example usage
if __name__ == "__main__":
    # Test the backtester
    import yfinance as yf
    
    # Get sample data
    data = yf.download('AAPL', start='2023-01-01', end='2024-01-01')
    
    # Create dummy predictions
    np.random.seed(42)
    predictions = np.random.choice([0, 1], size=len(data), p=[0.6, 0.4])
    
    # Run backtest
    backtester = Backtester(initial_capital=100000)
    result = backtester.run_backtest(data, predictions)
    
    # Print results
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Total Trades: {result.total_trades}")
    
    # Test strategy comparison
    strategy_backtester = StrategyBacktester()
    results = strategy_backtester.compare_strategies(data, predictions)
    
    # Visualize results
    visualizer = BacktestVisualizer()
    visualizer.plot_equity_curves(results)
    visualizer.print_summary(results) 