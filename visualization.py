"""
Visualization module for plotting results and charts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Visualizer:
    """Create visualizations for experiment results."""
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """Initialize visualizer with plotting style."""
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use('default')
        
        sns.set_palette("husl")
    
    def plot_price_history(self, df, title="BTC Price History", save_path=None):
        """
        Plot historical price data.
        
        Args:
            df: DataFrame with OHLCV data
            title: Plot title
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Price plot
        axes[0].plot(df.index, df['close'], label='Close Price', linewidth=1.5)
        axes[0].set_ylabel('Price (USD)', fontsize=12)
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Volume plot
        axes[1].bar(df.index, df['volume'], alpha=0.5, label='Volume')
        axes[1].set_ylabel('Volume', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.close()
        return fig
    
    def plot_predictions_vs_actual(self, y_true, y_pred, model_name='Model', 
                                   save_path=None):
        """
        Plot predictions vs actual values.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Time series comparison
        indices = range(len(y_true))
        axes[0].plot(indices, y_true, label='Actual', linewidth=1.5, alpha=0.7)
        axes[0].plot(indices, y_pred, label='Predicted', linewidth=1.5, alpha=0.7)
        axes[0].set_xlabel('Sample Index', fontsize=12)
        axes[0].set_ylabel('Value', fontsize=12)
        axes[0].set_title(f'{model_name}: Predictions vs Actual', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[1].scatter(y_true, y_pred, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', 
                    linewidth=2, label='Perfect Prediction')
        
        axes[1].set_xlabel('Actual Values', fontsize=12)
        axes[1].set_ylabel('Predicted Values', fontsize=12)
        axes[1].set_title('Scatter Plot: Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.close()
        return fig
    
    def plot_training_history(self, history, model_name='Model', save_path=None):
        """
        Plot training history for deep learning models.
        
        Args:
            history: Training history object
            model_name: Name of the model
            save_path: Path to save figure (optional)
        """
        if history is None:
            print("No training history available")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history.history:
            axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title(f'{model_name}: Training Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Metric plot
        metric_key = 'mae' if 'mae' in history.history else 'accuracy'
        axes[1].plot(history.history[metric_key], label=f'Training {metric_key.upper()}', linewidth=2)
        if f'val_{metric_key}' in history.history:
            axes[1].plot(history.history[f'val_{metric_key}'], 
                        label=f'Validation {metric_key.upper()}', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel(metric_key.upper(), fontsize=12)
        axes[1].set_title(f'{model_name}: Training {metric_key.upper()}', 
                         fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.close()
        return fig
    
    def plot_model_comparison(self, results_df, metric='rmse', save_path=None):
        """
        Compare multiple models.
        
        Args:
            results_df: DataFrame with model results
            metric: Metric to compare
            save_path: Path to save figure (optional)
        """
        if results_df.empty:
            print("No results to plot")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = results_df.index
        values = results_df[metric]
        
        bars = ax.bar(models, values, alpha=0.7, edgecolor='black')
        
        # Color bars by performance
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'Model Comparison: {metric.upper()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (model, value) in enumerate(zip(models, values)):
            ax.text(i, value, f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.close()
        return fig
    
    def plot_feature_importance(self, feature_names, importance_scores, 
                               top_n=20, save_path=None):
        """
        Plot feature importance.
        
        Args:
            feature_names: List of feature names
            importance_scores: Importance scores
            top_n: Number of top features to show
            save_path: Path to save figure (optional)
        """
        # Sort features by importance
        indices = np.argsort(importance_scores)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_scores = importance_scores[indices]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_scores, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.close()
        return fig
    
    def plot_portfolio_performance(self, portfolio_values, trades=None, 
                                   initial_capital=10000, save_path=None):
        """
        Plot portfolio performance over time.
        
        Args:
            portfolio_values: Series of portfolio values
            trades: List of trade dictionaries (optional)
            initial_capital: Initial capital
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Portfolio value
        axes[0].plot(portfolio_values, linewidth=2, label='Portfolio Value')
        axes[0].axhline(y=initial_capital, color='r', linestyle='--', 
                       linewidth=1, label='Initial Capital')
        
        if trades:
            buy_trades = [t for t in trades if t['type'] == 'BUY']
            sell_trades = [t for t in trades if t['type'] == 'SELL']
            
            if buy_trades:
                buy_times = [t['time'] for t in buy_trades]
                buy_values = [portfolio_values[t['time']] for t in buy_trades]
                axes[0].scatter(buy_times, buy_values, color='green', 
                              marker='^', s=100, label='Buy', zorder=5)
            
            if sell_trades:
                sell_times = [t['time'] for t in sell_trades]
                sell_values = [portfolio_values[t['time']] for t in sell_trades]
                axes[0].scatter(sell_times, sell_values, color='red', 
                              marker='v', s=100, label='Sell', zorder=5)
        
        axes[0].set_ylabel('Portfolio Value (USD)', fontsize=12)
        axes[0].set_title('Portfolio Performance', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Returns
        returns = pd.Series(portfolio_values).pct_change() * 100
        axes[1].plot(returns, linewidth=1, alpha=0.7, label='Returns')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].fill_between(range(len(returns)), returns, 0, 
                            where=(returns > 0), alpha=0.3, color='green')
        axes[1].fill_between(range(len(returns)), returns, 0, 
                            where=(returns <= 0), alpha=0.3, color='red')
        
        axes[1].set_xlabel('Time Step', fontsize=12)
        axes[1].set_ylabel('Returns (%)', fontsize=12)
        axes[1].set_title('Portfolio Returns', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.close()
        return fig
    
    def create_interactive_candlestick(self, df, predictions=None, save_path=None):
        """
        Create interactive candlestick chart with Plotly.
        
        Args:
            df: DataFrame with OHLCV data
            predictions: Optional predictions to overlay
            save_path: Path to save HTML file (optional)
        """
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03, 
                           row_heights=[0.7, 0.3])
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='BTC Price'
        ), row=1, col=1)
        
        # Volume bars
        colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
                 for i in range(len(df))]
        
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.5
        ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title='BTC Price and Volume',
            yaxis_title='Price (USD)',
            yaxis2_title='Volume',
            xaxis2_title='Date',
            template='plotly_dark',
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive chart saved to {save_path}")
        
        return fig
