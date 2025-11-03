"""
Evaluation module for model performance assessment.
Includes metrics calculation and comparison utilities.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


class ModelEvaluator:
    """Evaluate model predictions."""
    
    def __init__(self, task='regression'):
        """
        Initialize evaluator.
        
        Args:
            task: 'regression' or 'classification'
        """
        self.task = task
        self.results = {}
    
    def evaluate_regression(self, y_true, y_pred, model_name='Model'):
        """
        Evaluate regression predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
        
        Returns:
            dict: Dictionary of metrics
        """
        # Flatten arrays if needed
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Mean absolute percentage error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Directional accuracy (for returns)
        if len(y_true) > 1:
            true_direction = np.sign(y_true)
            pred_direction = np.sign(y_pred)
            directional_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            directional_accuracy = 0
        
        metrics = {
            'model': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def evaluate_classification(self, y_true, y_pred, model_name='Model'):
        """
        Evaluate classification predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
        
        Returns:
            dict: Dictionary of metrics
        """
        # Flatten and convert to binary
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Convert probabilities to binary if needed
        if y_pred.dtype == float and np.max(y_pred) <= 1.0:
            y_pred = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        metrics = {
            'model': model_name,
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def evaluate(self, y_true, y_pred, model_name='Model'):
        """Evaluate based on task type."""
        if self.task == 'regression':
            return self.evaluate_regression(y_true, y_pred, model_name)
        else:
            return self.evaluate_classification(y_true, y_pred, model_name)
    
    def get_results_df(self):
        """Get results as DataFrame."""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results).T
    
    def print_results(self):
        """Print formatted results."""
        if not self.results:
            print("No results to display")
            return
        
        df = self.get_results_df()
        print("\n" + "="*80)
        print("MODEL EVALUATION RESULTS")
        print("="*80)
        
        if self.task == 'regression':
            print("\nRegression Metrics:")
            print("-" * 80)
            for idx, row in df.iterrows():
                print(f"\n{row['model']}:")
                print(f"  RMSE:                  {row['rmse']:.6f}")
                print(f"  MAE:                   {row['mae']:.6f}")
                print(f"  R²:                    {row['r2']:.6f}")
                print(f"  MAPE:                  {row['mape']:.2f}%")
                print(f"  Directional Accuracy:  {row['directional_accuracy']:.2f}%")
        else:
            print("\nClassification Metrics:")
            print("-" * 80)
            for idx, row in df.iterrows():
                print(f"\n{row['model']}:")
                print(f"  Accuracy:    {row['accuracy']:.2f}%")
                print(f"  Precision:   {row['precision']:.2f}%")
                print(f"  Recall:      {row['recall']:.2f}%")
                print(f"  F1 Score:    {row['f1_score']:.2f}%")
                print(f"  Confusion Matrix:")
                print(f"    TN: {row['true_negatives']:<6} FP: {row['false_positives']}")
                print(f"    FN: {row['false_negatives']:<6} TP: {row['true_positives']}")
        
        print("\n" + "="*80)
    
    def compare_models(self, metric='rmse'):
        """
        Compare models by a specific metric.
        
        Args:
            metric: Metric to compare by
        
        Returns:
            pd.DataFrame: Sorted results by metric
        """
        if not self.results:
            return pd.DataFrame()
        
        df = self.get_results_df()
        
        if metric not in df.columns:
            print(f"Metric '{metric}' not found. Available metrics: {df.columns.tolist()}")
            return df
        
        # Sort by metric (ascending for error metrics, descending for accuracy metrics)
        ascending = metric in ['mse', 'rmse', 'mae', 'mape']
        df_sorted = df.sort_values(by=metric, ascending=ascending)
        
        return df_sorted
    
    def get_best_model(self, metric='rmse'):
        """
        Get the best performing model.
        
        Args:
            metric: Metric to evaluate by
        
        Returns:
            str: Name of best model
        """
        df_sorted = self.compare_models(metric)
        if df_sorted.empty:
            return None
        
        return df_sorted.index[0]


class BacktestEvaluator:
    """Evaluate trading strategies based on predictions."""
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
    
    def simulate_trading(self, predictions, actual_prices, threshold=0.0):
        """
        Simulate trading strategy based on predictions.
        
        Args:
            predictions: Predicted returns or directions
            actual_prices: Actual price series
            threshold: Threshold for trading decision
        
        Returns:
            dict: Trading metrics
        """
        capital = self.initial_capital
        position = 0  # 0: no position, 1: long
        trades = []
        portfolio_values = [capital]
        
        for i in range(len(predictions)):
            # Trading signal
            if predictions[i] > threshold and position == 0:
                # Buy
                position = 1
                entry_price = actual_prices[i]
                trades.append({
                    'type': 'BUY',
                    'price': entry_price,
                    'time': i
                })
            elif predictions[i] <= threshold and position == 1:
                # Sell
                position = 0
                exit_price = actual_prices[i]
                returns = (exit_price - entry_price) / entry_price
                capital *= (1 + returns)
                trades.append({
                    'type': 'SELL',
                    'price': exit_price,
                    'time': i,
                    'return': returns
                })
            
            # Update portfolio value
            if position == 1:
                portfolio_values.append(capital * (actual_prices[i] / entry_price))
            else:
                portfolio_values.append(capital)
        
        # Calculate metrics
        total_return = (capital - self.initial_capital) / self.initial_capital * 100
        num_trades = len([t for t in trades if t['type'] == 'SELL'])
        
        winning_trades = [t for t in trades if t['type'] == 'SELL' and t['return'] > 0]
        win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0
        
        # Sharpe ratio (simplified)
        returns = pd.Series(portfolio_values).pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Maximum drawdown
        cumulative = pd.Series(portfolio_values).cummax()
        drawdown = (pd.Series(portfolio_values) - cumulative) / cumulative * 100
        max_drawdown = drawdown.min()
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        return metrics, portfolio_values, trades
