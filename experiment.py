"""
Main experiment runner for BTC price prediction.
Trains multiple models with different strategies and evaluates them.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import BTCDataFetcher
from feature_engineering import FeatureEngineer
from models import LSTMModel, GRUModel, TransformerModel, RandomForestModel, XGBoostModel
from evaluation import ModelEvaluator, BacktestEvaluator
from visualization import Visualizer


class BTCPredictionExperiment:
    """Run comprehensive BTC prediction experiment."""
    
    def __init__(self, data_source='yfinance', target_horizon=1, task='regression'):
        """
        Initialize experiment.
        
        Args:
            data_source: Data source ('yfinance' or 'binance')
            target_horizon: Hours ahead to predict
            task: 'regression' (predict returns) or 'classification' (predict direction)
        """
        self.data_source = data_source
        self.target_horizon = target_horizon
        self.task = task
        self.data = None
        self.features = None
        self.models = {}
        self.results = {}
        
        # Create output directories
        os.makedirs('plots', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        print("="*80)
        print("BTC SHORT-TERM PRICE PREDICTION EXPERIMENT")
        print("="*80)
        print(f"Data Source: {data_source}")
        print(f"Target Horizon: {target_horizon} hour(s)")
        print(f"Task: {task}")
        print("="*80 + "\n")
    
    def fetch_data(self, days_back=90, interval='1h'):
        """
        Fetch historical BTC data.
        
        Args:
            days_back: Number of days of historical data
            interval: Data interval
        """
        print(f"\n[1/6] Fetching data ({days_back} days)...")
        
        fetcher = BTCDataFetcher()
        
        if self.data_source == 'yfinance':
            self.data = fetcher.fetch_data(
                source='yfinance',
                start_date=(datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d'),
                interval=interval
            )
        else:
            self.data = fetcher.fetch_data(
                source='binance',
                days_back=days_back,
                timeframe=interval
            )
        
        if self.data is None or len(self.data) == 0:
            raise ValueError("Failed to fetch data")
        
        print(f"✓ Data fetched: {len(self.data)} records")
        print(f"  Date range: {self.data.index[0]} to {self.data.index[-1]}")
        
        # Save raw data
        self.data.to_csv('data/btc_raw_data.csv')
        print("  Saved to: data/btc_raw_data.csv")
        
        return self.data
    
    def engineer_features(self):
        """Create features from raw data."""
        print(f"\n[2/6] Engineering features...")
        
        engineer = FeatureEngineer(self.data)
        self.features = engineer.build_all_features(target_horizon=self.target_horizon)
        
        print(f"✓ Features engineered: {len(self.features.columns)} features")
        print(f"  Samples after cleaning: {len(self.features)}")
        
        # Save features
        self.features.to_csv('data/btc_features.csv')
        print("  Saved to: data/btc_features.csv")
        
        return self.features
    
    def prepare_data(self, train_ratio=0.7, val_ratio=0.15, sequence_length=24):
        """
        Prepare train/val/test splits.
        
        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            sequence_length: Sequence length for RNN models
        
        Returns:
            tuple: (train, val, test) data dictionaries
        """
        print(f"\n[3/6] Preparing data splits...")
        
        # Define target column
        if self.task == 'regression':
            target_col = 'target_return'
        else:
            target_col = 'target_direction'
        
        # Get feature columns (exclude targets and original OHLCV)
        exclude_cols = ['target_price', 'target_return', 'target_direction']
        feature_cols = [col for col in self.features.columns if col not in exclude_cols]
        
        # Extract features and target
        X = self.features[feature_cols].values
        y = self.features[target_col].values
        
        # Split indices
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Create sequences for RNN models
        X_seq, y_seq = self._create_sequences(X, y, sequence_length)
        
        # Split data
        train_data = {
            'X': X[:train_end],
            'y': y[:train_end],
            'X_seq': X_seq[:train_end - sequence_length],
            'y_seq': y_seq[:train_end - sequence_length],
            'indices': self.features.index[:train_end]
        }
        
        val_data = {
            'X': X[train_end:val_end],
            'y': y[train_end:val_end],
            'X_seq': X_seq[train_end - sequence_length:val_end - sequence_length],
            'y_seq': y_seq[train_end - sequence_length:val_end - sequence_length],
            'indices': self.features.index[train_end:val_end]
        }
        
        test_data = {
            'X': X[val_end:],
            'y': y[val_end:],
            'X_seq': X_seq[val_end - sequence_length:],
            'y_seq': y_seq[val_end - sequence_length:],
            'indices': self.features.index[val_end:],
            'prices': self.features['close'].iloc[val_end:].values
        }
        
        print(f"✓ Data prepared:")
        print(f"  Train: {len(train_data['y'])} samples")
        print(f"  Validation: {len(val_data['y'])} samples")
        print(f"  Test: {len(test_data['y'])} samples")
        print(f"  Features: {X.shape[1]}")
        print(f"  Sequence length: {sequence_length}")
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.feature_cols = feature_cols
        
        return train_data, val_data, test_data
    
    def _create_sequences(self, X, y, seq_length):
        """Create sequences for RNN models."""
        X_seq, y_seq = [], []
        
        for i in range(seq_length, len(X)):
            X_seq.append(X[i-seq_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_models(self):
        """Train all models."""
        print(f"\n[4/6] Training models...")
        
        results = {}
        
        # Get input shapes
        n_features = self.train_data['X'].shape[1]
        seq_length = self.train_data['X_seq'].shape[1]
        
        # Define models to train
        models_config = [
            {
                'name': 'LSTM',
                'model': LSTMModel(
                    input_shape=(seq_length, n_features),
                    units=[64, 32],
                    dropout=0.2,
                    task=self.task
                ),
                'use_sequences': True
            },
            {
                'name': 'GRU',
                'model': GRUModel(
                    input_shape=(seq_length, n_features),
                    units=[64, 32],
                    dropout=0.2,
                    task=self.task
                ),
                'use_sequences': True
            },
            {
                'name': 'Transformer',
                'model': TransformerModel(
                    input_shape=(seq_length, n_features),
                    num_heads=4,
                    ff_dim=128,
                    num_blocks=2,
                    dropout=0.1,
                    task=self.task
                ),
                'use_sequences': True
            },
            {
                'name': 'RandomForest',
                'model': RandomForestModel(
                    n_estimators=100,
                    max_depth=10,
                    task=self.task
                ),
                'use_sequences': False
            },
            {
                'name': 'XGBoost',
                'model': XGBoostModel(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    task=self.task
                ),
                'use_sequences': False
            }
        ]
        
        for config in models_config:
            name = config['name']
            model = config['model']
            use_seq = config['use_sequences']
            
            print(f"\n  Training {name}...")
            
            try:
                if use_seq:
                    # RNN models use sequences
                    history = model.fit(
                        self.train_data['X_seq'],
                        self.train_data['y_seq'],
                        self.val_data['X_seq'],
                        self.val_data['y_seq'],
                        epochs=50,
                        batch_size=32
                    )
                    
                    # Predict on test set
                    predictions = model.predict(self.test_data['X_seq'])
                else:
                    # Tree-based models use flat features
                    history = model.fit(
                        self.train_data['X'],
                        self.train_data['y'],
                        self.val_data['X'],
                        self.val_data['y']
                    )
                    
                    # Predict on test set
                    predictions = model.predict(self.test_data['X'])
                
                # Store results
                self.models[name] = model
                results[name] = {
                    'predictions': predictions,
                    'history': history
                }
                
                print(f"    ✓ {name} trained successfully")
                
            except Exception as e:
                print(f"    ✗ {name} training failed: {e}")
                continue
        
        self.results = results
        print(f"\n✓ Training complete: {len(self.models)}/{len(models_config)} models trained")
        
        return results
    
    def evaluate_models(self):
        """Evaluate all trained models."""
        print(f"\n[5/6] Evaluating models...")
        
        evaluator = ModelEvaluator(task=self.task)
        
        for name, result in self.results.items():
            predictions = result['predictions']
            y_true = self.test_data['y_seq'] if name in ['LSTM', 'GRU', 'Transformer'] else self.test_data['y']
            
            # Flatten predictions if needed
            if len(predictions.shape) > 1:
                predictions = predictions.flatten()
            
            # Evaluate
            metrics = evaluator.evaluate(y_true, predictions, model_name=name)
            print(f"  ✓ {name} evaluated")
        
        # Print results
        evaluator.print_results()
        
        # Save results
        results_df = evaluator.get_results_df()
        results_df.to_csv('data/model_results.csv')
        print(f"\n✓ Results saved to: data/model_results.csv")
        
        self.evaluator = evaluator
        return evaluator
    
    def visualize_results(self):
        """Create visualizations."""
        print(f"\n[6/6] Creating visualizations...")
        
        viz = Visualizer()
        
        # Plot price history
        viz.plot_price_history(
            self.data,
            title="BTC Historical Price Data",
            save_path='plots/01_price_history.png'
        )
        print("  ✓ Price history plot saved")
        
        # Plot model comparison
        results_df = self.evaluator.get_results_df()
        metric = 'rmse' if self.task == 'regression' else 'accuracy'
        viz.plot_model_comparison(
            results_df,
            metric=metric,
            save_path='plots/02_model_comparison.png'
        )
        print("  ✓ Model comparison plot saved")
        
        # Plot predictions for each model
        for name, result in self.results.items():
            predictions = result['predictions']
            y_true = self.test_data['y_seq'] if name in ['LSTM', 'GRU', 'Transformer'] else self.test_data['y']
            
            if len(predictions.shape) > 1:
                predictions = predictions.flatten()
            
            viz.plot_predictions_vs_actual(
                y_true,
                predictions,
                model_name=name,
                save_path=f'plots/03_{name.lower()}_predictions.png'
            )
        print(f"  ✓ Prediction plots saved for {len(self.results)} models")
        
        # Plot training history for deep learning models
        for name in ['LSTM', 'GRU', 'Transformer']:
            if name in self.results and self.results[name]['history'] is not None:
                viz.plot_training_history(
                    self.results[name]['history'],
                    model_name=name,
                    save_path=f'plots/04_{name.lower()}_training.png'
                )
        print("  ✓ Training history plots saved")
        
        # Feature importance for tree models
        for name in ['RandomForest', 'XGBoost']:
            if name in self.models:
                model = self.models[name]
                importance = model.get_feature_importance()
                viz.plot_feature_importance(
                    self.feature_cols,
                    importance,
                    top_n=20,
                    save_path=f'plots/05_{name.lower()}_features.png'
                )
        print("  ✓ Feature importance plots saved")
        
        print("\n✓ All visualizations created in 'plots/' directory")
    
    def run_experiment(self, days_back=90, interval='1h'):
        """
        Run complete experiment pipeline.
        
        Args:
            days_back: Days of historical data to fetch
            interval: Data interval
        """
        try:
            # Step 1: Fetch data
            self.fetch_data(days_back=days_back, interval=interval)
            
            # Step 2: Engineer features
            self.engineer_features()
            
            # Step 3: Prepare data
            self.prepare_data()
            
            # Step 4: Train models
            self.train_models()
            
            # Step 5: Evaluate models
            self.evaluate_models()
            
            # Step 6: Visualize results
            self.visualize_results()
            
            print("\n" + "="*80)
            print("EXPERIMENT COMPLETED SUCCESSFULLY!")
            print("="*80)
            print("\nOutputs:")
            print("  - Data files: data/")
            print("  - Plots: plots/")
            print("  - Model results: data/model_results.csv")
            print("="*80 + "\n")
            
            # Print best model
            metric = 'rmse' if self.task == 'regression' else 'accuracy'
            best_model = self.evaluator.get_best_model(metric=metric)
            print(f"Best model by {metric.upper()}: {best_model}\n")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point."""
    # Configuration
    DATA_SOURCE = 'yfinance'  # 'yfinance' or 'binance'
    DAYS_BACK = 90  # Days of historical data
    INTERVAL = '1h'  # Data interval
    TARGET_HORIZON = 1  # Hours ahead to predict
    TASK = 'regression'  # 'regression' or 'classification'
    
    # Run experiment
    experiment = BTCPredictionExperiment(
        data_source=DATA_SOURCE,
        target_horizon=TARGET_HORIZON,
        task=TASK
    )
    
    success = experiment.run_experiment(
        days_back=DAYS_BACK,
        interval=INTERVAL
    )
    
    return experiment if success else None


if __name__ == "__main__":
    experiment = main()
