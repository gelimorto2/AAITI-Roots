# Usage Guide

This guide provides detailed instructions on how to use the BTC prediction experiment framework.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration Options](#configuration-options)
3. [Module Usage](#module-usage)
4. [Advanced Usage](#advanced-usage)
5. [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/gelimorto2/AAITI-Roots.git
cd AAITI-Roots

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Default Experiment

```bash
python experiment.py
```

This will:
- Fetch 90 days of hourly BTC data
- Create 70+ technical features
- Train 5 AI models (LSTM, GRU, Transformer, Random Forest, XGBoost)
- Evaluate and compare models
- Generate visualizations

**Expected Runtime**: 10-30 minutes depending on hardware

**Output**:
- `data/` - Raw data and features CSV files
- `plots/` - All visualization charts (PNG files)
- `data/model_results.csv` - Evaluation metrics

## Configuration Options

### Basic Configuration

Edit `experiment.py` to customize:

```python
# Data source
DATA_SOURCE = 'yfinance'  # Options: 'yfinance', 'binance'

# Historical data
DAYS_BACK = 90  # Number of days (min: 7, recommended: 30-180)

# Data interval
INTERVAL = '1h'  # Options: '1m', '5m', '15m', '1h', '4h', '1d'

# Prediction target
TARGET_HORIZON = 1  # Hours ahead to predict (1-24)

# Task type
TASK = 'regression'  # Options: 'regression', 'classification'
```

### Advanced Configuration

#### Model Hyperparameters

In `experiment.py`, modify the `models_config` list:

```python
# LSTM Configuration
{
    'name': 'LSTM',
    'model': LSTMModel(
        input_shape=(seq_length, n_features),
        units=[64, 32],  # Layer sizes
        dropout=0.2,     # Dropout rate
        task=self.task
    ),
    'use_sequences': True
}

# Random Forest Configuration
{
    'name': 'RandomForest',
    'model': RandomForestModel(
        n_estimators=100,  # Number of trees
        max_depth=10,      # Max tree depth
        task=self.task
    ),
    'use_sequences': False
}
```

#### Training Parameters

```python
# In train_models() method
if use_seq:
    history = model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=50,        # Training epochs
        batch_size=32     # Batch size
    )
```

#### Data Split Ratios

```python
# In prepare_data() method
train_ratio = 0.7   # 70% training
val_ratio = 0.15    # 15% validation
# Remaining 15% for testing
```

## Module Usage

### 1. Data Fetching

```python
from data_fetcher import BTCDataFetcher
from datetime import datetime, timedelta

fetcher = BTCDataFetcher()

# Method 1: Yahoo Finance (recommended)
df = fetcher.fetch_data(
    source='yfinance',
    start_date='2024-01-01',
    end_date='2024-03-01',
    interval='1h'
)

# Method 2: Binance
df = fetcher.fetch_data(
    source='binance',
    days_back=30,
    timeframe='1h'
)

# Save/Load data
from data_fetcher import save_data, load_data
save_data(df, 'my_data.csv')
df = load_data('my_data.csv')
```

### 2. Feature Engineering

```python
from feature_engineering import FeatureEngineer

# Initialize with raw OHLCV data
engineer = FeatureEngineer(df)

# Build all features
df_features = engineer.build_all_features(target_horizon=1)

# Or add features selectively
engineer = FeatureEngineer(df)
engineer.add_price_features()          # Returns, ratios
engineer.add_moving_averages([5, 20])   # SMA, EMA
engineer.add_momentum_indicators()      # RSI, MACD, Stochastic
engineer.add_volatility_indicators()    # Bollinger Bands, ATR
engineer.add_volume_indicators()        # OBV, Volume ratios
engineer.add_time_features()            # Hour, day encoding
engineer.add_lag_features([1, 6, 24])   # Lagged values
engineer.create_target(horizon=1)       # Target variable

# Get final dataframe
df_features = engineer.df

# Get feature names
feature_names = engineer.get_feature_names(exclude_targets=True)
```

### 3. Training Models

#### LSTM Model

```python
from models import LSTMModel
import numpy as np

# Prepare sequence data
sequence_length = 24
X_seq, y_seq = [], []
for i in range(sequence_length, len(X)):
    X_seq.append(X[i-sequence_length:i])
    y_seq.append(y[i])
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Create and train model
model = LSTMModel(
    input_shape=(sequence_length, n_features),
    units=[64, 32],
    dropout=0.2,
    task='regression'
)

history = model.fit(
    X_train_seq, y_train_seq,
    X_val_seq, y_val_seq,
    epochs=50,
    batch_size=32
)

# Make predictions
predictions = model.predict(X_test_seq)
```

#### Random Forest Model

```python
from models import RandomForestModel

# Create and train model
model = RandomForestModel(
    n_estimators=100,
    max_depth=10,
    task='regression'
)

model.fit(X_train, y_train, X_val, y_val)

# Make predictions
predictions = model.predict(X_test)

# Get feature importance
importance = model.get_feature_importance()
```

#### XGBoost Model

```python
from models import XGBoostModel

model = XGBoostModel(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    task='regression'
)

model.fit(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test)
importance = model.get_feature_importance()
```

### 4. Evaluation

```python
from evaluation import ModelEvaluator

# Regression evaluation
evaluator = ModelEvaluator(task='regression')

metrics = evaluator.evaluate(
    y_true=y_test,
    y_pred=predictions,
    model_name='LSTM'
)

print(f"RMSE: {metrics['rmse']:.6f}")
print(f"MAE: {metrics['mae']:.6f}")
print(f"R²: {metrics['r2']:.6f}")
print(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%")

# Compare multiple models
evaluator.evaluate(y_test, pred_lstm, 'LSTM')
evaluator.evaluate(y_test, pred_rf, 'RandomForest')
evaluator.evaluate(y_test, pred_xgb, 'XGBoost')

evaluator.print_results()
best_model = evaluator.get_best_model(metric='rmse')

# Classification evaluation
evaluator_clf = ModelEvaluator(task='classification')
metrics = evaluator_clf.evaluate(y_true, y_pred, 'Model')
print(f"Accuracy: {metrics['accuracy']:.2f}%")
print(f"F1 Score: {metrics['f1_score']:.2f}%")
```

### 5. Visualization

```python
from visualization import Visualizer

viz = Visualizer()

# Price history
viz.plot_price_history(
    df,
    title="BTC Price History",
    save_path='price_history.png'
)

# Predictions vs Actual
viz.plot_predictions_vs_actual(
    y_true,
    y_pred,
    model_name='LSTM',
    save_path='predictions.png'
)

# Model comparison
results_df = evaluator.get_results_df()
viz.plot_model_comparison(
    results_df,
    metric='rmse',
    save_path='comparison.png'
)

# Training history (for deep learning models)
viz.plot_training_history(
    history,
    model_name='LSTM',
    save_path='training.png'
)

# Feature importance
viz.plot_feature_importance(
    feature_names,
    importance_scores,
    top_n=20,
    save_path='features.png'
)

# Interactive candlestick chart
fig = viz.create_interactive_candlestick(
    df,
    save_path='candlestick.html'
)
```

## Advanced Usage

### Custom Feature Sets

```python
# Minimal features
engineer = FeatureEngineer(df)
engineer.add_price_features()
engineer.add_moving_averages([5, 20])
engineer.create_target(horizon=1)
df_minimal = engineer.df

# Custom indicators
engineer = FeatureEngineer(df)
engineer.add_price_features()
engineer.add_momentum_indicators()  # RSI, MACD
engineer.add_volatility_indicators()  # Bollinger Bands
engineer.create_target(horizon=1)
df_custom = engineer.df
```

### Multiple Prediction Horizons

```python
# Predict 1 hour ahead
experiment_1h = BTCPredictionExperiment(
    target_horizon=1,
    task='regression'
)
experiment_1h.run_experiment()

# Predict 4 hours ahead
experiment_4h = BTCPredictionExperiment(
    target_horizon=4,
    task='regression'
)
experiment_4h.run_experiment()

# Predict 24 hours ahead
experiment_24h = BTCPredictionExperiment(
    target_horizon=24,
    task='regression'
)
experiment_24h.run_experiment()
```

### Backtesting Trading Strategies

```python
from evaluation import BacktestEvaluator

# Simulate trading based on predictions
backtester = BacktestEvaluator(initial_capital=10000)

metrics, portfolio_values, trades = backtester.simulate_trading(
    predictions=predictions,
    actual_prices=actual_prices,
    threshold=0.0  # Trade when predicted return > 0
)

print(f"Total Return: {metrics['total_return']:.2f}%")
print(f"Win Rate: {metrics['win_rate']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")

# Visualize portfolio
viz.plot_portfolio_performance(
    portfolio_values,
    trades=trades,
    initial_capital=10000,
    save_path='portfolio.png'
)
```

### Ensemble Methods

```python
# Average predictions from multiple models
predictions_ensemble = (
    predictions_lstm * 0.3 +
    predictions_rf * 0.3 +
    predictions_xgb * 0.4
)

# Weighted by performance
weights = {
    'LSTM': 0.25,
    'RandomForest': 0.35,
    'XGBoost': 0.40
}

predictions_weighted = sum(
    results[name]['predictions'] * weight
    for name, weight in weights.items()
)
```

## Troubleshooting

### Common Issues

#### 1. No Data Retrieved

```python
# If yfinance fails, try Binance
fetcher.fetch_data(source='binance', days_back=30, timeframe='1h')

# Or check internet connection
import requests
response = requests.get('https://www.google.com', timeout=5)
print(response.status_code)  # Should be 200
```

#### 2. Memory Issues

```python
# Reduce data size
DAYS_BACK = 30  # Instead of 90
sequence_length = 12  # Instead of 24

# Or use smaller models
units=[32, 16]  # Instead of [64, 32]
n_estimators=50  # Instead of 100
```

#### 3. Training Too Slow

```python
# Reduce epochs
epochs=20  # Instead of 50

# Use smaller batch size
batch_size=64  # Instead of 32

# Train fewer models
# Comment out models you don't need in experiment.py
```

#### 4. NaN Values in Features

```python
# Check data quality
print(df.isnull().sum())

# More aggressive dropna
df_features.dropna(inplace=True)

# Or fill NaN values
df_features.fillna(method='ffill', inplace=True)
```

### Performance Tips

1. **GPU Acceleration**: Install `tensorflow-gpu` for faster deep learning
2. **Parallel Processing**: Already enabled for Random Forest and XGBoost
3. **Data Caching**: Save features to avoid recomputation
4. **Incremental Learning**: Train on new data without retraining from scratch

### Getting Help

- Check the main README.md for overview
- Review the code comments in each module
- Open an issue on GitHub for bugs or questions

## Example Workflows

### Workflow 1: Quick Analysis

```bash
# 1. Run with defaults
python experiment.py

# 2. Check results
cat data/model_results.csv

# 3. View plots
ls plots/
```

### Workflow 2: Custom Analysis

```python
# custom_experiment.py
from experiment import BTCPredictionExperiment

# Short-term prediction (1 hour)
exp = BTCPredictionExperiment(
    data_source='yfinance',
    target_horizon=1,
    task='classification'  # Predict up/down
)
exp.run_experiment(days_back=30, interval='1h')

# Review best model
print(f"Best model: {exp.evaluator.get_best_model('accuracy')}")
```

### Workflow 3: Feature Analysis

```python
from data_fetcher import BTCDataFetcher
from feature_engineering import FeatureEngineer
from models import RandomForestModel
from visualization import Visualizer

# Get data
fetcher = BTCDataFetcher()
df = fetcher.fetch_data(source='yfinance', ...)

# Build features
engineer = FeatureEngineer(df)
df_features = engineer.build_all_features()

# Train model
X = df_features[feature_cols].values
y = df_features['target_return'].values
model = RandomForestModel()
model.fit(X[:len(X)//2], y[:len(y)//2])

# Analyze feature importance
importance = model.get_feature_importance()
viz = Visualizer()
viz.plot_feature_importance(
    feature_cols,
    importance,
    top_n=30,
    save_path='top_features.png'
)
```

---

For more information, see the main [README.md](README.md).
