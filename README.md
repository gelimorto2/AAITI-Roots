# BTC Short-Term Price Prediction Experiment

A comprehensive machine learning experiment to train and evaluate multiple AI models for predicting Bitcoin (BTC) price movements on short-term (hourly) timeframes.

## Overview

This project implements a complete experimental framework for training and comparing different AI models and strategies to predict BTC price movements. It includes:

- **Multiple AI Model Types**: LSTM, GRU, Transformer, Random Forest, XGBoost
- **Rich Feature Engineering**: Technical indicators, momentum, volatility, volume, and time features
- **Comprehensive Evaluation**: Regression and classification metrics, backtesting capabilities
- **Advanced Visualization**: Price charts, predictions, model comparisons, feature importance

## Features

### Data Fetching
- Multiple data sources (Yahoo Finance, Binance via CCXT)
- Configurable timeframes (1m, 5m, 15m, 1h, 4h, 1d)
- Historical data with OHLCV (Open, High, Low, Close, Volume)

### Feature Engineering
- **Price Features**: Returns, log returns, price changes, ratios
- **Moving Averages**: SMA, EMA with multiple windows (5, 10, 20, 50)
- **Momentum Indicators**: RSI, Stochastic Oscillator, MACD
- **Volatility Indicators**: Bollinger Bands, ATR, historical volatility
- **Volume Indicators**: OBV, volume changes, volume ratios
- **Time Features**: Cyclical encoding for hour, day, week, month
- **Lag Features**: Historical values at different time lags

### AI Models

1. **LSTM (Long Short-Term Memory)**
   - Recurrent neural network for time series
   - Captures long-term dependencies
   - Multiple layers with dropout

2. **GRU (Gated Recurrent Unit)**
   - Simplified RNN architecture
   - Faster training than LSTM
   - Effective for sequential patterns

3. **Transformer**
   - Multi-head attention mechanism
   - Parallel processing of sequences
   - State-of-the-art architecture

4. **Random Forest**
   - Ensemble of decision trees
   - Robust to overfitting
   - Feature importance analysis

5. **XGBoost**
   - Gradient boosting framework
   - High performance on tabular data
   - Advanced regularization

### Evaluation Metrics

**Regression Task** (Predicting Returns):
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)
- Directional Accuracy

**Classification Task** (Predicting Direction):
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Installation

1. Clone the repository:
```bash
git clone https://github.com/gelimorto2/AAITI-Roots.git
cd AAITI-Roots
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete experiment with default settings:

```bash
python experiment.py
```

This will:
1. Fetch 90 days of hourly BTC data from Yahoo Finance
2. Engineer features with technical indicators
3. Train 5 different AI models (LSTM, GRU, Transformer, Random Forest, XGBoost)
4. Evaluate models on test data
5. Generate visualizations and save results

### Custom Configuration

Edit the configuration in `experiment.py`:

```python
# Configuration
DATA_SOURCE = 'yfinance'  # 'yfinance' or 'binance'
DAYS_BACK = 90  # Days of historical data
INTERVAL = '1h'  # Data interval: '1m', '5m', '15m', '1h', '4h', '1d'
TARGET_HORIZON = 1  # Hours ahead to predict
TASK = 'regression'  # 'regression' or 'classification'
```

### Using Individual Modules

#### 1. Data Fetching

```python
from data_fetcher import BTCDataFetcher
from datetime import datetime, timedelta

fetcher = BTCDataFetcher()

# Fetch from Yahoo Finance
df = fetcher.fetch_data(
    source='yfinance',
    start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
    end_date=datetime.now().strftime('%Y-%m-%d'),
    interval='1h'
)

# Or fetch from Binance
df = fetcher.fetch_data(
    source='binance',
    days_back=30,
    timeframe='1h'
)
```

#### 2. Feature Engineering

```python
from feature_engineering import FeatureEngineer

engineer = FeatureEngineer(df)
df_features = engineer.build_all_features(target_horizon=1)

# Or add features individually
engineer = FeatureEngineer(df)
engineer.add_price_features()
engineer.add_moving_averages(windows=[5, 10, 20])
engineer.add_momentum_indicators()
engineer.add_volatility_indicators()
engineer.create_target(horizon=1)
```

#### 3. Training Models

```python
from models import LSTMModel, RandomForestModel

# LSTM Model
lstm = LSTMModel(
    input_shape=(24, 50),  # (sequence_length, n_features)
    units=[64, 32],
    dropout=0.2,
    task='regression'
)
lstm.fit(X_train, y_train, X_val, y_val, epochs=50)
predictions = lstm.predict(X_test)

# Random Forest Model
rf = RandomForestModel(n_estimators=100, max_depth=10, task='regression')
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
```

#### 4. Evaluation

```python
from evaluation import ModelEvaluator

evaluator = ModelEvaluator(task='regression')
metrics = evaluator.evaluate(y_true, y_pred, model_name='LSTM')
evaluator.print_results()

# Compare models
best_model = evaluator.get_best_model(metric='rmse')
```

#### 5. Visualization

```python
from visualization import Visualizer

viz = Visualizer()

# Plot price history
viz.plot_price_history(df, save_path='price_history.png')

# Plot predictions vs actual
viz.plot_predictions_vs_actual(y_true, y_pred, model_name='LSTM', 
                                save_path='predictions.png')

# Compare models
viz.plot_model_comparison(results_df, metric='rmse', 
                          save_path='comparison.png')

# Feature importance
viz.plot_feature_importance(feature_names, importance_scores,
                            save_path='features.png')
```

## Output Structure

After running the experiment, the following outputs are generated:

```
AAITI-Roots/
├── data/
│   ├── btc_raw_data.csv          # Raw OHLCV data
│   ├── btc_features.csv          # Engineered features
│   └── model_results.csv         # Evaluation metrics
├── plots/
│   ├── 01_price_history.png      # Historical price chart
│   ├── 02_model_comparison.png   # Model performance comparison
│   ├── 03_*_predictions.png      # Predictions vs actual for each model
│   ├── 04_*_training.png         # Training history for deep learning models
│   └── 05_*_features.png         # Feature importance for tree models
└── experiment.py
```

## Experiment Results

The experiment evaluates models on:

1. **Prediction Accuracy**: How well models predict future prices/returns
2. **Directional Accuracy**: Percentage of correct up/down predictions
3. **Model Comparison**: Ranking models by performance metrics
4. **Feature Importance**: Which features contribute most to predictions

Example output:

```
================================================================================
MODEL EVALUATION RESULTS
================================================================================

Regression Metrics:
--------------------------------------------------------------------------------

LSTM:
  RMSE:                  0.002345
  MAE:                   0.001876
  R²:                    0.453210
  MAPE:                  12.34%
  Directional Accuracy:  56.78%

XGBoost:
  RMSE:                  0.002123
  MAE:                   0.001654
  R²:                    0.487652
  MAPE:                  11.23%
  Directional Accuracy:  58.92%

Best model by RMSE: XGBoost
```

## Strategies Implemented

The experiment implements multiple prediction strategies:

1. **Time Horizon Strategies**: 1-hour, 4-hour, 1-day predictions
2. **Feature Set Strategies**: 
   - Minimal features (price only)
   - Technical indicators
   - Full feature set
3. **Model Strategies**: 
   - Deep learning (LSTM, GRU, Transformer)
   - Tree-based (Random Forest, XGBoost)
4. **Target Strategies**:
   - Regression: Predict exact returns
   - Classification: Predict price direction

## Key Findings

The experiment typically shows:

- Tree-based models (XGBoost, Random Forest) perform well on tabular features
- Deep learning models capture sequential patterns effectively
- Feature engineering significantly improves prediction accuracy
- Short-term predictions (1-hour) are more accurate than longer horizons
- Directional accuracy is often more reliable than exact price prediction

## Limitations

- Past performance doesn't guarantee future results
- Market conditions change (concept drift)
- High-frequency trading requires more sophisticated infrastructure
- Transaction costs and slippage not fully modeled
- Not financial advice - for educational purposes only

## Future Enhancements

Potential improvements:

- [ ] Ensemble methods combining multiple models
- [ ] Real-time data streaming and predictions
- [ ] Additional data sources (news sentiment, social media)
- [ ] More advanced architectures (LSTM with attention, etc.)
- [ ] Reinforcement learning for trading strategies
- [ ] Cross-cryptocurrency analysis
- [ ] Market regime detection
- [ ] Risk management integration

## Requirements

- Python 3.7+
- TensorFlow 2.13.0
- scikit-learn 1.3.0
- XGBoost 1.7.6
- pandas, numpy, matplotlib, seaborn
- yfinance, ccxt
- ta (technical analysis library)

See `requirements.txt` for complete dependencies.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Cryptocurrency trading carries significant risk. Always do your own research and consult with financial professionals before making investment decisions.

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an experimental framework for learning and research. Use at your own risk.