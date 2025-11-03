# Architecture Documentation

This document describes the architecture and design decisions of the BTC prediction experiment framework.

## System Overview

The framework follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│                      (experiment.py)                         │
└───────────────┬─────────────────────────────────────────────┘
                │
┌───────────────┴─────────────────────────────────────────────┐
│                     Application Layer                        │
├──────────────────┬──────────────────┬───────────────────────┤
│  Data Fetching   │  Feature Eng.    │    Visualization      │
│ (data_fetcher)   │ (feature_eng)    │  (visualization)      │
└──────────────────┴──────────────────┴───────────────────────┘
                │
┌───────────────┴─────────────────────────────────────────────┐
│                      Model Layer                             │
├──────────────────┬──────────────────┬───────────────────────┤
│  Deep Learning   │  Tree-based      │    Evaluation         │
│   (LSTM, GRU,    │  (RF, XGBoost)   │   (evaluation)        │
│   Transformer)   │                  │                       │
└──────────────────┴──────────────────┴───────────────────────┘
                │
┌───────────────┴─────────────────────────────────────────────┐
│                    Infrastructure Layer                      │
│         (TensorFlow, scikit-learn, XGBoost, etc.)           │
└─────────────────────────────────────────────────────────────┘
```

## Module Design

### 1. Data Fetcher (`data_fetcher.py`)

**Purpose**: Fetch historical BTC price data from multiple sources.

**Components**:
- `BTCDataFetcher`: Main class for data retrieval
  - `fetch_from_yfinance()`: Yahoo Finance integration
  - `fetch_from_binance()`: Binance API via CCXT
  - `fetch_data()`: Unified interface

**Design Decisions**:
- Multiple data sources for redundancy
- Standardized OHLCV format
- Error handling for network issues
- CSV caching support

**Data Flow**:
```
External API → BTCDataFetcher → Pandas DataFrame → CSV (optional)
```

### 2. Feature Engineering (`feature_engineering.py`)

**Purpose**: Transform raw OHLCV data into predictive features.

**Components**:
- `FeatureEngineer`: Feature creation pipeline
  - Price-based features (returns, ratios)
  - Technical indicators (MA, RSI, MACD, etc.)
  - Time-based features (cyclical encoding)
  - Lag features

**Design Decisions**:
- Fluent interface (method chaining)
- Modular feature groups
- Automatic NaN handling
- Configurable target horizons

**Feature Pipeline**:
```
Raw OHLCV → Price Features → Technical Indicators → Time Features → 
Lag Features → Target Creation → Clean DataFrame
```

**Feature Categories**:
1. **Price Features** (8 features)
   - Returns, log returns, price changes, ratios
   
2. **Moving Averages** (12 features for 4 windows)
   - SMA, EMA, price-to-MA ratios
   
3. **Momentum Indicators** (7 features)
   - RSI, Stochastic Oscillator, MACD (3 variants)
   
4. **Volatility Indicators** (7 features)
   - Bollinger Bands (5), ATR, rolling volatility
   
5. **Volume Indicators** (4 features)
   - OBV, volume changes, volume ratios
   
6. **Time Features** (8 features)
   - Hour, day, month (raw + cyclical encoding)
   
7. **Lag Features** (18 features for 6 lags)
   - Historical prices, volumes, returns

**Total**: 70+ features (depending on configuration)

### 3. Models (`models.py`)

**Purpose**: Implement multiple AI model architectures.

**Architecture**:
```
BaseModel (abstract)
    │
    ├── LSTMModel (deep learning)
    ├── GRUModel (deep learning)
    ├── TransformerModel (deep learning)
    ├── RandomForestModel (tree-based)
    └── XGBoostModel (tree-based)
```

**Model Specifications**:

1. **LSTM (Long Short-Term Memory)**
   - Architecture: 2-layer LSTM with dropout
   - Input: 3D sequences (samples, timesteps, features)
   - Parameters: ~50K-200K (depending on configuration)
   - Best for: Sequential patterns, long-term dependencies

2. **GRU (Gated Recurrent Unit)**
   - Architecture: 2-layer GRU with dropout
   - Input: 3D sequences
   - Parameters: ~40K-150K (fewer than LSTM)
   - Best for: Similar to LSTM but faster training

3. **Transformer**
   - Architecture: Multi-head attention + feed-forward
   - Input: 3D sequences
   - Parameters: ~100K-400K
   - Best for: Parallel processing, attention mechanisms

4. **Random Forest**
   - Architecture: Ensemble of decision trees
   - Input: 2D flat features
   - Parameters: Controlled by n_estimators and max_depth
   - Best for: Feature importance, robustness

5. **XGBoost**
   - Architecture: Gradient boosted trees
   - Input: 2D flat features
   - Parameters: Controlled by hyperparameters
   - Best for: High performance on tabular data

**Design Decisions**:
- Unified interface (fit/predict)
- Built-in scaling
- Early stopping for deep learning
- Feature importance for tree models

### 4. Evaluation (`evaluation.py`)

**Purpose**: Assess model performance and trading strategies.

**Components**:

1. **ModelEvaluator**
   - Regression metrics: RMSE, MAE, R², MAPE, directional accuracy
   - Classification metrics: Accuracy, precision, recall, F1, confusion matrix
   - Model comparison and ranking

2. **BacktestEvaluator**
   - Trading simulation
   - Portfolio tracking
   - Performance metrics: returns, Sharpe ratio, drawdown, win rate

**Design Decisions**:
- Task-agnostic evaluation
- Comprehensive metrics
- Trading strategy simulation
- Result persistence (CSV)

### 5. Visualization (`visualization.py`)

**Purpose**: Create informative charts and plots.

**Visualizations**:
1. Price history (with volume)
2. Predictions vs actual (time series + scatter)
3. Training history (loss + metrics)
4. Model comparison (bar charts)
5. Feature importance (horizontal bars)
6. Portfolio performance (value + returns)
7. Interactive candlesticks (Plotly)

**Design Decisions**:
- Matplotlib for static plots
- Plotly for interactive charts
- Consistent styling
- Automatic saving

### 6. Experiment Runner (`experiment.py`)

**Purpose**: Orchestrate the complete ML pipeline.

**Pipeline Stages**:
```
1. Data Fetching (90 days hourly data)
   ↓
2. Feature Engineering (70+ features)
   ↓
3. Data Preparation (train/val/test splits)
   ↓
4. Model Training (5 models in parallel)
   ↓
5. Model Evaluation (comprehensive metrics)
   ↓
6. Visualization (10+ plots)
   ↓
7. Results Output (CSV + images)
```

**Experiment Configuration**:
- Data source selection
- Time period and interval
- Target horizon
- Task type (regression/classification)
- Train/val/test ratios
- Model hyperparameters

## Data Flow

### End-to-End Pipeline

```
External API
    ↓
[Raw OHLCV Data]
    ↓
Feature Engineering
    ↓
[Feature Matrix (70+ features)]
    ↓
Data Splitting
    ↓
[Train (70%) | Val (15%) | Test (15%)]
    ↓
Model Training (5 models)
    ├─→ LSTM
    ├─→ GRU
    ├─→ Transformer
    ├─→ Random Forest
    └─→ XGBoost
    ↓
Predictions
    ↓
Evaluation Metrics
    ↓
[Results & Visualizations]
```

### Data Transformations

1. **Raw Data** → OHLCV DataFrame (5 columns)
2. **Feature Engineering** → Feature DataFrame (70+ columns)
3. **Sequence Creation** → 3D Arrays for RNNs (samples, timesteps, features)
4. **Scaling** → Standardized features (mean=0, std=1)
5. **Predictions** → Target values (returns or directions)

## Design Patterns

### 1. Strategy Pattern
- Multiple data sources (yfinance, binance)
- Multiple model types (LSTM, RF, XGBoost)
- Switchable evaluation metrics

### 2. Template Method Pattern
- BaseModel defines interface
- Subclasses implement specific algorithms
- Common preprocessing/postprocessing

### 3. Builder Pattern
- FeatureEngineer chains feature additions
- Fluent interface for configuration

### 4. Factory Pattern
- Model creation in experiment runner
- Configuration-based instantiation

## Performance Considerations

### Memory Optimization
- Incremental data loading
- Feature selection options
- Batch processing for predictions

### Computation Optimization
- Parallel training (n_jobs=-1 for tree models)
- GPU support (TensorFlow)
- Early stopping to prevent overtraining
- Cached intermediate results

### Scalability
- Modular design allows independent scaling
- Stateless components
- CSV-based persistence

## Error Handling

### Robust Data Fetching
- Try multiple sources
- Timeout handling
- Graceful degradation

### Model Training
- Try-catch blocks around each model
- Continue on individual failures
- Report partial results

### Validation
- Input data validation
- Shape checking
- NaN detection and handling

## Testing Strategy

### Unit Testing
- Individual module functions
- Edge cases
- Error conditions

### Integration Testing
- End-to-end pipeline
- Multiple data sources
- All model types

### Validation
- Syntax checking
- Module structure verification
- Example scripts

## Future Enhancements

### Short-term
- [ ] Model checkpointing
- [ ] Hyperparameter tuning (GridSearch)
- [ ] Cross-validation
- [ ] Additional metrics (Sortino ratio, Calmar ratio)

### Medium-term
- [ ] Real-time prediction API
- [ ] Web dashboard
- [ ] Model ensemble methods
- [ ] Additional data sources (news, social media)

### Long-term
- [ ] Reinforcement learning for trading
- [ ] Multi-cryptocurrency support
- [ ] Distributed training
- [ ] Production deployment (Docker, K8s)

## Dependencies

### Core ML/Data
- TensorFlow 2.13+ (deep learning)
- scikit-learn 1.3+ (preprocessing, metrics, RF)
- XGBoost 1.7+ (gradient boosting)
- pandas 2.0+ (data manipulation)
- numpy 1.24+ (numerical computing)

### Data & Visualization
- yfinance 0.2.28+ (Yahoo Finance API)
- ccxt 4.0+ (cryptocurrency exchange APIs)
- ta 0.11+ (technical analysis)
- matplotlib 3.7+ (plotting)
- seaborn 0.12+ (statistical visualization)
- plotly 5.17+ (interactive charts)

### Utilities
- requests 2.31+ (HTTP requests)

## Configuration

### Environment Variables
None currently used (all configuration via Python)

### Configuration Files
- `requirements.txt`: Python dependencies
- `.gitignore`: Git exclusions

### Runtime Configuration
All in `experiment.py`:
- DATA_SOURCE
- DAYS_BACK
- INTERVAL
- TARGET_HORIZON
- TASK

## Directory Structure

```
AAITI-Roots/
├── data/                      # Generated data files
│   ├── btc_raw_data.csv
│   ├── btc_features.csv
│   └── model_results.csv
├── plots/                     # Generated visualizations
│   ├── 01_price_history.png
│   ├── 02_model_comparison.png
│   ├── 03_*_predictions.png
│   ├── 04_*_training.png
│   └── 05_*_features.png
├── data_fetcher.py           # Data acquisition
├── feature_engineering.py    # Feature creation
├── models.py                 # ML models
├── evaluation.py             # Metrics & backtesting
├── visualization.py          # Plotting
├── experiment.py             # Main runner
├── example.py                # Examples
├── test_structure.py         # Validation
├── requirements.txt          # Dependencies
├── README.md                 # Main documentation
├── USAGE.md                  # Usage guide
├── ARCHITECTURE.md           # This file
└── .gitignore               # Git exclusions
```

## Conclusion

This architecture provides:
- **Modularity**: Independent, reusable components
- **Extensibility**: Easy to add new models, features, data sources
- **Maintainability**: Clear structure, documented code
- **Testability**: Isolated components, example scripts
- **Usability**: Simple API, comprehensive documentation

The design balances simplicity for users with flexibility for developers, making it suitable for both learning and research purposes.
