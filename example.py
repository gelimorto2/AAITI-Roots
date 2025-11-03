"""
Example script demonstrating key features of the BTC prediction framework.
This script can be run without fetching real data by using synthetic data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def create_synthetic_btc_data(n_samples=1000, start_price=50000):
    """
    Create synthetic BTC-like data for testing.
    
    Args:
        n_samples: Number of hourly samples
        start_price: Starting price
    
    Returns:
        pd.DataFrame: Synthetic OHLCV data
    """
    print("Creating synthetic BTC data...")
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(hours=n_samples)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Generate price with trend and noise
    trend = np.linspace(0, 0.2, n_samples)  # 20% upward trend
    noise = np.random.randn(n_samples) * 0.02  # 2% random noise
    returns = trend + noise
    
    # Generate prices
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close prices
    opens = prices * (1 + np.random.randn(n_samples) * 0.005)
    highs = np.maximum(prices, opens) * (1 + np.abs(np.random.randn(n_samples)) * 0.01)
    lows = np.minimum(prices, opens) * (1 - np.abs(np.random.randn(n_samples)) * 0.01)
    closes = prices
    volumes = np.random.uniform(100, 1000, n_samples) * 1e6  # Random volume
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=timestamps)
    
    print(f"✓ Created {n_samples} samples from {df.index[0]} to {df.index[-1]}")
    return df


def example_1_data_and_features():
    """Example 1: Data fetching and feature engineering."""
    print("\n" + "="*80)
    print("EXAMPLE 1: DATA AND FEATURE ENGINEERING")
    print("="*80 + "\n")
    
    # Create synthetic data
    df = create_synthetic_btc_data(n_samples=500)
    
    print("\nRaw data sample:")
    print(df.head())
    print(f"\nData shape: {df.shape}")
    
    # Feature engineering
    from feature_engineering import FeatureEngineer
    
    engineer = FeatureEngineer(df)
    df_features = engineer.build_all_features(target_horizon=1)
    
    print(f"\nFeatures created: {len(df_features.columns)} features")
    print(f"Sample count after cleaning: {len(df_features)}")
    
    print("\nFeature categories:")
    feature_names = engineer.get_feature_names(exclude_targets=True)
    print(f"  - Price features: {len([f for f in feature_names if 'price' in f or 'returns' in f])}")
    print(f"  - Moving averages: {len([f for f in feature_names if 'sma' in f or 'ema' in f])}")
    print(f"  - Momentum indicators: {len([f for f in feature_names if 'rsi' in f or 'macd' in f or 'stoch' in f])}")
    print(f"  - Volatility indicators: {len([f for f in feature_names if 'bb' in f or 'atr' in f or 'volatility' in f])}")
    print(f"  - Volume indicators: {len([f for f in feature_names if 'volume' in f or 'obv' in f])}")
    print(f"  - Time features: {len([f for f in feature_names if 'hour' in f or 'day' in f or 'month' in f])}")
    print(f"  - Lag features: {len([f for f in feature_names if 'lag' in f])}")
    
    return df, df_features, feature_names


def example_2_model_training():
    """Example 2: Training different models."""
    print("\n" + "="*80)
    print("EXAMPLE 2: MODEL TRAINING")
    print("="*80 + "\n")
    
    # Get data and features
    df = create_synthetic_btc_data(n_samples=500)
    
    from feature_engineering import FeatureEngineer
    engineer = FeatureEngineer(df)
    df_features = engineer.build_all_features(target_horizon=1)
    
    # Prepare data
    feature_cols = engineer.get_feature_names(exclude_targets=True)
    X = df_features[feature_cols].values
    y = df_features['target_return'].values
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"Data splits:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Features: {X.shape[1]}")
    
    # Train Random Forest (fastest for demo)
    print("\nTraining Random Forest model...")
    from models import RandomForestModel
    
    rf_model = RandomForestModel(n_estimators=50, max_depth=8, task='regression')
    rf_model.fit(X_train, y_train, X_val, y_val)
    
    print("✓ Random Forest trained")
    
    # Make predictions
    predictions = rf_model.predict(X_test)
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    print(f"Sample actual values: {y_test[:5]}")
    
    return rf_model, predictions, y_test, feature_cols


def example_3_evaluation():
    """Example 3: Model evaluation."""
    print("\n" + "="*80)
    print("EXAMPLE 3: MODEL EVALUATION")
    print("="*80 + "\n")
    
    # Train model and get predictions
    _, predictions, y_test, _ = example_2_model_training()
    
    # Evaluate
    from evaluation import ModelEvaluator
    
    evaluator = ModelEvaluator(task='regression')
    metrics = evaluator.evaluate(y_test, predictions, model_name='RandomForest')
    
    print("\nEvaluation Metrics:")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  R²: {metrics['r2']:.6f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
    
    return metrics


def example_4_visualization():
    """Example 4: Creating visualizations."""
    print("\n" + "="*80)
    print("EXAMPLE 4: VISUALIZATION")
    print("="*80 + "\n")
    
    import os
    os.makedirs('example_plots', exist_ok=True)
    
    # Get data and predictions
    df = create_synthetic_btc_data(n_samples=500)
    model, predictions, y_test, feature_cols = example_2_model_training()
    
    from visualization import Visualizer
    viz = Visualizer()
    
    # 1. Price history
    print("Creating price history plot...")
    viz.plot_price_history(
        df,
        title="Synthetic BTC Price Data",
        save_path='example_plots/price_history.png'
    )
    print("✓ Saved to: example_plots/price_history.png")
    
    # 2. Predictions vs Actual
    print("\nCreating predictions plot...")
    viz.plot_predictions_vs_actual(
        y_test,
        predictions,
        model_name='RandomForest',
        save_path='example_plots/predictions.png'
    )
    print("✓ Saved to: example_plots/predictions.png")
    
    # 3. Feature importance
    print("\nCreating feature importance plot...")
    importance = model.get_feature_importance()
    viz.plot_feature_importance(
        feature_cols,
        importance,
        top_n=20,
        save_path='example_plots/feature_importance.png'
    )
    print("✓ Saved to: example_plots/feature_importance.png")
    
    print("\n✓ All plots saved in 'example_plots/' directory")


def example_5_comparison():
    """Example 5: Comparing multiple models."""
    print("\n" + "="*80)
    print("EXAMPLE 5: COMPARING MULTIPLE MODELS")
    print("="*80 + "\n")
    
    # Get data and features
    df = create_synthetic_btc_data(n_samples=500)
    
    from feature_engineering import FeatureEngineer
    engineer = FeatureEngineer(df)
    df_features = engineer.build_all_features(target_horizon=1)
    
    # Prepare data
    feature_cols = engineer.get_feature_names(exclude_targets=True)
    X = df_features[feature_cols].values
    y = df_features['target_return'].values
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    # Train multiple models
    from models import RandomForestModel, XGBoostModel
    from evaluation import ModelEvaluator
    
    evaluator = ModelEvaluator(task='regression')
    
    print("Training models...")
    
    # Random Forest
    print("\n  Training Random Forest...")
    rf_model = RandomForestModel(n_estimators=50, max_depth=8, task='regression')
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    evaluator.evaluate(y_test, rf_pred, 'RandomForest')
    print("  ✓ Random Forest complete")
    
    # XGBoost
    print("\n  Training XGBoost...")
    xgb_model = XGBoostModel(n_estimators=50, max_depth=6, task='regression')
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    evaluator.evaluate(y_test, xgb_pred, 'XGBoost')
    print("  ✓ XGBoost complete")
    
    # Print comparison
    print("\n" + "-"*80)
    evaluator.print_results()
    
    # Get best model
    best = evaluator.get_best_model(metric='rmse')
    print(f"\n✓ Best model by RMSE: {best}")
    
    # Visualize comparison
    import os
    os.makedirs('example_plots', exist_ok=True)
    
    from visualization import Visualizer
    viz = Visualizer()
    
    results_df = evaluator.get_results_df()
    viz.plot_model_comparison(
        results_df,
        metric='rmse',
        save_path='example_plots/model_comparison.png'
    )
    print("✓ Comparison plot saved to: example_plots/model_comparison.png")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("BTC PREDICTION FRAMEWORK - EXAMPLES")
    print("="*80)
    print("\nThese examples use synthetic data for demonstration.")
    print("To use real BTC data, run 'python experiment.py'\n")
    
    try:
        # Example 1: Data and Features
        example_1_data_and_features()
        
        # Example 3: Evaluation (includes training from example 2)
        example_3_evaluation()
        
        # Example 4: Visualization
        example_4_visualization()
        
        # Example 5: Model Comparison
        example_5_comparison()
        
        print("\n" + "="*80)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nNext steps:")
        print("  1. Review plots in 'example_plots/' directory")
        print("  2. Explore individual modules in Python REPL")
        print("  3. Run full experiment: python experiment.py")
        print("  4. See USAGE.md for more examples")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
