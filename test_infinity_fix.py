"""
Test to verify that infinity values in features are properly handled.
This test reproduces the issue from the problem statement and verifies the fix.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from feature_engineering import FeatureEngineer
from models import LSTMModel, GRUModel, TransformerModel, RandomForestModel, XGBoostModel


def create_data_with_infinity_edge_cases(n_samples=500):
    """
    Create synthetic data with edge cases that cause infinity values.
    
    Edge cases that cause infinity:
    - Zero volumes (causes inf in pct_change)
    - Very small denominators in ratio calculations
    - Identical consecutive prices
    """
    timestamps = [datetime.now() - timedelta(hours=n_samples-i) for i in range(n_samples)]
    
    # Generate mostly normal data
    prices = 50000 + np.random.randn(n_samples) * 500
    volumes = np.random.uniform(100, 1000, n_samples) * 1e6
    
    # Add edge cases that would cause infinity
    volumes[50] = 0  # Zero volume -> volume.pct_change() creates inf
    volumes[100] = 0
    prices[100] = prices[99]  # Identical prices
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n_samples) * 0.001),
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': volumes
    }, index=timestamps)
    
    return df


def test_no_infinity_in_features():
    """Test that feature engineering produces no infinity values."""
    print("\n" + "="*80)
    print("TEST: No Infinity Values in Features")
    print("="*80)
    
    # Create problematic data
    df = create_data_with_infinity_edge_cases(n_samples=500)
    print(f"\nCreated test data: {df.shape}")
    
    # Engineer features
    engineer = FeatureEngineer(df)
    df_features = engineer.build_all_features(target_horizon=1)
    
    print(f"Features created: {df_features.shape}")
    
    # Check for infinity values
    numeric_cols = df_features.select_dtypes(include=[np.number])
    has_inf = np.isinf(numeric_cols.values).any()
    inf_count = np.isinf(numeric_cols.values).sum()
    
    print(f"\nInfinity values: {inf_count}")
    
    if has_inf:
        print("\n✗ TEST FAILED: Features still contain infinity values")
        # List columns with infinity
        for col in numeric_cols.columns:
            if np.isinf(numeric_cols[col]).any():
                print(f"  - {col}: {np.isinf(numeric_cols[col]).sum()} inf values")
        return False
    else:
        print("✓ TEST PASSED: No infinity values in features")
        return True


def test_models_train_without_infinity_errors():
    """Test that all models can train without infinity errors."""
    print("\n" + "="*80)
    print("TEST: Models Train Without Infinity Errors")
    print("="*80)
    
    # Create and prepare data
    df = create_data_with_infinity_edge_cases(n_samples=500)
    engineer = FeatureEngineer(df)
    df_features = engineer.build_all_features(target_horizon=1)
    
    # Prepare training data
    feature_cols = engineer.get_feature_names(exclude_targets=True)
    X = df_features[feature_cols].values
    y = df_features['target_return'].values
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    
    # Create sequences for RNN models
    seq_length = 24
    X_train_seq, y_train_seq = [], []
    for i in range(seq_length, len(X_train)):
        X_train_seq.append(X_train[i-seq_length:i])
        y_train_seq.append(y_train[i])
    X_train_seq = np.array(X_train_seq)
    y_train_seq = np.array(y_train_seq)
    
    X_val_seq, y_val_seq = [], []
    for i in range(seq_length, len(X_val)):
        X_val_seq.append(X_val[i-seq_length:i])
        y_val_seq.append(y_val[i])
    X_val_seq = np.array(X_val_seq)
    y_val_seq = np.array(y_val_seq)
    
    print(f"\nTraining data:")
    print(f"  Train: {len(y_train)} samples ({len(y_train_seq)} sequences)")
    print(f"  Val: {len(y_val)} samples ({len(y_val_seq)} sequences)")
    
    # Test each model
    models_config = [
        ('LSTM', lambda: LSTMModel(
            input_shape=(seq_length, X.shape[1]), 
            units=[32, 16], 
            task='regression'
        ), True),
        ('GRU', lambda: GRUModel(
            input_shape=(seq_length, X.shape[1]), 
            units=[32, 16], 
            task='regression'
        ), True),
        ('Transformer', lambda: TransformerModel(
            input_shape=(seq_length, X.shape[1]), 
            num_heads=2, 
            ff_dim=64, 
            num_blocks=1, 
            task='regression'
        ), True),
        ('RandomForest', lambda: RandomForestModel(
            n_estimators=50, 
            max_depth=8, 
            task='regression'
        ), False),
        ('XGBoost', lambda: XGBoostModel(
            n_estimators=50, 
            max_depth=6, 
            task='regression'
        ), False),
    ]
    
    models_trained = 0
    models_failed = 0
    failed_models = []
    
    print("\nTraining models:")
    for name, model_fn, use_seq in models_config:
        print(f"  Training {name}...", end=' ')
        try:
            model = model_fn()
            if use_seq:
                model.fit(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=5, batch_size=16)
            else:
                model.fit(X_train, y_train, X_val, y_val)
            print("✓")
            models_trained += 1
        except Exception as e:
            print(f"✗ ({e})")
            models_failed += 1
            failed_models.append(name)
    
    print(f"\nResults: {models_trained}/5 models trained successfully")
    
    if models_failed > 0:
        print(f"✗ TEST FAILED: {models_failed} models failed to train")
        print(f"  Failed models: {', '.join(failed_models)}")
        return False
    else:
        print("✓ TEST PASSED: All 5 models trained successfully")
        return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("INFINITY FIX VALIDATION TESTS")
    print("="*80)
    print("\nThese tests verify that the infinity value fix works correctly.")
    print("They reproduce the issue from the problem statement and verify it's fixed.")
    
    results = []
    
    # Test 1: No infinity in features
    results.append(('No Infinity in Features', test_no_infinity_in_features()))
    
    # Test 2: Models train without errors
    results.append(('Models Train Successfully', test_models_train_without_infinity_errors()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED - Infinity fix is working correctly!")
    else:
        print("✗ SOME TESTS FAILED - Please review errors above")
    print("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
