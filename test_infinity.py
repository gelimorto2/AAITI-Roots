import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from feature_engineering import FeatureEngineer
from models import LSTMModel, GRUModel, TransformerModel, RandomForestModel, XGBoostModel

# Create synthetic data with edge cases that could cause infinity
def create_problematic_data(n_samples=500):
    """Create synthetic data with potential infinity issues."""
    timestamps = [datetime.now() - timedelta(hours=n_samples-i) for i in range(n_samples)]
    
    # Generate data with zero volumes and constant prices (can cause division by zero)
    prices = np.ones(n_samples) * 50000
    volumes = np.zeros(n_samples)  # Zero volumes
    volumes[10:] = 100  # Some non-zero volumes
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': volumes
    }, index=timestamps)
    
    return df

# Test with problematic data
df = create_problematic_data(n_samples=200)
print("Created test data with potential infinity issues")
print(f"Data shape: {df.shape}")

# Engineer features
engineer = FeatureEngineer(df)
df_features = engineer.build_all_features(target_horizon=1)

print(f"\nFeatures created: {df_features.shape}")

# Check for infinity and NaN values
print("\nChecking for infinity and NaN values:")
print(f"  Infinity values: {np.isinf(df_features.select_dtypes(include=[np.number]).values).sum()}")
print(f"  NaN values: {np.isnan(df_features.select_dtypes(include=[np.number]).values).sum()}")

if np.isinf(df_features.select_dtypes(include=[np.number]).values).any():
    print("\nColumns with infinity values:")
    for col in df_features.columns:
        if df_features[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            if np.isinf(df_features[col]).any():
                print(f"  - {col}: {np.isinf(df_features[col]).sum()} infinite values")

# Try to train a model
print("\n\nAttempting to train models with this data...")
feature_cols = engineer.get_feature_names(exclude_targets=True)
X = df_features[feature_cols].values
y = df_features['target_return'].values

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X contains infinity: {np.isinf(X).any()}")
print(f"X contains NaN: {np.isnan(X).any()}")
print(f"X min: {np.min(X[~np.isinf(X)])}, X max: {np.max(X[~np.isinf(X)])}")

# Split data
train_size = int(0.7 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

print(f"\nTraining RandomForest...")
try:
    from models import RandomForestModel
    model = RandomForestModel(n_estimators=10, max_depth=5, task='regression')
    model.fit(X_train, y_train)
    print("  ✓ RandomForest trained successfully")
except Exception as e:
    print(f"  ✗ RandomForest failed: {e}")
