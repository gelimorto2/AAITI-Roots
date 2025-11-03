import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from feature_engineering import FeatureEngineer

# Create data with very small values that can cause division issues
def create_problematic_data(n_samples=500):
    """Create synthetic data with extreme values."""
    timestamps = [datetime.now() - timedelta(hours=n_samples-i) for i in range(n_samples)]
    
    # Generate mostly normal data with a few edge cases
    prices = 50000 + np.random.randn(n_samples) * 100
    
    # Add some edge cases
    prices[100] = prices[99]  # Identical consecutive prices (0% change)
    prices[101] = prices[100]
    prices[102] = prices[101]
    
    # Very small price at index 200 to cause large ratio
    prices[200] = 0.01
    prices[201] = 50000
    
    volumes = np.random.uniform(100, 1000, n_samples) * 1e6
    volumes[50] = 0  # Zero volume
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n_samples) * 0.001),
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': volumes
    }, index=timestamps)
    
    return df

df = create_problematic_data(n_samples=500)
print("Created test data")

# Look at what happens during feature engineering step by step
from feature_engineering import FeatureEngineer

engineer = FeatureEngineer(df)

# Add price features and check
engineer.add_price_features()
print("\nAfter price features:")
inf_count = np.isinf(engineer.df.select_dtypes(include=[np.number]).values).sum()
print(f"  Infinity count: {inf_count}")
if inf_count > 0:
    for col in engineer.df.columns:
        if engineer.df[col].dtype in [np.float64, np.float32]:
            if np.isinf(engineer.df[col]).any():
                print(f"  - {col} has infinity")

# Add moving averages
engineer.add_moving_averages()
print("\nAfter moving averages:")
inf_count = np.isinf(engineer.df.select_dtypes(include=[np.number]).values).sum()
print(f"  Infinity count: {inf_count}")
if inf_count > 0:
    for col in engineer.df.columns:
        if engineer.df[col].dtype in [np.float64, np.float32]:
            if np.isinf(engineer.df[col]).any():
                print(f"  - {col} has {np.isinf(engineer.df[col]).sum()} infinity values")

# Add momentum indicators
engineer.add_momentum_indicators()
print("\nAfter momentum indicators:")
inf_count = np.isinf(engineer.df.select_dtypes(include=[np.number]).values).sum()
print(f"  Infinity count: {inf_count}")

# Add volatility indicators
engineer.add_volatility_indicators()
print("\nAfter volatility indicators:")
inf_count = np.isinf(engineer.df.select_dtypes(include=[np.number]).values).sum()
print(f"  Infinity count: {inf_count}")

# Add volume indicators
engineer.add_volume_indicators()
print("\nAfter volume indicators:")
inf_count = np.isinf(engineer.df.select_dtypes(include=[np.number]).values).sum()
print(f"  Infinity count: {inf_count}")
if inf_count > 0:
    for col in engineer.df.columns:
        if engineer.df[col].dtype in [np.float64, np.float32]:
            if np.isinf(engineer.df[col]).any():
                print(f"  - {col} has {np.isinf(engineer.df[col]).sum()} infinity values")
                print(f"    Max finite value: {engineer.df[col][~np.isinf(engineer.df[col])].max()}")

