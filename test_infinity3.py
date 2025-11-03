import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Simulate the exact problematic scenarios
print("Testing division by zero scenarios:\n")

# Scenario 1: pct_change with zero values
print("1. pct_change() with zero values:")
s = pd.Series([100, 0, 100, 200])
result = s.pct_change()
print(f"   Series: {s.tolist()}")
print(f"   pct_change: {result.tolist()}")
print(f"   Contains inf: {np.isinf(result).any()}")

# Scenario 2: Division by zero in ratios
print("\n2. Direct division with zero:")
a = pd.Series([100, 100, 100])
b = pd.Series([10, 0, 20])
result = a / b
print(f"   a / b where b has zero: {result.tolist()}")
print(f"   Contains inf: {np.isinf(result).any()}")

# Scenario 3: close_sma ratio when SMA has zero (or very small values)
print("\n3. Ratio calculations:")
close = pd.Series([50000, 50000, 0.01, 50000])
sma = pd.Series([50000, 49000, 0.01, 50000])
result = close / sma
print(f"   close / sma: {result.tolist()}")
print(f"   Contains inf: {np.isinf(result).any()}")

# Scenario 4: What happens after dropna?
print("\n4. After dropna():")
df = pd.DataFrame({'a': [1, 2, np.inf, 4], 'b': [5, 6, 7, 8]})
print(f"   Before: {df.shape} rows, contains inf: {np.isinf(df.values).any()}")
df_clean = df.dropna()
print(f"   After dropna: {df_clean.shape} rows, contains inf: {np.isinf(df_clean.values).any()}")
df_clean_inf = df.replace([np.inf, -np.inf], np.nan).dropna()
print(f"   After replace inf then dropna: {df_clean_inf.shape} rows, contains inf: {np.isinf(df_clean_inf.values).any()}")

