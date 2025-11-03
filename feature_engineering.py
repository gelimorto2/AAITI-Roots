"""
Feature engineering module for BTC price prediction.
Creates technical indicators and additional features.
"""

import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice


class FeatureEngineer:
    """Create features from OHLCV data for model training."""
    
    def __init__(self, df):
        """
        Initialize with OHLCV data.
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume]
        """
        self.df = df.copy()
    
    def add_price_features(self):
        """Add basic price-based features."""
        # Returns
        self.df['returns'] = self.df['close'].pct_change()
        self.df['log_returns'] = np.log(self.df['close'] / self.df['close'].shift(1))
        
        # Price changes
        self.df['price_change'] = self.df['close'] - self.df['open']
        self.df['high_low_diff'] = self.df['high'] - self.df['low']
        self.df['high_close_diff'] = self.df['high'] - self.df['close']
        self.df['close_low_diff'] = self.df['close'] - self.df['low']
        
        # Price ratios
        self.df['close_open_ratio'] = self.df['close'] / self.df['open']
        
        return self
    
    def add_moving_averages(self, windows=[5, 10, 20, 50]):
        """Add moving average features."""
        for window in windows:
            # Simple Moving Average
            sma = SMAIndicator(close=self.df['close'], window=window)
            self.df[f'sma_{window}'] = sma.sma_indicator()
            
            # Exponential Moving Average
            ema = EMAIndicator(close=self.df['close'], window=window)
            self.df[f'ema_{window}'] = ema.ema_indicator()
            
            # Price to MA ratios
            self.df[f'close_sma_{window}_ratio'] = self.df['close'] / self.df[f'sma_{window}']
        
        return self
    
    def add_momentum_indicators(self):
        """Add momentum-based indicators."""
        # RSI
        rsi = RSIIndicator(close=self.df['close'], window=14)
        self.df['rsi'] = rsi.rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close']
        )
        self.df['stoch_k'] = stoch.stoch()
        self.df['stoch_d'] = stoch.stoch_signal()
        
        # MACD
        macd = MACD(close=self.df['close'])
        self.df['macd'] = macd.macd()
        self.df['macd_signal'] = macd.macd_signal()
        self.df['macd_diff'] = macd.macd_diff()
        
        return self
    
    def add_volatility_indicators(self):
        """Add volatility-based indicators."""
        # Bollinger Bands
        bb = BollingerBands(close=self.df['close'])
        self.df['bb_high'] = bb.bollinger_hband()
        self.df['bb_low'] = bb.bollinger_lband()
        self.df['bb_mid'] = bb.bollinger_mavg()
        self.df['bb_width'] = bb.bollinger_wband()
        self.df['bb_pband'] = bb.bollinger_pband()
        
        # Average True Range
        atr = AverageTrueRange(
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close']
        )
        self.df['atr'] = atr.average_true_range()
        
        # Historical volatility
        self.df['volatility_10'] = self.df['returns'].rolling(window=10).std()
        self.df['volatility_20'] = self.df['returns'].rolling(window=20).std()
        
        return self
    
    def add_volume_indicators(self):
        """Add volume-based indicators."""
        # On Balance Volume
        obv = OnBalanceVolumeIndicator(close=self.df['close'], volume=self.df['volume'])
        self.df['obv'] = obv.on_balance_volume()
        
        # Volume changes
        self.df['volume_change'] = self.df['volume'].pct_change()
        self.df['volume_sma_20'] = self.df['volume'].rolling(window=20).mean()
        self.df['volume_ratio'] = self.df['volume'] / self.df['volume_sma_20']
        
        return self
    
    def add_time_features(self):
        """Add time-based features."""
        self.df['hour'] = self.df.index.hour
        self.df['day_of_week'] = self.df.index.dayofweek
        self.df['day_of_month'] = self.df.index.day
        self.df['month'] = self.df.index.month
        
        # Cyclical encoding for time features
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        self.df['day_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
        
        return self
    
    def add_lag_features(self, lags=[1, 2, 3, 6, 12, 24]):
        """Add lagged features."""
        for lag in lags:
            self.df[f'close_lag_{lag}'] = self.df['close'].shift(lag)
            self.df[f'volume_lag_{lag}'] = self.df['volume'].shift(lag)
            self.df[f'returns_lag_{lag}'] = self.df['returns'].shift(lag)
        
        return self
    
    def create_target(self, horizon=1, threshold=0.0):
        """
        Create target variable for prediction.
        
        Args:
            horizon: Number of periods ahead to predict
            threshold: Threshold for classification (default 0 for up/down)
        
        Returns:
            DataFrame with target columns added
        """
        # Regression target: future price
        self.df['target_price'] = self.df['close'].shift(-horizon)
        
        # Regression target: future return
        self.df['target_return'] = (self.df['close'].shift(-horizon) - self.df['close']) / self.df['close']
        
        # Classification target: price direction
        self.df['target_direction'] = (self.df['target_return'] > threshold).astype(int)
        
        return self
    
    def build_all_features(self, target_horizon=1):
        """Build all features at once."""
        print("Building features...")
        
        self.add_price_features()
        self.add_moving_averages()
        self.add_momentum_indicators()
        self.add_volatility_indicators()
        self.add_volume_indicators()
        self.add_time_features()
        self.add_lag_features()
        self.create_target(horizon=target_horizon)
        
        # Replace infinity values with NaN before dropping
        initial_len = len(self.df)
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Drop NaN values
        self.df.dropna(inplace=True)
        final_len = len(self.df)
        
        print(f"Features built. Rows: {initial_len} -> {final_len}")
        print(f"Total features: {len(self.df.columns)}")
        
        return self.df
    
    def get_feature_names(self, exclude_targets=True):
        """Get list of feature column names."""
        if exclude_targets:
            exclude_cols = ['target_price', 'target_return', 'target_direction']
            return [col for col in self.df.columns if col not in exclude_cols]
        return list(self.df.columns)


if __name__ == "__main__":
    # Example usage
    from data_fetcher import BTCDataFetcher
    from datetime import datetime, timedelta
    
    # Fetch data
    fetcher = BTCDataFetcher()
    df = fetcher.fetch_data(
        source='yfinance',
        start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d'),
        interval='1h'
    )
    
    if df is not None:
        # Create features
        engineer = FeatureEngineer(df)
        df_features = engineer.build_all_features(target_horizon=1)
        
        print("\nFeature columns:")
        print(engineer.get_feature_names())
        print(f"\nData shape: {df_features.shape}")
        print("\nSample data:")
        print(df_features.head())
