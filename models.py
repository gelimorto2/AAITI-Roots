"""
Multiple AI model implementations for BTC price prediction.
Includes LSTM, GRU, Transformer, Random Forest, and XGBoost models.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class BaseModel:
    """Base class for all models."""
    
    def __init__(self, name):
        self.name = name
        self.model = None
        self.scaler = None
        self.history = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model."""
        raise NotImplementedError
    
    def predict(self, X):
        """Make predictions."""
        raise NotImplementedError
    
    def save(self, filepath):
        """Save model to file."""
        raise NotImplementedError
    
    def load(self, filepath):
        """Load model from file."""
        raise NotImplementedError


class LSTMModel(BaseModel):
    """LSTM model for time series prediction."""
    
    def __init__(self, input_shape, units=[64, 32], dropout=0.2, task='regression'):
        super().__init__('LSTM')
        self.input_shape = input_shape
        self.units = units
        self.dropout = dropout
        self.task = task
        self.scaler = StandardScaler()
        self._build_model()
    
    def _build_model(self):
        """Build LSTM architecture."""
        model = keras.Sequential([
            layers.Input(shape=self.input_shape)
        ])
        
        # Add LSTM layers
        for i, unit in enumerate(self.units):
            return_sequences = i < len(self.units) - 1
            model.add(layers.LSTM(unit, return_sequences=return_sequences))
            model.add(layers.Dropout(self.dropout))
        
        # Output layer
        if self.task == 'regression':
            model.add(layers.Dense(1))
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        else:  # classification
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        self.model = model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Train LSTM model."""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val.reshape(X_val.shape[0], -1))
            X_val_scaled = X_val_scaled.reshape(X_val.shape)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss', 
                         patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss' if X_val is not None else 'loss',
                            factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions."""
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1))
        X_scaled = X_scaled.reshape(X.shape)
        return self.model.predict(X_scaled, verbose=0)


class GRUModel(BaseModel):
    """GRU model for time series prediction."""
    
    def __init__(self, input_shape, units=[64, 32], dropout=0.2, task='regression'):
        super().__init__('GRU')
        self.input_shape = input_shape
        self.units = units
        self.dropout = dropout
        self.task = task
        self.scaler = StandardScaler()
        self._build_model()
    
    def _build_model(self):
        """Build GRU architecture."""
        model = keras.Sequential([
            layers.Input(shape=self.input_shape)
        ])
        
        # Add GRU layers
        for i, unit in enumerate(self.units):
            return_sequences = i < len(self.units) - 1
            model.add(layers.GRU(unit, return_sequences=return_sequences))
            model.add(layers.Dropout(self.dropout))
        
        # Output layer
        if self.task == 'regression':
            model.add(layers.Dense(1))
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        else:  # classification
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        self.model = model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Train GRU model."""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val.reshape(X_val.shape[0], -1))
            X_val_scaled = X_val_scaled.reshape(X_val.shape)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss',
                         patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss' if X_val is not None else 'loss',
                            factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions."""
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1))
        X_scaled = X_scaled.reshape(X.shape)
        return self.model.predict(X_scaled, verbose=0)


class TransformerModel(BaseModel):
    """Transformer model for time series prediction."""
    
    def __init__(self, input_shape, num_heads=4, ff_dim=128, num_blocks=2, 
                 dropout=0.1, task='regression'):
        super().__init__('Transformer')
        self.input_shape = input_shape
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.task = task
        self.scaler = StandardScaler()
        self._build_model()
    
    def _transformer_block(self, x, ff_dim, num_heads, dropout, embed_dim):
        """Single transformer block."""
        # Multi-head attention
        key_dim = embed_dim // num_heads
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=key_dim
        )(x, x)
        attn_output = layers.Dropout(dropout)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed forward network
        ff_output = layers.Dense(ff_dim, activation='relu')(x)
        ff_output = layers.Dropout(dropout)(ff_output)
        ff_output = layers.Dense(embed_dim)(ff_output)
        ff_output = layers.Dropout(dropout)(ff_output)
        
        return layers.LayerNormalization(epsilon=1e-6)(x + ff_output)
    
    def _build_model(self):
        """Build Transformer architecture."""
        inputs = layers.Input(shape=self.input_shape)
        x = inputs
        embed_dim = self.input_shape[-1]
        
        # Add transformer blocks
        for _ in range(self.num_blocks):
            x = self._transformer_block(x, self.ff_dim, self.num_heads, self.dropout, embed_dim)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        
        # Output layer
        if self.task == 'regression':
            outputs = layers.Dense(1)(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        else:  # classification
            outputs = layers.Dense(1, activation='sigmoid')(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        self.model = model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Train Transformer model."""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val.reshape(X_val.shape[0], -1))
            X_val_scaled = X_val_scaled.reshape(X_val.shape)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss',
                         patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss' if X_val is not None else 'loss',
                            factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions."""
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1))
        X_scaled = X_scaled.reshape(X.shape)
        return self.model.predict(X_scaled, verbose=0)


class RandomForestModel(BaseModel):
    """Random Forest model."""
    
    def __init__(self, n_estimators=100, max_depth=10, task='regression'):
        super().__init__('RandomForest')
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.task = task
        self.scaler = StandardScaler()
        
        if task == 'regression':
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train Random Forest model."""
        # Flatten if needed
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        
        return None
    
    def predict(self, X):
        """Make predictions."""
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        return self.model.feature_importances_


class XGBoostModel(BaseModel):
    """XGBoost model."""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, task='regression'):
        super().__init__('XGBoost')
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.task = task
        self.scaler = StandardScaler()
        
        if task == 'regression':
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                n_jobs=-1
            )
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model."""
        # Flatten if needed
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Prepare validation set if provided
        eval_set = None
        if X_val is not None:
            if len(X_val.shape) > 2:
                X_val = X_val.reshape(X_val.shape[0], -1)
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_val_scaled, y_val)]
        
        # Train
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        return None
    
    def predict(self, X):
        """Make predictions."""
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        return self.model.feature_importances_
