# models/lstm_model.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.layers as layers
import keras.callbacks as callbacks
import keras.models as models
from typing import Tuple, Dict
from pathlib import Path
import logging
from .base_model import BaseModel

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class LSTMModel(BaseModel):
    """
    LSTM Neural Network for time series prediction
    """
    
    def __init__(self, model_name: str = "LSTM"):
        super().__init__(model_name=model_name, model_type="regression")
        
        # Default hyperparameters
        self.config = {
            'lstm_units': [128, 64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 15,
            'reduce_lr_patience': 7
        }
    
    def build(self, input_shape: Tuple, **kwargs):
        """
        Build LSTM model architecture
        
        Args:
            input_shape: (lookback_steps, n_features)
            **kwargs: Override default config
        """
        # Update config with any provided kwargs
        self.config.update(kwargs)
        
        logger.info(f"Building LSTM model with input shape: {input_shape}")
        
        # Create model using Sequential API
        model = keras.Sequential(name=self.model_name)
        
        # First LSTM layer
        model.add(layers.LSTM(
            self.config['lstm_units'][0],
            return_sequences=True,
            input_shape=input_shape,
            name='lstm_1'
        ))
        model.add(layers.Dropout(self.config['dropout_rate'], name='dropout_1'))
        
        # Second LSTM layer
        model.add(layers.LSTM(
            self.config['lstm_units'][1],
            return_sequences=True,
            name='lstm_2'
        ))
        model.add(layers.Dropout(self.config['dropout_rate'], name='dropout_2'))
        
        # Third LSTM layer
        model.add(layers.LSTM(
            self.config['lstm_units'][2],
            return_sequences=False,
            name='lstm_3'
        ))
        model.add(layers.Dropout(self.config['dropout_rate'], name='dropout_3'))
        
        # Dense layers
        model.add(layers.Dense(16, activation='relu', name='dense_1'))
        model.add(layers.Dense(1, name='output'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
        logger.info(f"LSTM model built successfully")
        logger.info(f"Total parameters: {model.count_params():,}")
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, **kwargs) -> Dict:
        """
        Train the LSTM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional training parameters
        
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        logger.info(f"Training LSTM model...")
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Create directory for checkpoints
        Path('models/saved').mkdir(parents=True, exist_ok=True)
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                'models/saved/lstm_best.keras',  # Use .keras format
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callback_list,
            verbose=1
        )
        
        self.is_trained = True
        self.training_history = history.history
        
        # Log final metrics
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        logger.info(f"Training complete!")
        logger.info(f"Final training loss: {final_train_loss:.4f}")
        logger.info(f"Final validation loss: {final_val_loss:.4f}")
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
        
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def predict_with_uncertainty(self, X: np.ndarray, n_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation using Monte Carlo Dropout
        
        Args:
            X: Input features
            n_iterations: Number of MC iterations
        
        Returns:
            (mean_predictions, std_predictions)
        """
        predictions = []
        
        # Enable dropout during inference
        for _ in range(n_iterations):
            pred = self.model(X, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        mean_pred = predictions.mean(axis=0).flatten()
        std_pred = predictions.std(axis=0).flatten()
        
        return mean_pred, std_pred
    
    def _save_model(self, path: Path):
        """Save LSTM model"""
        model_path = path / f"{self.model_name}.keras"
        self.model.save(model_path)
        logger.info(f"LSTM model saved to {model_path}")
    
    def _load_model(self, path: Path):
        """Load LSTM model"""
        model_path = path / f"{self.model_name}.keras"
        self.model = keras.models.load_model(model_path)
        logger.info(f"LSTM model loaded from {model_path}")