# models/base_model.py
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all prediction models"""
    
    def __init__(self, model_name: str, model_type: str = "regression"):
        self.model_name = model_name
        self.model_type = model_type  # "regression" or "classification"
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.training_history = {}
        self.feature_names = []
    
    @abstractmethod
    def build(self, input_shape: Tuple, **kwargs):
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray, **kwargs) -> Dict:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    def prepare_sequences(self, data: np.ndarray, lookback: int, 
                         target_col_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction
        
        Args:
            data: Scaled feature array
            lookback: Number of time steps to look back
            target_col_idx: Index of target column in data
        
        Returns:
            X, y arrays
        """
        X, y = [], []
        
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i, target_col_idx])
        
        return np.array(X), np.array(y)
    
    def scale_data(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Scale the data
        
        Args:
            data: Input data
            fit: Whether to fit the scaler (True for training, False for inference)
        
        Returns:
            Scaled data
        """
        if fit:
            return self.scaler.fit_transform(data)
        else:
            return self.scaler.transform(data)
    
    def inverse_scale(self, data: np.ndarray, target_col_idx: int = 0) -> np.ndarray:
        """
        Inverse transform predictions back to original scale
        
        Args:
            data: Scaled predictions
            target_col_idx: Index of target column
        
        Returns:
            Original scale predictions
        """
        # Create dummy array with same shape as training data
        dummy = np.zeros((len(data), self.scaler.n_features_in_))
        dummy[:, target_col_idx] = data.flatten()
        
        # Inverse transform
        inversed = self.scaler.inverse_transform(dummy)
        return inversed[:, target_col_idx]
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
        
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        predictions = self.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions),
            'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
        }
        
        logger.info(f"{self.model_name} Evaluation Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        return metrics
    
    def save(self, path: str):
        """Save model and scaler"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model-specific artifacts (implemented by child classes)
        self._save_model(save_path)
        
        # Save scaler
        joblib.dump(self.scaler, save_path / f"{self.model_name}_scaler.pkl")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }
        joblib.dump(metadata, save_path / f"{self.model_name}_metadata.pkl")
        
        logger.info(f"Model saved to {save_path}")
    
    def load(self, path: str):
        """Load model and scaler"""
        load_path = Path(path)
        
        # Load model-specific artifacts
        self._load_model(load_path)
        
        # Load scaler
        self.scaler = joblib.load(load_path / f"{self.model_name}_scaler.pkl")
        
        # Load metadata
        metadata = joblib.load(load_path / f"{self.model_name}_metadata.pkl")
        self.is_trained = metadata['is_trained']
        self.feature_names = metadata['feature_names']
        self.training_history = metadata['training_history']
        
        logger.info(f"Model loaded from {load_path}")
    
    @abstractmethod
    def _save_model(self, path: Path):
        """Save model-specific artifacts (implemented by child classes)"""
        pass
    
    @abstractmethod
    def _load_model(self, path: Path):
        """Load model-specific artifacts (implemented by child classes)"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'name': self.model_name,
            'type': self.model_type,
            'is_trained': self.is_trained,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names
        }