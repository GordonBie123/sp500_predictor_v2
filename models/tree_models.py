# models/tree_models.py
import numpy as np
from typing import Tuple, Dict
from pathlib import Path
import logging
from .base_model import BaseModel
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib

logger = logging.getLogger(__name__)

class RandomForestModel(BaseModel):
    """Random Forest for stock prediction"""
    
    def __init__(self, model_name: str = "RandomForest"):
        super().__init__(model_name=model_name, model_type="regression")
        
        self.config = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
    
    def build(self, input_shape: Tuple = None, **kwargs):
        """Build Random Forest model"""
        self.config.update(kwargs)
        
        self.model = RandomForestRegressor(**self.config)
        logger.info(f"Random Forest model built with {self.config['n_estimators']} trees")
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, **kwargs) -> Dict:
        """Train Random Forest"""
        
        if self.model is None:
            self.build()
        
        # Flatten sequences if 3D
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_val = X_val.reshape(X_val.shape[0], -1)
        
        logger.info(f"Training Random Forest...")
        logger.info(f"Training samples: {len(X_train)}, Features: {X_train.shape[1]}")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on validation set
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)
        
        self.training_history = {
            'train_r2': train_score,
            'val_r2': val_score,
            'feature_importance': self.model.feature_importances_
        }
        
        logger.info(f"Training complete!")
        logger.info(f"Training R²: {train_score:.4f}")
        logger.info(f"Validation R²: {val_score:.4f}")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Flatten if 3D
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if not self.feature_names:
            return {}
        
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance
    
    def _save_model(self, path: Path):
        """Save Random Forest model"""
        joblib.dump(self.model, path / f"{self.model_name}.pkl")
    
    def _load_model(self, path: Path):
        """Load Random Forest model"""
        self.model = joblib.load(path / f"{self.model_name}.pkl")


class XGBoostModel(BaseModel):
    """XGBoost for stock prediction"""
    
    def __init__(self, model_name: str = "XGBoost"):
        super().__init__(model_name=model_name, model_type="regression")
        
        self.config = {
            'n_estimators': 200,
            'max_depth': 7,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        self.use_early_stopping = False
    
    def build(self, input_shape: Tuple = None, **kwargs):
        """Build XGBoost model"""
        self.config.update(kwargs)
        
        # Check if early stopping is requested
        if 'early_stopping_rounds' in kwargs:
            self.use_early_stopping = True
            self.early_stopping_rounds = kwargs.pop('early_stopping_rounds')
        
        self.model = xgb.XGBRegressor(**self.config)
        logger.info(f"XGBoost model built")
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, **kwargs) -> Dict:
        """Train XGBoost"""
        
        if self.model is None:
            self.build(**kwargs)
        
        # Flatten sequences if 3D
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_val = X_val.reshape(X_val.shape[0], -1)
        
        logger.info(f"Training XGBoost...")
        logger.info(f"Training samples: {len(X_train)}, Features: {X_train.shape[1]}")
        
        # Prepare training arguments
        fit_params = {
            'eval_set': [(X_train, y_train), (X_val, y_val)],
            'verbose': False
        }
        
        # Add early stopping if enabled
        if self.use_early_stopping:
            fit_params['early_stopping_rounds'] = self.early_stopping_rounds
        
        # Train
        self.model.fit(X_train, y_train, **fit_params)
        
        self.is_trained = True
        
        # Get evaluation results
        results = self.model.evals_result()
        
        self.training_history = {
            'train_rmse': results['validation_0']['rmse'],
            'val_rmse': results['validation_1']['rmse'],
            'feature_importance': self.model.feature_importances_
        }
        
        logger.info(f"Training complete!")
        
        # Only log best_iteration if early stopping was used
        if self.use_early_stopping and hasattr(self.model, 'best_iteration'):
            logger.info(f"Best iteration: {self.model.best_iteration}")
        
        logger.info(f"Final validation RMSE: {results['validation_1']['rmse'][-1]:.4f}")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Flatten if 3D
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if not self.feature_names:
            return {}
        
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance
    
    def _save_model(self, path: Path):
        """Save XGBoost model"""
        self.model.save_model(path / f"{self.model_name}.json")
    
    def _load_model(self, path: Path):
        """Load XGBoost model"""
        self.model = xgb.XGBRegressor()
        self.model.load_model(path / f"{self.model_name}.json")