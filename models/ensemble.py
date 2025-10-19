# models/ensemble.py
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from .base_model import BaseModel
from .lstm_model import LSTMModel
from .tree_models import RandomForestModel, XGBoostModel

logger = logging.getLogger(__name__)

class EnsembleModel(BaseModel):
    """
    Ensemble model that combines LSTM, Random Forest, and XGBoost
    Uses weighted averaging or stacking approach
    """
    
    def __init__(self, model_name: str = "Ensemble", ensemble_method: str = "weighted"):
        super().__init__(model_name=model_name, model_type="regression")
        
        self.ensemble_method = ensemble_method  # 'weighted' or 'stacking'
        
        # Initialize sub-models
        self.lstm = LSTMModel(model_name="LSTM")
        self.rf = RandomForestModel(model_name="RandomForest")
        self.xgb = XGBoostModel(model_name="XGBoost")
        
        # Default weights (can be learned)
        self.weights = {
            'lstm': 0.4,
            'rf': 0.3,
            'xgb': 0.3
        }
        
        self.models = {
            'lstm': self.lstm,
            'rf': self.rf,
            'xgb': self.xgb
        }
        
        logger.info(f"Ensemble model initialized with method: {ensemble_method}")
    
    def build(self, input_shape: Tuple, **kwargs):
        """
        Build all sub-models
        
        Args:
            input_shape: (lookback_steps, n_features) for LSTM
            **kwargs: Config for each model
        """
        logger.info("Building ensemble sub-models...")
        
        # Build LSTM
        lstm_config = kwargs.get('lstm_config', {})
        self.lstm.build(input_shape=input_shape, **lstm_config)
        logger.info("✓ LSTM built")
        
        # Build Random Forest
        rf_config = kwargs.get('rf_config', {})
        self.rf.build(**rf_config)
        logger.info("✓ Random Forest built")
        
        # Build XGBoost
        xgb_config = kwargs.get('xgb_config', {})
        self.xgb.build(**xgb_config)
        logger.info("✓ XGBoost built")
        
        logger.info("All ensemble sub-models built successfully")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, **kwargs) -> Dict:
        """
        Train all sub-models
        
        Args:
            X_train: Training features (3D for LSTM)
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Training configs
        
        Returns:
            Combined training history
        """
        logger.info(f"Training ensemble model with {self.ensemble_method} method...")
        
        training_results = {}
        
        # Train LSTM
        logger.info("\n" + "="*50)
        logger.info("Training LSTM...")
        logger.info("="*50)
        lstm_history = self.lstm.train(X_train, y_train, X_val, y_val)
        training_results['lstm'] = lstm_history
        
        # Train Random Forest
        logger.info("\n" + "="*50)
        logger.info("Training Random Forest...")
        logger.info("="*50)
        rf_history = self.rf.train(X_train, y_train, X_val, y_val)
        training_results['rf'] = rf_history
        
        # Train XGBoost
        logger.info("\n" + "="*50)
        logger.info("Training XGBoost...")
        logger.info("="*50)
        xgb_history = self.xgb.train(X_train, y_train, X_val, y_val)
        training_results['xgb'] = xgb_history
        
        # Learn optimal weights if using weighted ensemble
        if self.ensemble_method == "weighted":
            self._learn_weights(X_val, y_val)
        
        self.is_trained = True
        self.training_history = training_results
        
        # Evaluate ensemble
        logger.info("\n" + "="*50)
        logger.info("Evaluating Ensemble...")
        logger.info("="*50)
        ensemble_metrics = self.evaluate(X_val, y_val)
        training_results['ensemble'] = ensemble_metrics
        
        return training_results
    
    def _learn_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Learn optimal weights based on validation performance
        Uses simple inverse error weighting
        """
        logger.info("Learning optimal ensemble weights...")
        
        # Get predictions from each model
        lstm_pred = self.lstm.predict(X_val)
        rf_pred = self.rf.predict(X_val)
        xgb_pred = self.xgb.predict(X_val)
        
        # Calculate errors
        from sklearn.metrics import mean_squared_error
        lstm_error = mean_squared_error(y_val, lstm_pred)
        rf_error = mean_squared_error(y_val, rf_pred)
        xgb_error = mean_squared_error(y_val, xgb_pred)
        
        # Inverse error weighting (lower error = higher weight)
        lstm_weight = 1 / (lstm_error + 1e-6)
        rf_weight = 1 / (rf_error + 1e-6)
        xgb_weight = 1 / (xgb_error + 1e-6)
        
        # Normalize weights to sum to 1
        total_weight = lstm_weight + rf_weight + xgb_weight
        self.weights = {
            'lstm': lstm_weight / total_weight,
            'rf': rf_weight / total_weight,
            'xgb': xgb_weight / total_weight
        }
        
        logger.info(f"Learned weights: LSTM={self.weights['lstm']:.3f}, "
                   f"RF={self.weights['rf']:.3f}, XGB={self.weights['xgb']:.3f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions
        
        Args:
            X: Input features
        
        Returns:
            Ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained yet")
        
        # Get predictions from each model
        lstm_pred = self.lstm.predict(X)
        rf_pred = self.rf.predict(X)
        xgb_pred = self.xgb.predict(X)
        
        # Combine using weights
        ensemble_pred = (
            self.weights['lstm'] * lstm_pred +
            self.weights['rf'] * rf_pred +
            self.weights['xgb'] * xgb_pred
        )
        
        return ensemble_pred
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation
        
        Returns:
            (mean_predictions, std_predictions)
        """
        # Get predictions from each model
        lstm_pred = self.lstm.predict(X)
        rf_pred = self.rf.predict(X)
        xgb_pred = self.xgb.predict(X)
        
        # Stack predictions
        all_preds = np.vstack([lstm_pred, rf_pred, xgb_pred])
        
        # Calculate mean and std
        mean_pred = all_preds.mean(axis=0)
        std_pred = all_preds.std(axis=0)
        
        return mean_pred, std_pred
    
    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from each individual model"""
        return {
            'lstm': self.lstm.predict(X),
            'rf': self.rf.predict(X),
            'xgb': self.xgb.predict(X),
            'ensemble': self.predict(X)
        }
    
    def evaluate_all(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """Evaluate all models including ensemble"""
        results = {}
        
        # Evaluate each model
        results['lstm'] = self.lstm.evaluate(X_test, y_test)
        results['rf'] = self.rf.evaluate(X_test, y_test)
        results['xgb'] = self.xgb.evaluate(X_test, y_test)
        results['ensemble'] = self.evaluate(X_test, y_test)
        
        return results
    
    def _save_model(self, path: Path):
        """Save ensemble and all sub-models"""
        # Save each sub-model
        self.lstm.save(str(path / "lstm"))
        self.rf.save(str(path / "rf"))
        self.xgb.save(str(path / "xgb"))
        
        # Save ensemble-specific data
        import joblib
        ensemble_data = {
            'weights': self.weights,
            'ensemble_method': self.ensemble_method
        }
        joblib.dump(ensemble_data, path / "ensemble_data.pkl")
        
        logger.info(f"Ensemble model saved to {path}")
    
    def _load_model(self, path: Path):
        """Load ensemble and all sub-models"""
        # Load each sub-model
        self.lstm.load(str(path / "lstm"))
        self.rf.load(str(path / "rf"))
        self.xgb.load(str(path / "xgb"))
        
        # Load ensemble-specific data
        import joblib
        ensemble_data = joblib.load(path / "ensemble_data.pkl")
        self.weights = ensemble_data['weights']
        self.ensemble_method = ensemble_data['ensemble_method']
        
        logger.info(f"Ensemble model loaded from {path}")