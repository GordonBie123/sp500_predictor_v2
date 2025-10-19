# training/trainer.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import json

from data.pipeline import DataPipeline
from features.feature_store import FeatureStore
from models.ensemble import EnsembleModel

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Complete training pipeline for stock prediction models
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize components
        self.data_pipeline = DataPipeline()
        self.feature_store = FeatureStore()
        self.model = None
        
        # Training parameters
        self.lookback_days = self.config.get('lookback_days', 60)
        self.prediction_horizon = self.config.get('prediction_horizon', 5)
        self.train_split = self.config.get('train_split', 0.7)
        self.val_split = self.config.get('val_split', 0.15)
        
        # Storage
        self.training_results = {}
        
    def prepare_data(self, symbol: str, lookback_months: int = 24) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fetch and prepare data for training
        
        Args:
            symbol: Stock ticker symbol
            lookback_months: Number of months of historical data to fetch
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info(f"Preparing data for {symbol}...")
        
        # Calculate date range for fetching more data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_months * 30)  # Approximate months to days
        
        logger.info(f"Fetching data from {start_date.date()} to {end_date.date()}")
        
        # Step 1: Fetch raw stock data with explicit date range
        raw_data = self.data_pipeline.fetch_and_process(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        logger.info(f"Fetched {len(raw_data)} rows of raw data")
        
        # Check if we have enough data
        min_required = self.lookback_days + self.prediction_horizon + 100
        if len(raw_data) < min_required:
            raise ValueError(
                f"Not enough raw data. Need at least {min_required} rows, got {len(raw_data)}. "
                f"Try increasing lookback_months or choosing a different stock."
            )
        
        # Step 2: Generate features
        logger.info("Generating features...")
        features_data = self.feature_store.generate_features(raw_data)
        logger.info(f"Generated {len(self.feature_store.get_feature_names())} features")
        logger.info(f"Feature data shape after generation: {features_data.shape}")
        
        # Check again after feature generation (some rows may be dropped due to NaN)
        if len(features_data) < min_required:
            logger.warning(
                f"Lost {len(raw_data) - len(features_data)} rows during feature generation "
                f"(likely due to rolling window calculations)"
            )
            raise ValueError(
                f"Not enough data after feature generation. Need at least {min_required} rows, got {len(features_data)}. "
                f"Original data had {len(raw_data)} rows. Try increasing lookback_months."
            )
        
        # Step 3: Prepare for modeling
        X, y = self._create_sequences(features_data)
        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        
        # Verify we have 3D data
        if len(X.shape) != 3:
            raise ValueError(f"Expected 3D array, got shape {X.shape}")
        
        # Verify we have enough sequences for train/val/test split
        min_sequences = 150  # Minimum for reasonable train/val/test split
        if len(X) < min_sequences:
            raise ValueError(
                f"Not enough sequences created. Need at least {min_sequences}, got {len(X)}. "
                f"Try fetching more historical data by increasing lookback_months."
            )
        
        # Step 4: Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y)
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction
        
        Args:
            data: DataFrame with features
        
        Returns:
            X, y arrays
        """
        from sklearn.preprocessing import MinMaxScaler
        
        logger.info(f"Creating sequences from data with shape: {data.shape}")
        
        # Remove non-numeric columns (Date, Symbol, etc.)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_data = data[numeric_cols].copy()
        
        logger.info(f"Numeric columns: {len(numeric_cols)}")
        logger.info(f"Sample columns: {list(numeric_cols[:10])}")
        
        # Find Close price column index (target)
        if 'Close' not in numeric_data.columns:
            raise ValueError("Close price column not found in data")
        
        target_col_idx = numeric_data.columns.get_loc('Close')
        logger.info(f"Target column (Close) index: {target_col_idx}")
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        logger.info(f"Scaled data shape: {scaled_data.shape}")
        
        # Save scaler for later use
        self.scaler = scaler
        self.feature_columns = numeric_data.columns.tolist()
        self.target_col_idx = target_col_idx
        
        # Create sequences
        X, y = [], []
        
        max_index = len(scaled_data) - self.prediction_horizon
        logger.info(f"Creating sequences from index {self.lookback_days} to {max_index}")
        
        for i in range(self.lookback_days, max_index):
            # Input: lookback_days of all features
            X.append(scaled_data[i-self.lookback_days:i])
            # Target: Close price prediction_horizon days ahead
            y.append(scaled_data[i + self.prediction_horizon - 1, target_col_idx])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created sequences - X shape: {X.shape}, y shape: {y.shape}")
        
        if len(X) == 0:
            raise ValueError("No sequences created! Check if data is long enough.")
        
        return X, y
    
    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """Split data into train, validation, and test sets"""
        n_samples = len(X)
        
        train_end = int(n_samples * self.train_split)
        val_end = int(n_samples * (self.train_split + self.val_split))
        
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]
        
        logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(self, symbol: str, model_type: str = "ensemble") -> Dict:
        """
        Complete training pipeline
        
        Args:
            symbol: Stock ticker symbol
            model_type: 'ensemble', 'lstm', 'rf', or 'xgb'
        
        Returns:
            Training results
        """
        logger.info("="*60)
        logger.info(f"Starting training for {symbol}")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        try:
            # Prepare data
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(symbol)
            
            logger.info(f"\nData preparation complete:")
            logger.info(f"  X_train shape: {X_train.shape}")
            logger.info(f"  X_val shape: {X_val.shape}")
            logger.info(f"  X_test shape: {X_test.shape}")
            
        except Exception as e:
            logger.error(f"Error during data preparation: {str(e)}")
            raise
        
        # Initialize model
        if model_type == "ensemble":
            self.model = EnsembleModel()
        elif model_type == "lstm":
            from models.lstm_model import LSTMModel
            self.model = LSTMModel()
        elif model_type == "rf":
            from models.tree_models import RandomForestModel
            self.model = RandomForestModel()
        elif model_type == "xgb":
            from models.tree_models import XGBoostModel
            self.model = XGBoostModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        logger.info(f"\nBuilding {model_type} model with input shape: {input_shape}")
        self.model.build(input_shape=input_shape)
        
        # Train model
        logger.info(f"\nTraining {model_type} model...")
        if model_type == "ensemble":
            # For ensemble, reduce LSTM epochs for faster training
            training_history = self.model.train(
                X_train, y_train, X_val, y_val,
                lstm_config={'epochs': 20}  # Reduced for faster training
            )
        else:
            if model_type == "lstm":
                training_history = self.model.train(
                    X_train, y_train, X_val, y_val,
                    epochs=20
                )
            else:
                training_history = self.model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        if model_type == "ensemble":
            test_results = self.model.evaluate_all(X_test, y_test)
        else:
            test_results = {model_type: self.model.evaluate(X_test, y_test)}
        
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store results
        self.training_results = {
            'symbol': symbol,
            'model_type': model_type,
            'training_history': training_history,
            'test_results': test_results,
            'training_time': training_time,
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val),
            'n_test_samples': len(X_test),
            'n_features': X_train.shape[2],
            'lookback_days': self.lookback_days,
            'prediction_horizon': self.prediction_horizon
        }
        
        logger.info("\n" + "="*60)
        logger.info("Training Complete!")
        logger.info("="*60)
        logger.info(f"Total training time: {training_time:.2f} seconds")
        logger.info(f"Test set results:")
        for model_name, metrics in test_results.items():
            logger.info(f"\n{model_name}:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        return self.training_results
    
    def save_model(self, path: str = "models/saved/trained_model"):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(str(save_path))
        
        # Save scaler and metadata
        import joblib
        joblib.dump(self.scaler, save_path / "scaler.pkl")
        
        metadata = {
            'feature_columns': self.feature_columns,
            'target_col_idx': self.target_col_idx,
            'lookback_days': self.lookback_days,
            'prediction_horizon': self.prediction_horizon,
            'training_results': self.training_results
        }
        
        with open(save_path / "metadata.json", 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_safe_metadata = self._make_json_safe(metadata)
            json.dump(json_safe_metadata, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    def _make_json_safe(self, obj):
        """Convert numpy types to Python types for JSON"""
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj