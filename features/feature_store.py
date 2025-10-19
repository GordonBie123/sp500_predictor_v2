# features/feature_store.py
import pandas as pd
from typing import List, Dict, Optional
import logging
from .technical_features import TechnicalFeatures
from .regime_features import RegimeFeatures

logger = logging.getLogger(__name__)

class FeatureStore:
    """
    Central manager for all feature engineering
    Coordinates multiple feature generators
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.feature_generators = []
        self.all_feature_names = []
        
        # Initialize feature generators
        self._initialize_generators()
    
    def _initialize_generators(self):
        """Initialize all feature generators"""
        
        # Technical features
        tech_config = self.config.get('technical', {})
        self.technical = TechnicalFeatures(config=tech_config)
        self.feature_generators.append(self.technical)
        
        # Regime features
        self.regime = RegimeFeatures()
        self.feature_generators.append(self.regime)
        
        logger.info(f"Initialized {len(self.feature_generators)} feature generators")
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from raw OHLCV data
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            DataFrame with all calculated features
        """
        logger.info(f"Generating features for {len(data)} rows")
        
        result = data.copy()
        feature_count = 0
        
        # Apply each feature generator
        for generator in self.feature_generators:
            try:
                logger.info(f"Applying {generator.name}...")
                result = generator.calculate(result)
                feature_count += len(generator.get_feature_names())
                self.all_feature_names.extend(generator.get_feature_names())
            except Exception as e:
                logger.error(f"Error in {generator.name}: {str(e)}")
                raise
        
        logger.info(f"Generated total of {feature_count} features")
        
        # Remove NaN rows (from indicator calculations)
        initial_rows = len(result)
        result = result.dropna()
        dropped_rows = initial_rows - len(result)
        
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with NaN values")
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get all generated feature names"""
        return self.all_feature_names
    
    def get_feature_metadata(self) -> Dict:
        """Get metadata about all features"""
        metadata = {
            'total_features': len(self.all_feature_names),
            'generators': []
        }
        
        for generator in self.feature_generators:
            metadata['generators'].append(generator.get_metadata())
        
        return metadata
    
    def get_features_by_category(self) -> Dict[str, List[str]]:
        """Get features organized by category"""
        categories = {}
        
        for generator in self.feature_generators:
            categories[generator.name] = generator.get_feature_names()
        
        return categories