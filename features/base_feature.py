# features/base_feature.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class BaseFeature(ABC):
    """Abstract base class for all feature generators"""
    
    def __init__(self, name: str):
        self.name = name
        self.feature_names = []
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features and add them to the dataframe
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            DataFrame with added feature columns
        """
        pass
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Return list of required columns in input data"""
        pass
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate that input data has required columns"""
        required = self.get_required_columns()
        missing = set(required) - set(data.columns)
        
        if missing:
            logger.error(f"{self.name}: Missing required columns: {missing}")
            return False
        
        return True
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names this generator creates"""
        return self.feature_names
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about this feature generator"""
        return {
            'name': self.name,
            'feature_count': len(self.feature_names),
            'features': self.feature_names,
            'required_columns': self.get_required_columns()
        }