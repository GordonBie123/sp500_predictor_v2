# features/regime_features.py
import pandas as pd
import numpy as np
from typing import List
from .base_feature import BaseFeature
import logging

logger = logging.getLogger(__name__)

class RegimeFeatures(BaseFeature):
    """
    Detect market regimes: volatility regimes, trend regimes, correlation regimes
    """
    
    def __init__(self):
        super().__init__(name="RegimeFeatures")
    
    def get_required_columns(self) -> List[str]:
        return ['Close', 'High', 'Low', 'Volume']
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market regime features"""
        
        if not self.validate_input(data):
            raise ValueError("Input validation failed")
        
        data = data.copy()
        logger.info(f"Calculating regime features for {len(data)} rows")
        
        data = self._add_volatility_regime(data)
        data = self._add_trend_regime(data)
        data = self._add_volume_regime(data)
        
        logger.info(f"Generated {len(self.feature_names)} regime features")
        
        return data
    
    def _add_volatility_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """Classify volatility regime (low, medium, high)"""
        
        # Calculate rolling volatility
        window = 30
        data['Rolling_volatility'] = data['Close'].pct_change().rolling(window=window).std()
        
        # Calculate percentile rank
        data['Volatility_percentile'] = data['Rolling_volatility'].rolling(
            window=252, min_periods=50
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        # Classify regime
        data['Volatility_regime'] = pd.cut(
            data['Volatility_percentile'],
            bins=[0, 0.33, 0.67, 1.0],
            labels=[0, 1, 2],  # 0=Low, 1=Medium, 2=High
            include_lowest=True
        ).astype(float)
        
        # Binary indicators
        data['Low_volatility'] = (data['Volatility_regime'] == 0).astype(int)
        data['High_volatility'] = (data['Volatility_regime'] == 2).astype(int)
        
        self.feature_names.extend([
            'Rolling_volatility', 'Volatility_percentile', 'Volatility_regime',
            'Low_volatility', 'High_volatility'
        ])
        
        return data
    
    def _add_trend_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """Classify trend regime (uptrend, downtrend, ranging)"""
        
        # Calculate trend using multiple moving averages
        data['SMA_short'] = data['Close'].rolling(window=20).mean()
        data['SMA_long'] = data['Close'].rolling(window=50).mean()
        
        # Trend strength
        data['Trend_strength'] = (data['SMA_short'] - data['SMA_long']) / data['SMA_long'] * 100
        
        # Classify trend regime
        def classify_trend(strength):
            if strength > 2:
                return 1  # Strong uptrend
            elif strength < -2:
                return -1  # Strong downtrend
            else:
                return 0  # Ranging
        
        data['Trend_regime'] = data['Trend_strength'].apply(classify_trend)
        
        # Binary indicators
        data['Uptrend'] = (data['Trend_regime'] == 1).astype(int)
        data['Downtrend'] = (data['Trend_regime'] == -1).astype(int)
        data['Ranging'] = (data['Trend_regime'] == 0).astype(int)
        
        # Trend persistence (how long in current trend)
        data['Trend_persistence'] = (
            data['Trend_regime'].groupby(
                (data['Trend_regime'] != data['Trend_regime'].shift()).cumsum()
            ).cumcount() + 1
        )
        
        self.feature_names.extend([
            'Trend_strength', 'Trend_regime', 'Uptrend', 'Downtrend', 'Ranging',
            'Trend_persistence'
        ])
        
        # Drop temporary columns
        data = data.drop(['SMA_short', 'SMA_long'], axis=1)
        
        return data
    
    def _add_volume_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """Classify volume regime"""
        
        # Volume percentile
        data['Volume_percentile'] = data['Volume'].rolling(
            window=60, min_periods=20
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        # Volume regime
        data['Volume_regime'] = pd.cut(
            data['Volume_percentile'],
            bins=[0, 0.33, 0.67, 1.0],
            labels=[0, 1, 2],  # 0=Low, 1=Normal, 2=High
            include_lowest=True
        ).astype(float)
        
        # Binary indicators
        data['Low_volume'] = (data['Volume_regime'] == 0).astype(int)
        data['High_volume'] = (data['Volume_regime'] == 2).astype(int)
        
        self.feature_names.extend([
            'Volume_percentile', 'Volume_regime', 'Low_volume', 'High_volume'
        ])
        
        return data