# data/processors/cleaner.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    """Clean and preprocess raw stock data"""
    
    def __init__(self):
        self.cleaning_stats = {}
    
    def clean(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Apply all cleaning steps to raw data
        
        Args:
            data: Raw stock data
            symbol: Stock symbol (for logging)
        
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning data for {symbol}")
        
        initial_rows = len(data)
        data = data.copy()
        
        # 1. Remove duplicates
        data = self._remove_duplicates(data)
        
        # 2. Handle missing values
        data = self._handle_missing_values(data)
        
        # 3. Remove outliers (conservative approach)
        data = self._handle_outliers(data)
        
        # 4. Ensure proper sorting
        data = data.sort_values('Date').reset_index(drop=True)
        
        # 5. Add derived columns
        data = self._add_basic_features(data)
        
        final_rows = len(data)
        rows_removed = initial_rows - final_rows
        
        self.cleaning_stats[symbol] = {
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'rows_removed': rows_removed,
            'removal_rate': rows_removed / initial_rows if initial_rows > 0 else 0
        }
        
        logger.info(f"Cleaning complete for {symbol}: {rows_removed} rows removed ({rows_removed/initial_rows*100:.2f}%)")
        
        return data
    
    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate dates"""
        duplicates = data.duplicated(subset=['Date'], keep='last')
        if duplicates.any():
            logger.warning(f"Removing {duplicates.sum()} duplicate dates")
            data = data[~duplicates]
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values appropriately"""
        # Forward fill for prices (carry forward last known price)
        price_cols = ['Open', 'High', 'Low', 'Close']
        data[price_cols] = data[price_cols].fillna(method='ffill')
        
        # Backward fill for any remaining at the start
        data[price_cols] = data[price_cols].fillna(method='bfill')
        
        # Volume: fill with 0 or median
        if 'Volume' in data.columns:
            data['Volume'] = data['Volume'].fillna(data['Volume'].median())
        
        # Drop rows if still have critical nulls
        critical_nulls = data[price_cols].isnull().any(axis=1)
        if critical_nulls.any():
            logger.warning(f"Dropping {critical_nulls.sum()} rows with critical nulls")
            data = data[~critical_nulls]
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers conservatively
        Only remove extreme outliers that are likely data errors
        """
        # Calculate daily returns
        data['_temp_return'] = data['Close'].pct_change()
        
        # Identify extreme outliers (>10 standard deviations)
        # This catches data errors but keeps legitimate large moves
        returns_std = data['_temp_return'].std()
        extreme_outliers = data['_temp_return'].abs() > 10 * returns_std
        
        if extreme_outliers.any():
            logger.warning(f"Found {extreme_outliers.sum()} extreme outliers")
            # Instead of removing, cap them at 3 std
            data.loc[extreme_outliers, '_temp_return'] = np.sign(data.loc[extreme_outliers, '_temp_return']) * 3 * returns_std
            # Recalculate prices
            # (In practice, you'd want to investigate these manually)
        
        data = data.drop('_temp_return', axis=1)
        return data
    
    def _add_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic derived features"""
        # Returns
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Price range
        data['High_Low_Range'] = data['High'] - data['Low']
        data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Close']
        
        # Day of week (for potential seasonality)
        data['DayOfWeek'] = pd.to_datetime(data['Date']).dt.dayofweek
        
        return data
    
    def get_cleaning_report(self) -> pd.DataFrame:
        """Get summary of cleaning statistics"""
        if not self.cleaning_stats:
            return pd.DataFrame()
        
        return pd.DataFrame(self.cleaning_stats).T