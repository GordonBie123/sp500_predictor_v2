# data/fetchers/base_fetcher.py
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import pandas as pd
from datetime import datetime
import hashlib
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BaseFetcher(ABC):
    """Abstract base class for all data fetchers"""
    
    def __init__(self, cache_enabled: bool = True, cache_dir: str = "data/cache"):
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def fetch(self, symbol: str, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """Fetch data for a given symbol and date range"""
        pass
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate fetched data"""
        pass
    
    def get_cached(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Retrieve data from cache"""
        if not self.cache_enabled:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check if cache is still valid (24 hours)
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age.total_seconds() < 24 * 3600:
                    logger.info(f"Cache hit: {cache_key}")
                    return cached_data
                else:
                    logger.info(f"Cache expired: {cache_key}")
            except Exception as e:
                logger.warning(f"Error reading cache: {e}")
        
        return None
    
    def save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """Save data to cache"""
        if not self.cache_enabled:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    def _generate_cache_key(self, symbol: str, start_date: datetime, end_date: datetime, **kwargs) -> str:
        """Generate a unique cache key"""
        key_string = f"{self.__class__.__name__}_{symbol}_{start_date.date()}_{end_date.date()}_{kwargs}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def fetch_with_cache(self, symbol: str, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """Fetch data with caching support"""
        cache_key = self._generate_cache_key(symbol, start_date, end_date, **kwargs)
        
        # Try to get from cache
        cached_data = self.get_cached(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Fetch fresh data
        logger.info(f"Fetching fresh data for {symbol}")
        data = self.fetch(symbol, start_date, end_date, **kwargs)
        
        # Validate
        if not self.validate(data):
            raise ValueError(f"Data validation failed for {symbol}")
        
        # Cache the data
        self.save_to_cache(cache_key, data)
        
        return data
