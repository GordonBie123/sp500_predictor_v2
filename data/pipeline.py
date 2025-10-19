# data/pipeline.py
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import logging
from .fetchers.stock_fetcher import StockDataFetcher
from .processors.cleaner import DataCleaner
from config import config

logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Orchestrates the entire data fetching and processing pipeline
    """
    
    def __init__(self):
        self.stock_fetcher = StockDataFetcher(
            cache_enabled=config.get('data.cache.enabled', True)
        )
        self.cleaner = DataCleaner()
        self.data_store = {}
    
    def fetch_and_process(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Complete pipeline: fetch, clean, and process data
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date (defaults to 1 year ago)
            end_date: End date (defaults to today)
            force_refresh: Force refresh from source (ignore cache)
        
        Returns:
            Processed DataFrame ready for feature engineering
        """
        logger.info(f"Starting data pipeline for {symbol}")
        
        # Set default dates
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            lookback_days = config.get('data.lookback_days', 252)
            start_date = end_date - timedelta(days=lookback_days * 1.5)  # Extra buffer
        
        try:
            # Step 1: Fetch raw data
            logger.info(f"Step 1: Fetching data for {symbol}")
            if force_refresh:
                raw_data = self.stock_fetcher.fetch(symbol, start_date, end_date)
            else:
                raw_data = self.stock_fetcher.fetch_with_cache(symbol, start_date, end_date)
            
            # Step 2: Clean data
            logger.info(f"Step 2: Cleaning data for {symbol}")
            cleaned_data = self.cleaner.clean(raw_data, symbol)
            
            # Step 3: Store in memory
            self.data_store[symbol] = {
                'data': cleaned_data,
                'last_updated': datetime.now(),
                'start_date': cleaned_data['Date'].min(),
                'end_date': cleaned_data['Date'].max()
            }
            
            logger.info(f"Pipeline complete for {symbol}: {len(cleaned_data)} rows")
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Pipeline failed for {symbol}: {str(e)}")
            raise
    
    def get_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get cached data for a symbol"""
        if symbol in self.data_store:
            return self.data_store[symbol]['data']
        return None
    
    def get_pipeline_status(self) -> Dict:
        """Get status of all symbols in the pipeline"""
        status = {}
        for symbol, info in self.data_store.items():
            status[symbol] = {
                'rows': len(info['data']),
                'last_updated': info['last_updated'],
                'date_range': f"{info['start_date']} to {info['end_date']}"
            }
        return status