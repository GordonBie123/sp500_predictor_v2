# data/fetchers/stock_fetcher.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
from .base_fetcher import BaseFetcher

logger = logging.getLogger(__name__)

class StockDataFetcher(BaseFetcher):
    """Fetch stock price data from Yahoo Finance"""
    
    def __init__(self, cache_enabled: bool = True):
        super().__init__(cache_enabled=cache_enabled)
        self.required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    def fetch(self, symbol: str, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            **kwargs: Additional arguments (interval, etc.)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Add buffer to start date to account for weekends/holidays
            buffer_days = 10
            start_with_buffer = start_date - timedelta(days=buffer_days)
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_with_buffer,
                end=end_date,
                interval=kwargs.get('interval', '1d'),
                auto_adjust=True  # Adjust for splits and dividends
            )
            
            if data.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # Reset index to have Date as column
            data.reset_index(inplace=True)
            
            # FIX: Remove timezone information to avoid comparison issues
            if 'Date' in data.columns:
                # Convert to timezone-naive datetime
                data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
            
            # Filter to requested date range (now both are timezone-naive)
            data = data[data['Date'] >= start_date]
            
            # Add symbol column
            data['Symbol'] = symbol
            
            # Ensure proper column order
            cols = ['Date', 'Symbol'] + self.required_columns
            data = data[cols]
            
            logger.info(f"Fetched {len(data)} rows for {symbol} from {data['Date'].min()} to {data['Date'].max()}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def validate(self, data: pd.DataFrame) -> bool:
        """
        Validate the fetched data
        
        Args:
            data: DataFrame to validate
        
        Returns:
            True if valid, False otherwise
        """
        if data.empty:
            logger.error("Data is empty")
            return False
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(data.columns)
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            return False
        
        # Check for null values in critical columns
        null_counts = data[self.required_columns].isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
            # Allow some nulls but not too many
            if (null_counts / len(data)).max() > 0.1:
                logger.error("Too many null values (>10%)")
                return False
        
        # Check for reasonable values
        if (data['Close'] <= 0).any():
            logger.error("Found non-positive prices")
            return False
        
        if (data['Volume'] < 0).any():
            logger.error("Found negative volume")
            return False
        
        # Check for suspicious price movements (>50% in one day)
        price_changes = data['Close'].pct_change().abs()
        if (price_changes > 0.5).any():
            suspicious_dates = data[price_changes > 0.5]['Date'].tolist()
            logger.warning(f"Large price movements detected on: {suspicious_dates}")
            # Don't fail validation, just warn
        
        # Check data continuity (no large gaps)
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            date_diffs = data['Date'].diff()
            large_gaps = date_diffs[date_diffs > timedelta(days=7)]
            if not large_gaps.empty:
                logger.warning(f"Found {len(large_gaps)} gaps larger than 7 days")
        
        logger.info("Data validation passed")
        return True
    
    def fetch_realtime(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch real-time quote data
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dictionary with current price info
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'current_price': info.get('currentPrice', info.get('regularMarketPrice')),
                'previous_close': info.get('previousClose'),
                'open': info.get('open', info.get('regularMarketOpen')),
                'day_high': info.get('dayHigh', info.get('regularMarketDayHigh')),
                'day_low': info.get('dayLow', info.get('regularMarketDayLow')),
                'volume': info.get('volume', info.get('regularMarketVolume')),
                'market_cap': info.get('marketCap'),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error fetching realtime data for {symbol}: {str(e)}")
            raise
    
    def fetch_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch company information
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dictionary with company info
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'description': info.get('longBusinessSummary', ''),
                'employees': info.get('fullTimeEmployees', 0),
                'website': info.get('website', ''),
                'market_cap': info.get('marketCap', 0)
            }
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'name': symbol,
                'sector': 'Unknown',
                'industry': 'Unknown'
            }