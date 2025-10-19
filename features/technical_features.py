# features/technical_features.py
import pandas as pd
import numpy as np
from typing import List, Dict
import ta
from .base_feature import BaseFeature
import logging

logger = logging.getLogger(__name__)

class TechnicalFeatures(BaseFeature):
    """
    Generate technical analysis features
    Includes: Moving averages, momentum indicators, volatility, volume indicators
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(name="TechnicalFeatures")
        
        # Default configuration
        self.config = config or {
            'sma_periods': [10, 20, 50, 200],
            'ema_periods': [12, 26],
            'rsi_period': 14,
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger': {'period': 20, 'std': 2},
            'atr_period': 14,
            'adx_period': 14
        }
        
    def get_required_columns(self) -> List[str]:
        return ['Open', 'High', 'Low', 'Close', 'Volume']
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        
        if not self.validate_input(data):
            raise ValueError("Input validation failed")
        
        data = data.copy()
        logger.info(f"Calculating technical features for {len(data)} rows")
        
        # Calculate each group of features
        data = self._add_moving_averages(data)
        data = self._add_momentum_indicators(data)
        data = self._add_volatility_indicators(data)
        data = self._add_volume_indicators(data)
        data = self._add_price_patterns(data)
        data = self._add_trend_indicators(data)
        
        logger.info(f"Generated {len(self.feature_names)} technical features")
        
        return data
    
    def _add_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add Simple and Exponential Moving Averages"""
        
        # Simple Moving Averages
        for period in self.config['sma_periods']:
            col_name = f'SMA_{period}'
            data[col_name] = ta.trend.sma_indicator(data['Close'], window=period)
            self.feature_names.append(col_name)
        
        # Exponential Moving Averages
        for period in self.config['ema_periods']:
            col_name = f'EMA_{period}'
            data[col_name] = ta.trend.ema_indicator(data['Close'], window=period)
            self.feature_names.append(col_name)
        
        # Price relative to moving averages
        data['Price_to_SMA20'] = data['Close'] / data['SMA_20']
        data['Price_to_SMA50'] = data['Close'] / data['SMA_50']
        self.feature_names.extend(['Price_to_SMA20', 'Price_to_SMA50'])
        
        # Moving average crossovers
        data['SMA_10_20_cross'] = (data['SMA_10'] > data['SMA_20']).astype(int)
        data['SMA_20_50_cross'] = (data['SMA_20'] > data['SMA_50']).astype(int)
        self.feature_names.extend(['SMA_10_20_cross', 'SMA_20_50_cross'])
        
        return data
    
    def _add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators (RSI, MACD, Stochastic, etc.)"""
        
        # RSI
        rsi_period = self.config['rsi_period']
        data['RSI'] = ta.momentum.rsi(data['Close'], window=rsi_period)
        self.feature_names.append('RSI')
        
        # RSI categories
        data['RSI_oversold'] = (data['RSI'] < 30).astype(int)
        data['RSI_overbought'] = (data['RSI'] > 70).astype(int)
        self.feature_names.extend(['RSI_oversold', 'RSI_overbought'])
        
        # MACD
        macd_config = self.config['macd']
        macd = ta.trend.MACD(
            data['Close'],
            window_fast=macd_config['fast'],
            window_slow=macd_config['slow'],
            window_sign=macd_config['signal']
        )
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['MACD_diff'] = macd.macd_diff()
        self.feature_names.extend(['MACD', 'MACD_signal', 'MACD_diff'])
        
        # MACD signal
        data['MACD_bullish'] = (data['MACD'] > data['MACD_signal']).astype(int)
        self.feature_names.append('MACD_bullish')
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            data['High'], data['Low'], data['Close']
        )
        data['Stoch_K'] = stoch.stoch()
        data['Stoch_D'] = stoch.stoch_signal()
        self.feature_names.extend(['Stoch_K', 'Stoch_D'])
        
        # Rate of Change (ROC)
        data['ROC_10'] = ta.momentum.roc(data['Close'], window=10)
        data['ROC_20'] = ta.momentum.roc(data['Close'], window=20)
        self.feature_names.extend(['ROC_10', 'ROC_20'])
        
        return data
    
    def _add_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        
        # Bollinger Bands
        bb_config = self.config['bollinger']
        bb = ta.volatility.BollingerBands(
            data['Close'],
            window=bb_config['period'],
            window_dev=bb_config['std']
        )
        data['BB_upper'] = bb.bollinger_hband()
        data['BB_middle'] = bb.bollinger_mavg()
        data['BB_lower'] = bb.bollinger_lband()
        data['BB_width'] = bb.bollinger_wband()
        data['BB_pct'] = bb.bollinger_pband()  # % position within bands
        self.feature_names.extend(['BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'BB_pct'])
        
        # Bollinger Band indicators
        data['Price_above_BB_upper'] = (data['Close'] > data['BB_upper']).astype(int)
        data['Price_below_BB_lower'] = (data['Close'] < data['BB_lower']).astype(int)
        self.feature_names.extend(['Price_above_BB_upper', 'Price_below_BB_lower'])
        
        # Average True Range (ATR)
        atr_period = self.config['atr_period']
        data['ATR'] = ta.volatility.average_true_range(
            data['High'], data['Low'], data['Close'], window=atr_period
        )
        data['ATR_pct'] = data['ATR'] / data['Close'] * 100
        self.feature_names.extend(['ATR', 'ATR_pct'])
        
        # Historical volatility (standard deviation of returns)
        data['Volatility_10'] = data['Close'].pct_change().rolling(window=10).std()
        data['Volatility_20'] = data['Close'].pct_change().rolling(window=20).std()
        data['Volatility_30'] = data['Close'].pct_change().rolling(window=30).std()
        self.feature_names.extend(['Volatility_10', 'Volatility_20', 'Volatility_30'])
        
        return data
    
    def _add_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        
        # Volume moving average
        data['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()
        data['Volume_ratio'] = data['Volume'] / data['Volume_SMA_20']
        self.feature_names.extend(['Volume_SMA_20', 'Volume_ratio'])
        
        # Volume spike detection
        data['Volume_spike'] = (data['Volume_ratio'] > 2.0).astype(int)
        self.feature_names.append('Volume_spike')
        
        # On-Balance Volume (OBV)
        data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
        self.feature_names.append('OBV')
        
        # Volume-Weighted Average Price (VWAP)
        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        data['Price_to_VWAP'] = data['Close'] / data['VWAP']
        self.feature_names.extend(['VWAP', 'Price_to_VWAP'])
        
        # Money Flow Index (MFI)
        data['MFI'] = ta.volume.money_flow_index(
            data['High'], data['Low'], data['Close'], data['Volume'], window=14
        )
        self.feature_names.append('MFI')
        
        return data
    
    def _add_price_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern features"""
        
        # Price ranges
        data['High_Low_Range'] = data['High'] - data['Low']
        data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Close'] * 100
        self.feature_names.extend(['High_Low_Range', 'High_Low_Pct'])
        
        # Price position in daily range
        data['Close_position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
        self.feature_names.append('Close_position')
        
        # Gap detection
        data['Gap_up'] = (data['Open'] > data['High'].shift(1)).astype(int)
        data['Gap_down'] = (data['Open'] < data['Low'].shift(1)).astype(int)
        self.feature_names.extend(['Gap_up', 'Gap_down'])
        
        # Candle body and shadows
        data['Body'] = abs(data['Close'] - data['Open'])
        data['Upper_shadow'] = data['High'] - np.maximum(data['Open'], data['Close'])
        data['Lower_shadow'] = np.minimum(data['Open'], data['Close']) - data['Low']
        self.feature_names.extend(['Body', 'Upper_shadow', 'Lower_shadow'])
        
        # Doji detection (small body)
        data['Is_doji'] = (data['Body'] < data['High_Low_Range'] * 0.1).astype(int)
        self.feature_names.append('Is_doji')
        
        return data
    
    def _add_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add trend strength and direction indicators"""
        
        # ADX (Average Directional Index) - trend strength
        adx_period = self.config['adx_period']
        data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=adx_period)
        self.feature_names.append('ADX')
        
        # Trend classification based on ADX
        data['Strong_trend'] = (data['ADX'] > 25).astype(int)
        data['Weak_trend'] = (data['ADX'] < 20).astype(int)
        self.feature_names.extend(['Strong_trend', 'Weak_trend'])
        
        # Aroon Indicator
        aroon = ta.trend.AroonIndicator(data['High'], data['Low'], window=25)
        data['Aroon_up'] = aroon.aroon_up()
        data['Aroon_down'] = aroon.aroon_down()
        data['Aroon_indicator'] = data['Aroon_up'] - data['Aroon_down']
        self.feature_names.extend(['Aroon_up', 'Aroon_down', 'Aroon_indicator'])
        
        # Price momentum (multiple timeframes)
        data['Momentum_5'] = data['Close'].pct_change(periods=5)
        data['Momentum_10'] = data['Close'].pct_change(periods=10)
        data['Momentum_20'] = data['Close'].pct_change(periods=20)
        self.feature_names.extend(['Momentum_5', 'Momentum_10', 'Momentum_20'])
        
        return data