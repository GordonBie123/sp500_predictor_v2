# config/__init__.py
import yaml
from pathlib import Path
from typing import Any, Dict
import os

class Config:
    """Configuration manager for the project"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._setup_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load environment variables for sensitive data
        config['api_keys'] = {
            'news_api': os.getenv('NEWS_API_KEY', ''),
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_KEY', '')
        }
        
        return config
    
    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            'data/cache',
            'logs',
            'models/saved',
            'outputs/predictions',
            'outputs/figures'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)
    
    @property
    def data_config(self):
        return self._config.get('data', {})
    
    @property
    def features_config(self):
        return self._config.get('features', {})
    
    @property
    def training_config(self):
        return self._config.get('training', {})

# Global config instance
config = Config()