# models/__init__.py (update)
from .base_model import BaseModel
from .lstm_model import LSTMModel
from .tree_models import RandomForestModel, XGBoostModel
from .ensemble import EnsembleModel

__all__ = ['BaseModel', 'LSTMModel', 'RandomForestModel', 'XGBoostModel', 'EnsembleModel']