# Stock Price Predictor V2

An advanced stock price prediction system using ensemble machine learning models with enhanced visualization and real-time predictions.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## Features

### Core Functionality
- **Multi-Model Ensemble**: Combines LSTM, Random Forest, and XGBoost for robust predictions
- **70+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, Moving Averages, and more
- **Market Regime Detection**: Automatically identifies volatility and trend regimes
- **Real-time Data**: Fetches latest stock data from Yahoo Finance
- **Interactive Web Interface**: Built with Streamlit for easy use

### Advanced Features
- **Confidence Intervals**: 95% prediction intervals for uncertainty quantification
- **Future Forecasts**: Predict stock prices up to 90 days ahead
- **Buy/Sell/Hold Recommendations**: AI-generated trading signals
- **Backtesting Framework**: Validate model performance on historical data
- **Model Comparison**: Compare performance across different models
- **Enhanced Visualizations**: Interactive charts with Plotly

## Architecture
```
stock-predictor-v2/
├── config/              # Configuration files
├── data/                # Data fetching and processing
│   ├── fetchers/        # Stock data fetchers
│   └── processors/      # Data cleaning and validation
├── features/            # Feature engineering
│   ├── technical_features.py    # 70+ technical indicators
│   ├── regime_features.py       # Market regime detection
│   └── feature_store.py         # Feature management
├── models/              # Machine learning models
│   ├── lstm_model.py    # LSTM neural network
│   ├── tree_models.py   # Random Forest & XGBoost
│   └── ensemble.py      # Ensemble model
├── training/            # Training pipeline
├── utils/               # Utility functions
│   ├── predictor.py     # Prediction engine
│   └── visualizations.py # Chart generation
├── tests/               # Unit tests
└── app.py              # Streamlit web application
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/stock-predictor-v2.git
cd stock-predictor-v2
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Create configuration file**
```bash
# Copy the example config
cp config/config.yaml.example config/config.yaml

# Edit with your settings (optional)
```

## Usage

### Running the Web Application
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### Basic Workflow

1. **Select a Stock**: Choose from S&P 500 stocks in the sidebar
2. **Configure Model**: Set lookback days and prediction horizon
3. **Train Model**: Click "Train Model" (takes 2-5 minutes)
4. **Generate Predictions**: Navigate to Predictions tab
5. **View Results**: See forecasts, confidence intervals, and recommendations

### Command-Line Usage

#### Train a Model
```python
from training.trainer import ModelTrainer

config = {
    'lookback_days': 30,
    'prediction_horizon': 5,
    'train_split': 0.7,
    'val_split': 0.15
}

trainer = ModelTrainer(config=config)
results = trainer.train(symbol='AAPL', model_type='ensemble')
trainer.save_model('models/saved/aapl_model')
```

#### Make Predictions
```python
from utils.predictor import PredictionEngine

predictor = PredictionEngine(trainer)
predictions = predictor.predict_future('AAPL', n_days=30)

print(f"Predicted price: ${predictions['predictions'][-1]:.2f}")
print(f"Recommendation: {predictions['recommendation']['action']}")
```

## Models

### LSTM Neural Network
- **Architecture**: 3-layer LSTM with dropout
- **Best for**: Long-term trends and sequential patterns
- **Training time**: 3-5 minutes
- **Accuracy**: High on trending markets

### Random Forest
- **Architecture**: 200 decision trees
- **Best for**: Non-linear relationships and feature importance
- **Training time**: 30-60 seconds
- **Accuracy**: Consistent across market conditions

### XGBoost
- **Architecture**: Gradient boosting with 200 estimators
- **Best for**: Complex interactions and stability
- **Training time**: 1-2 minutes
- **Accuracy**: Very high on structured data

### Ensemble Model
- **Architecture**: Weighted combination of all models
- **Weights**: Automatically learned based on validation performance
- **Best for**: Maximum accuracy and robustness
- **Training time**: 5-10 minutes

## Technical Indicators

The system generates 70+ technical features including:

### Momentum Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Rate of Change (ROC)

### Volatility Indicators
- Bollinger Bands
- Average True Range (ATR)
- Historical Volatility

### Volume Indicators
- On-Balance Volume (OBV)
- Volume-Weighted Average Price (VWAP)
- Money Flow Index (MFI)

### Trend Indicators
- Simple Moving Averages (10, 20, 50, 200-day)
- Exponential Moving Averages (12, 26-day)
- ADX (Average Directional Index)
- Aroon Indicator

## Performance

### Typical Metrics
- **R-squared**: 0.70 - 0.90
- **MAPE**: 2% - 5%
- **Directional Accuracy**: 65% - 75%

### Benchmarks (on AAPL, 1-year data)
| Model | Training Time | R² Score | RMSE | MAE |
|-------|--------------|----------|------|-----|
| LSTM | 3-5 min | 0.85 | 2.34 | 1.89 |
| Random Forest | 45 sec | 0.78 | 2.89 | 2.15 |
| XGBoost | 90 sec | 0.82 | 2.51 | 1.97 |
| **Ensemble** | **6 min** | **0.88** | **2.12** | **1.73** |

## Testing

Run the test suite:
```bash
# Test data pipeline
python tests/test_pipeline.py

# Test feature generation
python tests/test_features.py

# Test models
python tests/test_models.py

# Test complete training pipeline
python tests/test_training.py
```

## Configuration

Edit `config/config.yaml` to customize:
```yaml
data:
  lookback_days: 60
  prediction_horizon: 5
  train_split: 0.7
  val_split: 0.15

features:
  technical:
    - name: "sma"
      windows: [10, 20, 50, 200]
    - name: "rsi"
      period: 14

training:
  batch_size: 32
  epochs: 50
  early_stopping_patience: 15
```

## API Reference

### ModelTrainer
```python
trainer = ModelTrainer(config=dict)
trainer.train(symbol='AAPL', model_type='ensemble')
trainer.save_model(path='models/saved/model_name')
```

### PredictionEngine
```python
predictor = PredictionEngine(trainer)
predictions = predictor.predict_future(symbol='AAPL', n_days=30)
backtest = predictor.backtest_predictions(symbol='AAPL', test_days=30)
```

### FeatureStore
```python
from features.feature_store import FeatureStore

feature_store = FeatureStore()
features = feature_store.generate_features(raw_data)
feature_names = feature_store.get_feature_names()
```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'X'`
```bash
pip install -r requirements.txt
```

**Issue**: "Not enough data after feature generation"
```python
# Increase lookback_months when training
trainer.prepare_data(symbol='AAPL', lookback_months=24)
```

**Issue**: LSTM training is slow
```python
# Reduce epochs or use Random Forest
config = {'lookback_days': 30}  # Reduce from 60
# OR
trainer.train(symbol='AAPL', model_type='rf')  # Faster model
```

## Limitations and Disclaimers

⚠️ **Important**: This tool is for **educational purposes only**.

- Stock market predictions are inherently uncertain
- Past performance does not guarantee future results
- This system does not account for:
  - Breaking news or unexpected events
  - Macroeconomic policy changes
  - Company-specific events (earnings surprises, management changes)
  - Market sentiment shifts

**Always consult with qualified financial advisors before making investment decisions.**

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Roadmap

### Future Enhancements
- [ ] Sentiment analysis from news articles
- [ ] Multi-stock portfolio optimization
- [ ] Real-time predictions with WebSocket
- [ ] Email alerts for buy/sell signals
- [ ] Mobile app version
- [ ] Options pricing predictions
- [ ] Cryptocurrency support
- [ ] Advanced hyperparameter tuning
- [ ] Model interpretability dashboard

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Yahoo Finance for providing free stock data via `yfinance`
- TensorFlow team for the LSTM implementation
- Scikit-learn and XGBoost contributors
- Streamlit for the amazing web framework
- The open-source community