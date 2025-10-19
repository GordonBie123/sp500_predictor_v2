# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import logging

from training.trainer import ModelTrainer
from data.pipeline import DataPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor V2",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 10px;
        border-radius: 5px;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# S&P 500 popular stocks
POPULAR_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
    'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC',
    'ABBV', 'PFE', 'LLY', 'WMT', 'DIS', 'CSCO', 'NFLX', 'COST'
]

class StockPredictorApp:
    """Main Streamlit application"""
    
    def __init__(self):
        """Initialize the app"""
        # Initialize session state
        if 'trained_model' not in st.session_state:
            st.session_state.trained_model = None
        if 'trainer' not in st.session_state:
            st.session_state.trainer = None
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None
        if 'current_symbol' not in st.session_state:
            st.session_state.current_symbol = 'AAPL'
        if 'latest_data' not in st.session_state:
            st.session_state.latest_data = None
    
    def run(self):
        """Run the main application"""
        # Header
        st.title("Stock Price Predictor V2")
        st.markdown("*Advanced ML-based stock price prediction with ensemble models*")
        
        # Sidebar
        with st.sidebar:
            st.header("Configuration")
            
            # Stock selection
            symbol = st.selectbox(
                "Select Stock Symbol",
                POPULAR_STOCKS,
                index=POPULAR_STOCKS.index(st.session_state.current_symbol)
            )
            
            # Update if changed
            if symbol != st.session_state.current_symbol:
                st.session_state.current_symbol = symbol
                st.session_state.trained_model = None
                st.session_state.predictions = None
                st.session_state.latest_data = None
            
            st.markdown("---")
            
            # Model configuration
            st.subheader("Model Settings")
            
            model_type = st.selectbox(
                "Model Type",
                ["ensemble", "lstm", "rf", "xgb"],
                format_func=lambda x: {
                    "ensemble": "Ensemble (Best)",
                    "lstm": "LSTM Neural Network",
                    "rf": "Random Forest",
                    "xgb": "XGBoost"
                }[x]
            )
            
            lookback_days = st.slider(
                "Lookback Days",
                min_value=20,
                max_value=90,
                value=30,
                step=10,
                help="Number of historical days to consider"
            )
            
            prediction_horizon = st.slider(
                "Prediction Horizon (days)",
                min_value=1,
                max_value=30,
                value=5,
                step=1,
                help="How many days ahead to predict"
            )
            
            st.markdown("---")
            
            # Action buttons
            st.subheader("Actions")
            
            if st.button("Fetch Latest Data", use_container_width=True):
                self.fetch_data(symbol)
            
            if st.button("Train Model", type="primary", use_container_width=True):
                self.train_model(symbol, model_type, lookback_days, prediction_horizon)
            
            if st.button("Make Prediction", 
                        use_container_width=True,
                        disabled=st.session_state.trained_model is None):
                st.session_state.predictions = None  # Clear old predictions
            
            if st.button("Save Model", 
                        use_container_width=True,
                        disabled=st.session_state.trainer is None):
                self.save_model(symbol)
            
            st.markdown("---")
            
            # Model status
            st.subheader("Status")
            if st.session_state.trained_model:
                st.success("Model Trained")
                st.info(f"Model: {model_type.upper()}")
            else:
                st.warning("No Model Trained")
        
        # Main content area
        tabs = st.tabs(["Overview", "Predictions", "Performance", "About"])
        
        with tabs[0]:
            self.show_overview(symbol)
        
        with tabs[1]:
            self.show_predictions()
        
        with tabs[2]:
            self.show_performance()
        
        with tabs[3]:
            self.show_about()
    
    def fetch_data(self, symbol: str):
        """Fetch and display latest stock data"""
        with st.spinner(f"Fetching data for {symbol}..."):
            try:
                pipeline = DataPipeline()
                data = pipeline.fetch_and_process(symbol)
                
                # Store in session state
                st.session_state.latest_data = data
                
                st.success(f"Successfully fetched {len(data)} rows of data")
                
                # Display basic info
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Latest Close", f"${data['Close'].iloc[-1]:.2f}")
                
                with col2:
                    daily_change = data['Close'].pct_change().iloc[-1] * 100
                    st.metric("Daily Change", f"{daily_change:+.2f}%")
                
                with col3:
                    st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
                
                with col4:
                    st.metric("Data Points", len(data))
                
                # Show recent data
                st.subheader("Recent Data")
                st.dataframe(data.tail(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                logger.exception("Data fetch error")
    
    def train_model(self, symbol: str, model_type: str, lookback_days: int, prediction_horizon: int):
        """Train the selected model"""
        with st.spinner(f"Training {model_type.upper()} model... This may take several minutes..."):
            try:
                # Configure trainer
                config = {
                    'lookback_days': lookback_days,
                    'prediction_horizon': prediction_horizon,
                    'train_split': 0.7,
                    'val_split': 0.15
                }
                
                trainer = ModelTrainer(config=config)
                
                # Progress placeholder
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                progress_text.text("Fetching data...")
                progress_bar.progress(20)
                
                progress_text.text("Generating features...")
                progress_bar.progress(40)
                
                progress_text.text("Training model...")
                progress_bar.progress(60)
                
                # Train
                results = trainer.train(symbol=symbol, model_type=model_type)
                
                progress_bar.progress(100)
                progress_text.text("Training complete!")
                
                # Save to session state
                st.session_state.trainer = trainer
                st.session_state.trained_model = results
                
                # Clear progress
                progress_text.empty()
                progress_bar.empty()
                
                # Show results
                st.success("Model trained successfully")
                
                # Display metrics
                st.subheader("Training Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Training Time", f"{results['training_time']:.1f}s")
                
                with col2:
                    st.metric("Train Samples", results['n_train_samples'])
                
                with col3:
                    st.metric("Features", results['n_features'])
                
                # Test results
                if model_type == "ensemble":
                    st.subheader("Model Performance")
                    
                    for model_name, metrics in results['test_results'].items():
                        with st.expander(f"{model_name.upper()} Metrics"):
                            cols = st.columns(4)
                            cols[0].metric("R-squared", f"{metrics['r2']:.4f}")
                            cols[1].metric("RMSE", f"{metrics['rmse']:.4f}")
                            cols[2].metric("MAE", f"{metrics['mae']:.4f}")
                            cols[3].metric("MAPE", f"{metrics['mape']:.2f}%")
                else:
                    metrics = results['test_results'][model_type]
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("R-squared", f"{metrics['r2']:.4f}")
                    col2.metric("RMSE", f"{metrics['rmse']:.4f}")
                    col3.metric("MAE", f"{metrics['mae']:.4f}")
                    col4.metric("MAPE", f"{metrics['mape']:.2f}%")
                
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                logger.exception("Training error")
    
    def save_model(self, symbol: str):
        """Save the trained model"""
        if st.session_state.trainer is None:
            st.error("No model to save")
            return
        
        with st.spinner("Saving model..."):
            try:
                save_path = f"models/saved/{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.trainer.save_model(save_path)
                st.success(f"Model saved to {save_path}")
                
            except Exception as e:
                st.error(f"Error saving model: {str(e)}")
                logger.exception("Model save error")
    
    def show_overview(self, symbol: str):
        """Show stock overview"""
        st.header(f"{symbol} Overview")
        
        # Use cached data if available
        if st.session_state.latest_data is not None:
            data = st.session_state.latest_data
        else:
            try:
                with st.spinner("Loading data..."):
                    pipeline = DataPipeline()
                    data = pipeline.fetch_and_process(symbol)
                    st.session_state.latest_data = data
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return
        
        # Price chart
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=data['Date'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ))
        
        fig.update_layout(
            title=f'{symbol} Price History',
            yaxis_title='Price (USD)',
            xaxis_title='Date',
            height=500,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        fig_vol = go.Figure()
        
        fig_vol.add_trace(go.Bar(
            x=data['Date'],
            y=data['Volume'],
            name='Volume',
            marker_color='lightblue'
        ))
        
        fig_vol.update_layout(
            title='Trading Volume',
            yaxis_title='Volume',
            xaxis_title='Date',
            height=300
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # Statistics
        st.subheader("Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Price", f"${data['Close'].mean():.2f}")
            st.metric("Min Price", f"${data['Close'].min():.2f}")
        
        with col2:
            st.metric("Max Price", f"${data['Close'].max():.2f}")
            st.metric("Std Deviation", f"${data['Close'].std():.2f}")
        
        with col3:
            st.metric("Avg Volume", f"{data['Volume'].mean():,.0f}")
            st.metric("Total Days", len(data))
    
    def show_predictions(self):
        """Show prediction results with enhanced visualizations"""
        st.header("Predictions")
        
        if st.session_state.trained_model is None:
            st.info("Train a model first to see predictions. Use the sidebar to get started.")
            return
        
        if st.session_state.trainer is None:
            st.error("Trainer not available")
            return
        
        symbol = st.session_state.current_symbol
        
        # Prediction controls
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Generate Future Predictions")
        
        with col2:
            n_days = st.number_input(
                "Days to Predict",
                min_value=1,
                max_value=90,
                value=30,
                step=5
            )
        
        if st.button("Generate Predictions", type="primary"):
            with st.spinner("Generating predictions..."):
                try:
                    from utils.predictor import PredictionEngine
                    from utils.visualizations import (
                        create_prediction_chart,
                        create_recommendation_gauge
                    )
                    
                    # Create prediction engine
                    predictor = PredictionEngine(st.session_state.trainer)
                    
                    # Generate predictions
                    prediction_data = predictor.predict_future(symbol, n_days=n_days)
                    
                    # Store in session state
                    st.session_state.predictions = prediction_data
                    
                    st.success("Predictions generated successfully")
                    
                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
                    logger.exception("Prediction error")
                    return
        
        # Display predictions if available
        if st.session_state.predictions is not None:
            pred_data = st.session_state.predictions
            
            # Key metrics
            st.subheader("Prediction Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${pred_data['last_known_price']:.2f}"
                )
            
            with col2:
                final_price = pred_data['predictions'][-1]
                st.metric(
                    "Predicted Price",
                    f"${final_price:.2f}",
                    f"{pred_data['predicted_change']:+.2f}%"
                )
            
            with col3:
                st.metric(
                    "Expected Range",
                    f"${pred_data['min_price']:.2f} - ${pred_data['max_price']:.2f}"
                )
            
            with col4:
                st.metric(
                    "Volatility",
                    f"{pred_data['volatility']:.2f}%"
                )
            
            # Recommendation
            st.subheader("Recommendation")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                from utils.visualizations import create_recommendation_gauge
                gauge_fig = create_recommendation_gauge(pred_data['recommendation'])
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col2:
                st.markdown(f"### {pred_data['recommendation']['action']}")
                st.write(f"**Confidence:** {pred_data['recommendation']['confidence']}")
                st.write(f"**Reason:** {pred_data['recommendation']['reason']}")
                st.write(f"**Trend Strength:** {pred_data['recommendation']['trend_strength']:.2f}%")
            
            # Prediction chart
            st.subheader("Price Forecast")
            
            # Get historical data for context
            from data.pipeline import DataPipeline
            pipeline = DataPipeline()
            historical_data = pipeline.get_data(symbol)
            
            if historical_data is not None:
                # Show last 60 days
                historical_data = historical_data.tail(60)
            
            from utils.visualizations import create_prediction_chart
            pred_fig = create_prediction_chart(pred_data, historical_data)
            st.plotly_chart(pred_fig, use_container_width=True)
            
            # Prediction table
            with st.expander("View Detailed Predictions"):
                pred_df = pd.DataFrame({
                    'Date': pred_data['dates'],
                    'Predicted Price': [f"${p:.2f}" for p in pred_data['predictions']],
                    'Lower Bound': [f"${l:.2f}" for l in pred_data['lower_bounds']],
                    'Upper Bound': [f"${u:.2f}" for u in pred_data['upper_bounds']],
                    'Change from Current': [f"{((p - pred_data['last_known_price']) / pred_data['last_known_price'] * 100):+.2f}%" 
                                           for p in pred_data['predictions']]
                })
                st.dataframe(pred_df, use_container_width=True)
            
            # Backtest section
            st.markdown("---")
            st.subheader("Model Validation")
            
            if st.button("Run Backtest"):
                with st.spinner("Running backtest..."):
                    try:
                        from utils.predictor import PredictionEngine
                        from utils.visualizations import (
                            create_backtest_chart,
                            create_error_distribution_chart
                        )
                        
                        predictor = PredictionEngine(st.session_state.trainer)
                        backtest_data = predictor.backtest_predictions(symbol, test_days=30)
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        metrics = backtest_data['metrics']
                        
                        col1.metric("R-squared", f"{metrics['r2']:.4f}")
                        col2.metric("RMSE", f"{metrics['rmse']:.4f}")
                        col3.metric("MAPE", f"{metrics['mape']:.2f}%")
                        col4.metric("Direction Accuracy", f"{metrics['directional_accuracy']:.1f}%")
                        
                        # Backtest chart
                        backtest_fig = create_backtest_chart(backtest_data)
                        st.plotly_chart(backtest_fig, use_container_width=True)
                        
                        # Error distribution
                        error_fig = create_error_distribution_chart(backtest_data)
                        st.plotly_chart(error_fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error running backtest: {str(e)}")
                        logger.exception("Backtest error")
        else:
            st.info("Click 'Generate Predictions' to see forecasts and recommendations.")
    
    def show_performance(self):
        """Show model performance metrics"""
        st.header("Model Performance")
        
        if st.session_state.trained_model is None:
            st.info("Train a model first to see performance metrics. Use the sidebar to get started.")
            return
        
        results = st.session_state.trained_model
        
        st.subheader("Test Set Performance")
        
        # Create performance comparison
        if 'test_results' in results:
            models_data = []
            
            for model_name, metrics in results['test_results'].items():
                models_data.append({
                    'Model': model_name.upper(),
                    'R-squared': metrics['r2'],
                    'RMSE': metrics['rmse'],
                    'MAE': metrics['mae'],
                    'MAPE': metrics['mape']
                })
            
            df = pd.DataFrame(models_data)
            
            # Display table
            st.dataframe(
                df.style.highlight_max(subset=['R-squared'], color='lightgreen')
                       .highlight_min(subset=['RMSE', 'MAE', 'MAPE'], color='lightgreen'),
                use_container_width=True
            )
            
            # Bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=df['Model'],
                y=df['R-squared'],
                name='R-squared Score',
                text=df['R-squared'].round(4),
                textposition='auto'
            ))
            
            fig.update_layout(
                title='Model Comparison - R-squared Score',
                yaxis_title='R-squared Score',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics
            st.subheader("Detailed Metrics Explanation")
            
            with st.expander("What do these metrics mean?"):
                st.write("**R-squared (R2):** Measures how well predictions match actual values. Closer to 1 is better.")
                st.write("**RMSE (Root Mean Squared Error):** Average prediction error. Lower is better.")
                st.write("**MAE (Mean Absolute Error):** Average absolute difference between predicted and actual. Lower is better.")
                st.write("**MAPE (Mean Absolute Percentage Error):** Average error as a percentage. Lower is better.")
    
    def show_about(self):
        """Show about page"""
        st.header("About Stock Predictor V2")
        
        st.markdown("""
        ## Overview
        
        This advanced stock prediction system uses ensemble machine learning to forecast stock prices.
        
        ### Features
        
        - **70+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, and more
        - **Market Regime Detection**: Identifies volatility and trend regimes
        - **Ensemble Learning**: Combines LSTM, Random Forest, and XGBoost
        - **Real-time Data**: Fetches latest stock data from Yahoo Finance
        - **Enhanced Predictions**: Future forecasts with confidence intervals
        - **Backtesting**: Validate model performance on historical data
        
        ### Models
        
        1. **LSTM**: Deep learning model for sequential data
        2. **Random Forest**: Ensemble of decision trees
        3. **XGBoost**: Gradient boosting algorithm
        4. **Ensemble**: Weighted combination of all models
        
        ### How to Use
        
        1. Select a stock symbol from the sidebar
        2. Configure model settings (lookback days, prediction horizon)
        3. Click "Train Model" to train on historical data
        4. Go to "Predictions" tab and click "Generate Predictions"
        5. View forecasts, confidence intervals, and recommendations
        6. Run backtests to validate model accuracy
        
        ### Disclaimer
        
        **This tool is for educational purposes only.** Stock predictions are inherently uncertain.
        Always consult financial advisors before making investment decisions.
        
        ### Tech Stack
        
        - **Frontend**: Streamlit
        - **ML**: TensorFlow, XGBoost, Scikit-learn
        - **Data**: Yahoo Finance (yfinance)
        - **Visualization**: Plotly
        - **Feature Engineering**: TA-Lib, pandas
        
        ### Performance Notes
        
        - Training time varies by model type and data size
        - Ensemble models provide best accuracy but take longest to train
        - Random Forest is fastest while maintaining good accuracy
        - LSTM works best with longer historical data
        - Confidence intervals represent 95% prediction intervals
        
        ---
        
        Built with Python | Version 2.0
        """)

def main():
    """Main entry point"""
    app = StockPredictorApp()
    app.run()

if __name__ == "__main__":
    main()