import pandas as pd
import yfinance as yf
import nltk 
import requests
import hashlib 
import numpy as np
import datetime
import json
import tracemalloc
import logging
import time
import random
from functools import wraps
from rich.console import Console
from rich.table import Table
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any
from textblob import TextBlob

class SupervisorAI:
    def __init__(self, db_params):
        """Initialize the SupervisorAI with PostgreSQL connection parameters."""
        self.function_timings = {}
        self.memory_snapshots = {}
        self.db_params = db_params
        self._initialize_database()

    def _initialize_database(self):
        """Creates the necessary performance_metrics table if it doesn't exist."""
        try:
            import psycopg2
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id SERIAL PRIMARY KEY,
                    function_name TEXT NOT NULL,
                    execution_time REAL,
                    memory_usage BIGINT,
                    cpu_usage REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logging.error(f"Database initialization error: {e}")

    def monitor_performance(self, func):
        """Decorator to monitor the performance of a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            tracemalloc.start()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            memory_usage = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            elapsed_time = end_time - start_time
            self._save_performance_metrics(func.__name__, elapsed_time, memory_usage[1], 0.0)
            return result
        return wrapper

    def _save_performance_metrics(self, function_name, execution_time, memory_usage, cpu_usage):
        """Saves performance metrics to the PostgreSQL database."""
        try:
            import psycopg2
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance_metrics (function_name, execution_time, memory_usage, cpu_usage)
                VALUES (%s, %s, %s, %s)
            """, (function_name, execution_time, memory_usage, cpu_usage))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logging.error(f"Error saving metrics: {e}")

class FinancialAI:
    def __init__(self, volatility_data=None, params=None):
        """Initialize FinancialAI with custom parameters."""
        default_params = {
            'lookback_periods': [7, 21],
            'rsi_period': 14,
            'n_estimators': 100,
            'random_state': 42
        }
        self.params = params if params is not None else default_params
        self.lookback_periods = tuple(self.params['lookback_periods'])
        self.rsi_periods = self.params['rsi_period']
        self.volatility_data = volatility_data if volatility_data else {}
        self.model = RandomForestRegressor(
            n_estimators=self.params['n_estimators'],
            random_state=self.params['random_state']
        )
        self.scaler = MinMaxScaler()

    def analyze_market_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            df = pd.DataFrame(data)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif 'timestamp' in df.columns:
                df['Date'] = pd.to_datetime([datetime.datetime.fromtimestamp(ts) for ts in df['timestamp']])
                df.set_index('Date', inplace=True)

            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df = df.dropna(subset=['Close'])

            for period in self.lookback_periods:
                df[f'{period}-day MA'] = df['Close'].rolling(window=period).mean()

            delta = df['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)

            avg_gain = gain.rolling(window=self.rsi_periods).mean()
            avg_loss = loss.rolling(window=self.rsi_periods).mean()

            rs = avg_gain / avg_loss.replace(0, np.nan)
            df['RSI'] = 100 - (100 / (1 + rs))

            return {
                'current_price': df['Close'].iloc[-1] if not df.empty else None,
                **{f'{period}-day MA': df[f'{period}-day MA'].iloc[-1] if not df.empty else None for period in self.lookback_periods},
                'RSI': df['RSI'].iloc[-1] if not df.empty else None
            }

        except Exception as e:
            logging.error(f"Error in analyze_market_data: {e}")
            return {
                'current_price': None,
                '7-day MA': None,
                '21-day MA': None,
                'RSI': None
            }

    def assess_risk(self, portfolio: Dict[str, float]) -> float:
        total_value = sum(portfolio.values())
        if total_value == 0:
            return 0

        weights = {asset: quantity / total_value for asset, quantity in portfolio.items()}
        weighted_volatility = sum(weight * self.volatility_data.get(asset, 0.1) for asset, weight in weights.items())
        return weighted_volatility

    def forecast_price(self, data: List[Dict[str, Any]], periods: int = 7):
        try:
            df = pd.DataFrame(data)
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce').dropna()

            if len(df) < 2:
                return np.array([np.nan] * periods)

            X = np.arange(len(df)).reshape(-1, 1)
            y = df['Close'].values

            self.model.fit(X, y)

            future_X = np.arange(len(df), len(df) + periods).reshape(-1, 1)
            forecast = self.model.predict(future_X)

            return forecast

        except Exception as e:
            logging.error(f"Forecasting error: {e}")
            return np.array([np.nan] * periods)

class EconomicAI:
    def __init__(self, db_params, params=None):
        """Initialize EconomicAI with custom parameters."""
        default_params = {
            'lookback_window': 60,
            'n_estimators': 100,
            'test_size': 0.2,
            'random_state': 42
        }
        self.params = params if params is not None else default_params
        self.model = RandomForestRegressor(
            n_estimators=self.params['n_estimators'],
            random_state=self.params['random_state']
        )
        self.scaler = MinMaxScaler()
        self.lookback = self.params['lookback_window']
        self.db_params = db_params

    def train_model(self, data: List[Dict[str, Any]]):
        try:
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime([datetime.datetime.fromtimestamp(ts) for ts in df['timestamp']])
            df.set_index('Date', inplace=True)
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce').dropna()

            if df.empty or len(df) <= self.lookback:
                logging.warning("Insufficient data for training")
                return

            X = []
            y = []
            for i in range(self.lookback, len(df)):
                X.append(df['Close'].iloc[i-self.lookback:i].values)
                y.append(df['Close'].iloc[i])

            X = np.array(X)
            y = np.array(y)

            if len(X) < 2:
                logging.warning("Not enough training samples")
                return

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, 
                test_size=self.params['test_size'],
                random_state=self.params['random_state']
            )
            self.model.fit(X_train, y_train)

        except Exception as e:
            logging.error(f"Error in train_model: {e}")

    def predict_market_trends(self, data: List[Dict[str, Any]]):
        if not hasattr(self.model, 'predict'):
            return {"trend": "stable", "confidence": 0.5}

        try:
            df = pd.DataFrame(data)
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce').dropna()

            if len(df) <= self.lookback:
                return {"trend": "stable", "confidence": 0.5}

            X = df['Close'][-self.lookback:].values.reshape(1, -1)
            predicted_price = self.model.predict(X)[0]
            current_price = df['Close'].iloc[-1]

            trend = "upward" if predicted_price > current_price else "downward" if predicted_price < current_price else "stable"
            confidence = min(abs(predicted_price - current_price) / current_price, 1.0)

            return {"trend": trend, "confidence": confidence}

        except Exception as e:
            logging.error(f"Error in predict_market_trends: {e}")
            return {"trend": "stable", "confidence": 0.5}

    def optimize_tokenomics(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            market_sentiment = current_state.get('market_sentiment', 'neutral')
            current_emission_rate = current_state.get('emission_rate', 1.0)

            def optimize_emission_rate(emission_rate, sentiment):
                target_rate = 0.9 if sentiment == 'bearish' else 1.1 if sentiment == 'bullish' else 1.0
                return abs(emission_rate - target_rate)

            result = minimize(optimize_emission_rate, current_emission_rate, args=(market_sentiment,))
            current_state['emission_rate'] = float(result.x[0])
            return current_state

        except Exception as e:
            logging.error(f"Tokenomics optimization error: {e}")
            return {"error": str(e)}

class SecurityAI:
    def __init__(self, db_params, params=None):
        """Initialize SecurityAI with custom parameters."""
        default_params = {
            'contamination': 0.01,
            'random_state': 42
        }
        self.params = params if params is not None else default_params
        self.model = IsolationForest(
            contamination=self.params['contamination'],
            random_state=self.params['random_state']
        )
        self.trained = False
        self.db_params = db_params

    def train_model(self, transactions: List[Dict[str, Any]]) -> None:
        if not transactions:
            raise ValueError("No transactions provided for training")

        df = pd.DataFrame(transactions)
        df['amount'] = df['amount'].astype(float)
        df['transaction_hour'] = pd.to_datetime(df['timestamp']).dt.hour if 'timestamp' in df else 0
        df['transaction_day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek if 'timestamp' in df else 0

        X = df[['amount', 'transaction_hour', 'transaction_day_of_week']].values
        self.model.fit(X)
        self.trained = True

    def detect_anomalies(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.trained:
            raise RuntimeError("Model must be trained before detecting anomalies")

        if not transactions:
            return []

        df = pd.DataFrame(transactions)
        df['amount'] = df['amount'].astype(float)
        df['transaction_hour'] = pd.to_datetime(df['timestamp']).dt.hour if 'timestamp' in df else 0
        df['transaction_day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek if 'timestamp' in df else 0

        X = df[['amount', 'transaction_hour', 'transaction_day_of_week']].values
        df['anomaly_score'] = self.model.decision_function(X)
        df['is_anomaly'] = self.model.predict(X) == -1

        return df[df['is_anomaly']].to_dict('records')

    def monitor_network(self, network_data: Dict[str, Any]) -> List[str]:
        alerts = []
        volume = network_data.get('transaction_volume', 0)
        failed_transactions = network_data.get('failed_transactions', 0)
        historical_volume = network_data.get('historical_transaction_volume', [])

        if volume > 10000:
            alerts.append(f"High transaction volume detected: {volume}")

        if failed_transactions > 100:
            alerts.append(f"High number of failed transactions: {failed_transactions}")

        if historical_volume:
            average_volume = sum(historical_volume) / len(historical_volume)
            volume_change_percentage = abs((volume - average_volume) / average_volume) * 100
            if volume_change_percentage > 50:
                alerts.append(f"Significant change in transaction volume detected: {volume_change_percentage:.2f}%")

        return alerts