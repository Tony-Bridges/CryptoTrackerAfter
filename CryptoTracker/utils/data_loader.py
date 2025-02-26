import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def load_crypto_data(symbol: str, period: str = '1y'):
    """Load cryptocurrency data from Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {str(e)}")
        return None

def get_available_cryptos():
    """Return list of available cryptocurrencies."""
    return [
        "BTC-USD",
        "ETH-USD",
        "USDT-USD",
        "BNB-USD",
        "XRP-USD",
        "ADA-USD",
        "DOGE-USD",
        "SOL-USD"
    ]

def format_currency(value):
    """Format currency values with appropriate suffixes."""
    if value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    elif value >= 1e3:
        return f"${value/1e3:.2f}K"
    else:
        return f"${value:.2f}"
