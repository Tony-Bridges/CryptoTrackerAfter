import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd

def plot_price_chart(data: pd.DataFrame, symbol: str):
    """Create an interactive price chart with volume."""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    # Volume bars
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        yaxis='y2',
        opacity=0.3
    ))
    
    fig.update_layout(
        title=f'{symbol} Price and Volume',
        yaxis_title='Price (USD)',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right'
        ),
        height=600,
        template='plotly_dark'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_technical_indicators(data: pd.DataFrame):
    """Plot technical indicators (RSI, MACD, etc.)."""
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=rsi,
        name='RSI'
    ))
    
    fig.update_layout(
        title='Relative Strength Index (RSI)',
        yaxis_title='RSI',
        height=300,
        template='plotly_dark'
    )
    
    st.plotly_chart(fig, use_container_width=True)
