import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def create_candlestick_chart(data: pd.DataFrame, title: str = "Price Chart") -> go.Figure:
    """Create a candlestick chart from OHLCV data."""
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])
    
    fig.update_layout(
        title=title,
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_dark'
    )
    
    return fig

def create_volume_chart(data: pd.DataFrame) -> go.Figure:
    """Create a volume chart."""
    fig = go.Figure(data=[go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume'
    )])
    
    fig.update_layout(
        title='Trading Volume',
        yaxis_title='Volume',
        xaxis_title='Date',
        template='plotly_dark'
    )
    
    return fig

def create_risk_gauge(risk_score: float) -> go.Figure:
    """Create a risk gauge chart."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))
    
    fig.update_layout(title="Risk Assessment")
    return fig
