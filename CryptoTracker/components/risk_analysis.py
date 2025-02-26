import streamlit as st
import pandas as pd
import numpy as np

def calculate_risk_metrics(data: pd.DataFrame):
    """Calculate various risk metrics."""
    returns = data['Close'].pct_change().dropna()
    
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (returns.mean() * 252) / volatility
    var_95 = np.percentile(returns, 5)
    max_drawdown = (data['Close'] / data['Close'].expanding(min_periods=1).max() - 1).min()
    
    metrics = {
        'Daily Volatility': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Value at Risk (95%)': f"{var_95:.2%}",
        'Maximum Drawdown': f"{max_drawdown:.2%}"
    }
    
    return metrics

def display_risk_analysis(data: pd.DataFrame):
    """Display risk analysis metrics and visualizations."""
    st.subheader("Risk Analysis")
    
    metrics = calculate_risk_metrics(data)
    
    cols = st.columns(len(metrics))
    for col, (metric, value) in zip(cols, metrics.items()):
        with col:
            st.metric(metric, value)
    
    # Risk classification
    volatility = float(metrics['Daily Volatility'].strip('%')) / 100
    risk_level = "High" if volatility > 0.03 else "Medium" if volatility > 0.02 else "Low"
    
    st.markdown(f"""
        <div class="risk-indicator risk-{risk_level.lower()}">
            Risk Level: {risk_level}
        </div>
    """, unsafe_allow_html=True)
