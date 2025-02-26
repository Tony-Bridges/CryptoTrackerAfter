import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_security_data():
    """Generate mock security monitoring data."""
    now = datetime.now()
    times = [now - timedelta(minutes=i) for i in range(60)]
    
    data = {
        'timestamp': times,
        'transaction_count': np.random.normal(1000, 100, 60).astype(int),
        'failed_transactions': np.random.normal(5, 2, 60).astype(int),
        'network_load': np.random.uniform(0.4, 0.8, 60),
        'anomaly_score': np.random.uniform(0, 1, 60)
    }
    
    return pd.DataFrame(data)

def display_security_monitor():
    """Display security monitoring dashboard."""
    st.subheader("Security Monitor")
    
    data = generate_mock_security_data()
    
    # Current status indicators
    cols = st.columns(4)
    with cols[0]:
        st.metric("Active Transactions", f"{data['transaction_count'].iloc[0]:,}")
    with cols[1]:
        st.metric("Failed Transactions", data['failed_transactions'].iloc[0])
    with cols[2]:
        st.metric("Network Load", f"{data['network_load'].iloc[0]:.1%}")
    with cols[3]:
        st.metric("Anomaly Score", f"{data['anomaly_score'].iloc[0]:.2f}")
    
    # Security alerts
    if data['anomaly_score'].iloc[0] > 0.8:
        st.error("⚠️ High anomaly score detected!")
    elif data['failed_transactions'].iloc[0] > 10:
        st.warning("⚠️ Elevated failed transaction rate!")
