import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from crypto_sim.models import SecurityAI

st.set_page_config(page_title="Security Monitor", page_icon="🔒")

# Initialize AI
db_params = {
    'host': 'localhost',
    'database': 'crypto_db',
    'user': 'postgres',
    'password': 'password'
}
security_ai = SecurityAI(db_params=db_params)

def render_security_page():
    st.title("Security Monitor")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Transaction Monitoring")

        # Sample transaction data
        transactions = [
            {
                'timestamp': pd.Timestamp.now(),
                'amount': 1.5,
                'sender': '0x123...',
                'receiver': '0x456...'
            }
        ]

        # Train the model
        security_ai.train_model(transactions)

        # Detect anomalies
        anomalies = security_ai.detect_anomalies(transactions)

        if anomalies:
            st.error("⚠️ Anomalies Detected!")
            for anomaly in anomalies:
                st.markdown(f"""
                <div style='padding: 10px; border-radius: 5px; background-color: rgba(255, 0, 0, 0.1);'>
                    <p><strong>Amount:</strong> {anomaly['amount']}</p>
                    <p><strong>Time:</strong> {anomaly['timestamp']}</p>
                    <p><strong>Score:</strong> {anomaly['anomaly_score']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No anomalies detected")

    with col2:
        st.markdown("### Network Status")

        network_data = {
            'transaction_volume': 8500,
            'failed_transactions': 45,
            'historical_transaction_volume': [8000, 8200, 8400, 8300, 8600]
        }

        alerts = security_ai.monitor_network(network_data)

        st.metric("Transaction Volume", network_data['transaction_volume'])
        st.metric("Failed Transactions", network_data['failed_transactions'])

        if alerts:
            st.warning("⚠️ Network Alerts")
            for alert in alerts:
                st.markdown(f"- {alert}")

if __name__ == "__main__":
    render_security_page()