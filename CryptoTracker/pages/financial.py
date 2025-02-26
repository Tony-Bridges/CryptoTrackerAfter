import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from crypto_sim.models import FinancialAI

st.set_page_config(page_title="Financial Analysis", page_icon="💹")

# Initialize AI
financial_ai = FinancialAI(volatility_data={'BTC-USD': 0.12, 'ETH-USD': 0.15})

def render_financial_page():
    st.title("Financial Analysis")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### Portfolio Analysis")

        # Portfolio input
        portfolio = {}
        st.markdown("Enter your portfolio:")

        col_asset, col_amount = st.columns(2)
        with col_asset:
            asset = st.text_input("Asset")
        with col_amount:
            amount = st.number_input("Amount", min_value=0.0)

        if st.button("Add to Portfolio"):
            if asset and amount > 0:
                portfolio[asset] = amount
                st.success(f"Added {amount} {asset} to portfolio")

        if portfolio:
            st.markdown("### Portfolio Risk Analysis")
            risk_score = financial_ai.assess_risk(portfolio)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ]
                }
            ))

            fig.update_layout(title="Portfolio Risk Gauge")
            st.plotly_chart(fig)

    with col2:
        st.markdown("### Market Forecasts")
        selected_asset = st.selectbox("Select Asset", ["BTC-USD", "ETH-USD"])

        if selected_asset:
            forecast = financial_ai.forecast_price([{'Close': 100, 'timestamp': pd.Timestamp.now()}], periods=7)

            st.line_chart(pd.Series(forecast, name="Forecast"))

            st.markdown("### Trading Signals")
            st.markdown("Based on AI analysis:")

            if forecast[-1] > forecast[0]:
                st.success("🔼 Bullish Signal")
            else:
                st.error("🔽 Bearish Signal")

if __name__ == "__main__":
    render_financial_page()