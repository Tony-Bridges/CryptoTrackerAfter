import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from crypto_sim.models import EconomicAI

st.set_page_config(page_title="Economic Analysis", page_icon="📊")

# Initialize AI
db_params = {
    'host': 'localhost',
    'database': 'crypto_db',
    'user': 'postgres',
    'password': 'password'
}
economic_ai = EconomicAI(db_params=db_params)

def render_economic_page():
    st.title("Economic Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Market Trends")

        # Sample market data for demonstration
        market_data = [
            {'timestamp': pd.Timestamp.now(), 'Close': 30000},
            {'timestamp': pd.Timestamp.now() + pd.Timedelta(days=1), 'Close': 31000},
            # Add more sample data points
        ]

        # Train the model and get predictions
        economic_ai.train_model(market_data)
        prediction = economic_ai.predict_market_trends(market_data)

        # Display prediction results
        st.markdown("#### AI Market Prediction")
        trend_color = {
            "upward": "green",
            "downward": "red",
            "stable": "yellow"
        }[prediction["trend"]]

        st.markdown(f"""
        <div style='padding: 20px; border-radius: 10px; background-color: {trend_color}20;'>
            <h4>Trend: {prediction["trend"].title()}</h4>
            <p>Confidence: {prediction["confidence"]*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Economic Indicators")

        # Tokenomics optimization
        st.markdown("#### Tokenomics Analysis")

        current_state = {
            'market_sentiment': st.selectbox(
                "Market Sentiment",
                ["bullish", "neutral", "bearish"]
            ),
            'emission_rate': st.slider(
                "Current Emission Rate",
                0.1, 2.0, 1.0
            )
        }

        if st.button("Optimize Tokenomics"):
            optimized_state = economic_ai.optimize_tokenomics(current_state)

            st.markdown("#### Optimization Results")
            st.json(optimized_state)

        # Economic Metrics
        st.markdown("### Key Metrics")

        metrics = {
            "Market Cap": "$945B",
            "24h Volume": "$28B",
            "Dominance": "42%"
        }

        for metric, value in metrics.items():
            st.metric(metric, value)

if __name__ == "__main__":
    render_economic_page()