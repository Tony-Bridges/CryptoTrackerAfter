import streamlit as st
import pandas as pd
from components.charts import plot_price_chart, plot_technical_indicators
from components.risk_analysis import display_risk_analysis
from components.security_monitor import display_security_monitor
from components.model_config import display_model_configuration
from utils.data_loader import load_crypto_data, get_available_cryptos, format_currency

def render_dashboard():
    """Render the main dashboard."""
    st.title("Cryptocurrency Analysis Platform")

    # Add model configuration to sidebar
    display_model_configuration()

    # Sidebar controls
    st.sidebar.header("Settings")
    selected_crypto = st.sidebar.selectbox(
        "Select Cryptocurrency",
        get_available_cryptos()
    )

    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
    )

    # Load data
    data = load_crypto_data(selected_crypto, timeframe)

    if data is not None:
        # Get model parameters from session state
        model_params = st.session_state.get('model_parameters', None)

        # Current price metrics
        current_price = data['Close'].iloc[-1]
        price_change = data['Close'].pct_change().iloc[-1]

        cols = st.columns(3)
        with cols[0]:
            st.metric(
                "Current Price",
                format_currency(current_price),
                f"{price_change:.2%}"
            )
        with cols[1]:
            st.metric(
                "24h Volume",
                format_currency(data['Volume'].iloc[-1])
            )
        with cols[2]:
            st.metric(
                "Market Cap",
                format_currency(current_price * data['Volume'].iloc[-1])
            )

        # Create tabs for different analysis types
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Market Analysis", 
            "Risk Analysis", 
            "Security Monitor",
            "Blockchain Explorer",
            "Code Optimization"
        ])

        with tab1:
            st.subheader("Price Analysis")
            plot_price_chart(data, selected_crypto)
            plot_technical_indicators(data)

        with tab2:
            display_risk_analysis(data)

        with tab3:
            display_security_monitor()

        with tab4:
            st.subheader("Blockchain Explorer")
            if 'blockchain' not in st.session_state:
                st.session_state.blockchain = []

            # Block creation form
            with st.form("create_block"):
                st.write("Create New Block")
                transaction_data = st.text_area("Transaction Data")
                submitted = st.form_submit_button("Create Block")

                if submitted and transaction_data:
                    new_block = {
                        'index': len(st.session_state.blockchain),
                        'timestamp': pd.Timestamp.now(),
                        'data': transaction_data,
                        'previous_hash': '0' if not st.session_state.blockchain else st.session_state.blockchain[-1]['hash']
                    }
                    st.session_state.blockchain.append(new_block)
                    st.success("Block created successfully!")

            # Display blockchain
            if st.session_state.blockchain:
                for block in st.session_state.blockchain:
                    with st.expander(f"Block #{block['index']}"):
                        st.json(block)

        with tab5:
            st.subheader("Code Optimization")
            code_input = st.text_area("Enter code to analyze:", height=200)
            if st.button("Analyze Code"):
                if code_input:
                    # Placeholder for RemedialAI analysis
                    st.info("Code analysis in progress...")
                    st.code(code_input, language="python")
                    st.success("Analysis complete! Suggestions will appear here.")
    else:
        st.error("Failed to load cryptocurrency data. Please try again later.")