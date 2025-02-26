import streamlit as st
import json
from typing import Dict, Any

def load_default_parameters() -> Dict[str, Any]:
    """Load default model parameters."""
    return {
        'financial': {
            'lookback_periods': [7, 21],
            'rsi_period': 14,
            'n_estimators': 100,
            'random_state': 42
        },
        'economic': {
            'lookback_window': 60,
            'n_estimators': 100,
            'test_size': 0.2,
            'random_state': 42
        },
        'security': {
            'contamination': 0.01,
            'random_state': 42
        }
    }

def save_parameters(params: Dict[str, Any]):
    """Save parameters to session state."""
    if 'model_parameters' not in st.session_state:
        st.session_state.model_parameters = {}
    st.session_state.model_parameters.update(params)

def display_model_configuration():
    """Display and manage model parameter configuration."""
    st.sidebar.markdown("### Model Configuration")
    
    if 'model_parameters' not in st.session_state:
        st.session_state.model_parameters = load_default_parameters()
    
    if st.sidebar.checkbox("Show Advanced Settings"):
        st.sidebar.markdown("#### Financial Model Parameters")
        financial_params = st.session_state.model_parameters['financial']
        
        lookback = st.sidebar.multiselect(
            "Lookback Periods (days)",
            options=[3, 7, 14, 21, 30, 60],
            default=financial_params['lookback_periods']
        )
        
        rsi_period = st.sidebar.slider(
            "RSI Period",
            min_value=7,
            max_value=21,
            value=financial_params['rsi_period']
        )
        
        fin_estimators = st.sidebar.slider(
            "Number of Estimators (Financial)",
            min_value=50,
            max_value=500,
            value=financial_params['n_estimators']
        )
        
        st.sidebar.markdown("#### Economic Model Parameters")
        economic_params = st.session_state.model_parameters['economic']
        
        lookback_window = st.sidebar.slider(
            "Lookback Window",
            min_value=30,
            max_value=120,
            value=economic_params['lookback_window']
        )
        
        eco_estimators = st.sidebar.slider(
            "Number of Estimators (Economic)",
            min_value=50,
            max_value=500,
            value=economic_params['n_estimators']
        )
        
        st.sidebar.markdown("#### Security Model Parameters")
        security_params = st.session_state.model_parameters['security']
        
        contamination = st.sidebar.slider(
            "Anomaly Contamination",
            min_value=0.001,
            max_value=0.1,
            value=float(security_params['contamination']),
            format="%.3f"
        )
        
        # Update parameters
        new_params = {
            'financial': {
                'lookback_periods': lookback,
                'rsi_period': rsi_period,
                'n_estimators': fin_estimators,
                'random_state': 42
            },
            'economic': {
                'lookback_window': lookback_window,
                'n_estimators': eco_estimators,
                'test_size': 0.2,
                'random_state': 42
            },
            'security': {
                'contamination': contamination,
                'random_state': 42
            }
        }
        
        if st.sidebar.button("Apply Changes"):
            save_parameters(new_params)
            st.sidebar.success("Parameters updated successfully!")
