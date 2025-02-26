import streamlit as st
from utils.theme import apply_theme
from components.dashboard import render_dashboard

def main():
    # Apply theme and styling
    apply_theme()
    
    # Render main dashboard
    render_dashboard()

if __name__ == "__main__":
    main()
