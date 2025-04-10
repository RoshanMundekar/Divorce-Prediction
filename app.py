import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Divorce Prediction App",
    page_icon="💔",
    layout="wide"  # Ensures responsiveness
)

# Title
st.title("💔 Divorce Prediction System")

# Introduction
st.write(
    "This application helps predict the likelihood of divorce based on various relationship factors.Marriage is a fundamental aspect of human relationships, but various factors contribute to its success or failure. Divorce rates have been increasing, and predictive analytics can help identify at-risk relationships. The Divorce Prediction System aims to use machine learning (ML) techniques to analyze relationship patterns and predict the likelihood of divorce based on various factors such as communication, conflict resolution, and emotional satisfaction."
    "This project aims to predict the likelihood of divorce based on relationship patterns using ML techniques. By providing timely insights, the system can help couples improve their relationships and make informed decisions."
    
)

# Sidebar navigation
st.sidebar.title("Models")
st.sidebar.page_link("pages/steam.py", label="🔹 Predicts results")
st.sidebar.page_link("pages/test.py", label="🔹 ChatBot Ai")