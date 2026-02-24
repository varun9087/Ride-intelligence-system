# app.py

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Ride Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM PREMIUM CSS
# --------------------------------------------------

st.markdown("""
<style>
body {
    background-color: #0f172a;
}

.main-title {
    font-size: 32px;
    font-weight: 700;
}

.subtitle {
    color: #94a3b8;
    font-size: 16px;
    margin-bottom: 20px;
}

.card {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.4);
}

.card-title {
    font-size: 14px;
    color: #94a3b8;
}

.card-value {
    font-size: 28px;
    font-weight: 700;
    margin-top: 8px;
}

.section-header {
    font-size: 20px;
    font-weight: 600;
    margin-top: 30px;
    margin-bottom: 10px;
}

.footer {
    margin-top: 40px;
    text-align: center;
    color: #64748b;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = pickle.load(open("models/model.pkl", "rb"))

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    ["Executive Overview", "Trip Estimation", "Demand Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Ride Intelligence Platform")
st.sidebar.markdown("Version 1.0")

# ==================================================
# EXECUTIVE OVERVIEW
# ==================================================

if page == "Executive Overview":

    st.markdown('<div class="main-title">Ride Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Operational Performance and Predictive Analytics Dashboard</div>', unsafe_allow_html=True)

    df = pd.read_csv("data/train.csv")
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour

    total_rides = len(df)
    avg_duration = df["trip_duration"].mean() / 60
    peak_hour = df.groupby("hour").size().idxmax()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Total Trips</div>
            <div class="card-value">{total_rides:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Average Duration (min)</div>
            <div class="card-value">{avg_duration:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Peak Demand Hour</div>
            <div class="card-value">{peak_hour}:00</div>
        </div>
        """, unsafe_allow_html=True)

# ==================================================
# TRIP ESTIMATION
# ==================================================

elif page == "Trip Estimation":

    st.markdown('<div class="main-title">Trip Duration Estimation</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predict estimated trip duration and pricing based on ride inputs</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        distance = st.number_input("Distance (km)", 0.1, 50.0, 5.0)
        passenger_count = st.slider("Passenger Count", 1, 6, 2)

    with col2:
        pickup_time = st.time_input("Pickup Time", datetime.now().time())
        hour = pickup_time.hour

    with col3:
        pickup_date = st.date_input("Pickup Date")
        day = pickup_date.weekday()

    is_weekend = 1 if day >= 5 else 0

    st.markdown("")

    if st.button("Run Prediction"):

        input_data = np.array([[distance, passenger_count, hour, day, is_weekend]])
        prediction = model.predict(input_data)

        duration_minutes = prediction[0] / 60
        estimated_fare = 2.5 + (distance * 1.8)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Estimated Duration</div>
                <div class="card-value">{duration_minutes:.2f} minutes</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Estimated Fare</div>
                <div class="card-value">${estimated_fare:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

# ==================================================
# DEMAND ANALYTICS
# ==================================================

elif page == "Demand Analytics":

    st.markdown('<div class="main-title">Demand Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Time-based ride demand and geospatial distribution</div>', unsafe_allow_html=True)

    df = pd.read_csv("data/train.csv")
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day"] = df["pickup_datetime"].dt.dayofweek

    st.markdown('<div class="section-header">Hourly Demand</div>', unsafe_allow_html=True)
    hourly_demand = df.groupby("hour").size()
    st.line_chart(hourly_demand)

    st.markdown('<div class="section-header">Weekday Demand</div>', unsafe_allow_html=True)
    weekday_demand = df.groupby("day").size()
    st.bar_chart(weekday_demand)

    st.markdown('<div class="section-header">Pickup Location Heatmap</div>', unsafe_allow_html=True)

    sample_df = df[["pickup_latitude", "pickup_longitude"]].dropna().sample(3000)

    st.map(sample_df.rename(columns={
        "pickup_latitude": "lat",
        "pickup_longitude": "lon"
    }))

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown('<div class="footer">Ride Intelligence Platform | Built with Machine Learning and Streamlit</div>', unsafe_allow_html=True)