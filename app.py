# app.py

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page config
st.set_page_config(page_title="Ride Intelligence System", layout="wide")

st.title("🚖 Ride Intelligence System")
st.markdown("Operations & Trip Duration Analytics Platform")

# Load model
model = pickle.load(open("models/model.pkl", "rb"))

# Tabs
tab1, tab2 = st.tabs(["🚕 Trip Prediction", "📊 Demand Analytics"])


# TAB 1: TRIP PREDICTION


with tab1:
    st.subheader("Predict Trip Duration")

    col1, col2 = st.columns(2)

    with col1:
        distance = st.slider("Distance (km)", 0.1, 50.0, 5.0)
        passenger_count = st.slider("Passenger Count", 1, 6, 2)

    with col2:
        hour = st.slider("Hour of Day", 0, 23, 12)
        day = st.slider("Day of Week (0=Mon)", 0, 6, 2)

    is_weekend = 1 if day >= 5 else 0

    if st.button("Predict Duration"):
        input_data = np.array([[distance, passenger_count, hour, day, is_weekend]])
        prediction = model.predict(input_data)

        minutes = prediction[0] / 60

        st.success(f"Estimated Trip Duration: {minutes:.2f} minutes")

        estimated_fare = 2.5 + (distance * 1.8)
        st.info(f"Estimated Fare: ${estimated_fare:.2f}")


# TAB 2: DEMAND ANALYTICS

with tab2:
    st.subheader("Ride Demand Insights")

    df = pd.read_csv("data/train.csv")

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day"] = df["pickup_datetime"].dt.dayofweek

    # Hourly demand
    hourly_demand = df.groupby("hour").size()

    st.markdown("### 📈 Hourly Demand")
    st.line_chart(hourly_demand)

    # Weekday demand
    weekday_demand = df.groupby("day").size()

    st.markdown("### 📊 Weekday Demand")
    st.bar_chart(weekday_demand)

    # Heatmap (sample for performance)
    st.markdown("### 🗺 Pickup Location Heatmap")

    sample_df = df[["pickup_latitude", "pickup_longitude"]].dropna().sample(2000)

    st.map(sample_df.rename(columns={
        "pickup_latitude": "lat",
        "pickup_longitude": "lon"
    }))