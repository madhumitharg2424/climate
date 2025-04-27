import streamlit as st
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import base64

# --- Function to set background image ---
def set_background(jpg_file):
    with open(jpg_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}"); 
        background-size: cover; 
        background-position: center; 
        background-repeat: no-repeat; 
        background-attachment: fixed; 
    }} 
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- Load API Key ---
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# --- Load historical dataset ---
csv_file = "weather_regression_data.csv"

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
else:
    st.error(f"CSV file '{csv_file}' not found!")
    st.stop()

df["City"] = df["City"].str.lower()

# --- Streamlit UI ---
set_background("blue.jpg")  # Set background image

st.title("🌦️ Weather Prediction")

# User input
city_name = st.text_input("Enter City Name (example: chennai):", "").strip().lower()

# Initialize session state to track if data has been fetched and displayed
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

# --- Fetch live weather ---
def fetch_live_weather(city):
    params = {"q": city + ",IN", "appid": API_KEY, "units": "metric"}
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        return data["main"]["temp"], data["weather"][0]["description"]
    else:
        return None, None

# --- Predict future temperatures ---
def predict_future_temps(city):
    global df  # allow updating the dataframe

    city_df = df[df["City"] == city]
    if city_df.empty:
        return None, None, None, None

    city_df['Date'] = pd.to_datetime(city_df['Date'])
    city_df['DayOfYear'] = city_df['Date'].dt.dayofyear

    X = city_df[['DayOfYear']]
    y = city_df['Temperature']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict for the future 5 days based on today's date
    future_dates = []
    predictions = []
    today = datetime.now()
    for i in range(1, 6):  # Predict the next 5 days
        future_day = today + timedelta(days=i)
        day_of_year = future_day.timetuple().tm_yday
        temp_pred = model.predict(np.array([[day_of_year]]))[0]
        predictions.append(temp_pred)
        future_dates.append(future_day.strftime("%Y-%m-%d"))

    return predictions, future_dates

# --- Display output when Submit is pressed ---
submit_button = st.button("Submit")  # Add Submit button
stop_button = st.button("Stop")  # Add Stop button

if submit_button:
    if city_name:
        live_temp, condition = fetch_live_weather(city_name)

        if live_temp is not None:
            st.subheader(f"🌍 Live Weather in {city_name.capitalize()}")
            st.write(f"**Temperature:** {live_temp}°C")
            st.write(f"**Condition:** {condition.capitalize()}")

            predicted_temps, future_dates = predict_future_temps(city_name)

            if predicted_temps:
                st.subheader("📅 Predicted Temperature for Next 5 Days")
                pred_df = pd.DataFrame({
                    "Date": future_dates,
                    "Predicted Temperature (°C)": predicted_temps
                })
                st.dataframe(pred_df)

                # Plot prediction
                fig, ax = plt.subplots()
                ax.plot(future_dates, predicted_temps, marker='o', color='green')
                ax.set_xlabel("Date")
                ax.set_ylabel("Temperature (°C)")
                ax.set_title(f"5-Day Prediction for {city_name.capitalize()}")
                plt.xticks(rotation=45)
                st.pyplot(fig)

            else:
                st.error("❌ No historical data to predict!")
            st.session_state.prediction_made = True
        else:
            st.error("❌ City not found in API or network issue.")
    else:
        st.warning("⚠️ Please enter a valid city name.")

elif stop_button:
    # Reset session state to stop prediction
    st.session_state.prediction_made = False
    # Stop further script execution
    st.stop()

if st.session_state.prediction_made:
    st.success("Prediction and Weather Info Displayed.")
else:
    st.info("Click 'Submit' to get the weather prediction.")
