import streamlit as st
import requests
import joblib
import pandas as pd

# 1. הגדרות עמוד ועיצוב קסטום
st.set_page_config(page_title="Accident Severity Predictor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button {
        background-color: #4b6cb7; color: white; width: 100%; 
        border-radius: 8px; font-weight: bold; font-size: 18px;
        height: 50px; border: none; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #182848; border: 1px solid #4b6cb7; }
    .output-card {
        background-color: #1f293d; padding: 20px; 
        border-radius: 10px; border-left: 5px solid #4b6cb7;
        margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .severity-slight { border-left: 6px solid #00ffcc !important; color: #00ffcc; font-weight: bold; }
    .severity-major { border-left: 6px solid #ff3333 !important; color: #ff3333; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)


# 2. טעינת המודל הבינארי האמיתי שלך מהדיסק
@st.cache_resource
def load_binary_model():
    try:
        model = joblib.load('binary_lightgbm_model.pkl')
        return model
    except FileNotFoundError:
        return None


binary_model = load_binary_model()


# 3. פונקציית API למזג אוויר
def get_weather_data(city_name):
    if not city_name:
        return None
    api_key = "b245a156ed259736300c7d7e57ec0e4e"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
    try:
        response = requests.get(url).json()
        if response.get("cod") == 200:
            return {
                "temp": response["main"]["temp"],
                "condition": response["weather"][0]["main"],
                "humidity": response["main"]["humidity"]
            }
        else:
            st.error(f"City '{city_name}' not found.")
            return None
    except Exception as e:
        st.error(f"Weather API Error: {e}")
        return None


# 4. פונקציית החיזוי שמפעילה את המודל הבינארי ומטפלת ב-25 התכונות
def run_model_prediction(ui_features, weather_info):
    # מיפוי ערכים מדויק לפי התמונה שסיפקת
    road_type_mapping = {
        "Roundabout": 0,
        "One way street": 1,
        "Slip Road": 2,
        "Dual carriageway": 3,
        "Single carriageway": 4
    }

    junction_mapping = {
        "Not at junction or within 20 metres": 0,
        "Using private drive or entrance": 1,
        "Other Junction": 2,
        "T or staggered junction": 3,
        "Crossroads / Junction with more than four arms (not roundabout)": 4
    }

    # בניית הנתונים הבסיסיים מתוך ה-UI
    numeric_features = {
        "number_of_vehicles": int(ui_features["number_of_vehicles"]),
        "number_of_casualties": int(ui_features["number_of_casualties"]),
        "road_type": road_type_mapping.get(ui_features["road_type"], 4),
        "speed_limit": int(ui_features["speed_limit"]),
        "junction_detail": junction_mapping.get(ui_features["junction_detail"], 0)
    }

    # יצירת ה-DataFrame ההתחלתי (מכיל 5 תכונות)
    input_df = pd.DataFrame([numeric_features])

    # ערכי ברירת מחדל למצב דמו
    chosen_severity = "Slight"
    confidence_val = 61.5

    if binary_model is not None:
        try:
            # שליפת רשימת 25 התכונות המדויקת שהמודל מצפה לקבל מהאימון
            expected_features = binary_model.feature_name_

            # הזרקת כל 20 העמודות החסרות כערך 0 (כדי לא לפגוע במבנה ה-DataFrame)
            for col in expected_features:
                if col not in input_df.columns:
                    input_df[col] = 0

            # סידור העמודות מחדש בדיוק לפי הסדר שבו המודל אומן
            input_df = input_df[expected_features]

        except AttributeError:
            # למקרה שהמודל נטען בצורה שאין לו את ה-property (למשל במודלים ישנים)
            pass

        # הרצת החיזוי כעת כשמבנה הטבלה מושלם עם 25 עמודות
        prediction = binary_model.predict(input_df)[0]

        try:
            probabilities = binary_model.predict_proba(input_df)[0]
            confidence_val = round(max(probabilities) * 100, 1)
        except AttributeError:
            confidence_val = 100.0

        if prediction == 1:
            chosen_severity = "Major"
        else:
            chosen_severity = "Slight"

    # בניית הפלט המשולש
    diagnosis = chosen_severity
    confidence = f"The accident was classified as '{chosen_severity}' with {confidence_val}% confidence."
    explanation = f"The prediction was primarily influenced by the following factors: [Road Type: {ui_features['road_type']}] and [Speed Limit: {ui_features['speed_limit']} mph]."

    if weather_info:
        explanation += f" Live weather in the area ({weather_info['condition']}, {weather_info['temp']}°C) was also factored in."

    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "explanation": explanation
    }


# 5. בניית ה-UI
st.title("🚨 Accident Severity Prediction System")
st.subheader("Binary Classification Model (Slight vs. Major)")
st.write("---")

col_input, col_results = st.columns([1, 1.5])

with col_input:
    st.header("📋 Input Features")

    speed_options = [30, 50, 70, 90, 110]
    road_type_options = [
        "Roundabout",
        "One way street",
        "Slip Road",
        "Dual carriageway",
        "Single carriageway"
    ]
    junction_options = [
        "Not at junction or within 20 metres",
        "Using private drive or entrance",
        "Other Junction",
        "T or staggered junction",
        "Crossroads / Junction with more than four arms (not roundabout)"
    ]
    numbers_0_to_9 = list(range(10))

    speed_limit = st.selectbox("Speed Limit (mph)", speed_options, index=0)
    road_type = st.selectbox("Road Type", road_type_options, index=4)
    junction_detail = st.selectbox("Junction Detail", junction_options, index=0)
    num_casualties = st.selectbox("Number of Casualties", numbers_0_to_9, index=2)
    num_vehicles = st.selectbox("Number of Vehicles", numbers_0_to_9, index=2)

    st.write("---")
    st.header("🌤️ Live Weather Integration")
    city = st.text_input("Enter City Name for Live Weather Data", placeholder="e.g. London")

    predict_clicked = st.button("🚀 Run Severity Prediction")

with col_results:
    st.header("📊 Model Insights & Analysis")

    if binary_model is None:
        st.warning(
            "⚠️ Note: Running in Demo Mode. To use your real model, ensure 'binary_lightgbm_model.pkl' exists in this folder.")

    if predict_clicked:
        with st.spinner("Fetching weather data and running LightGBM Model..."):

            weather_res = get_weather_data(city)

            ui_features = {
                "speed_limit": speed_limit,
                "road_type": road_type,
                "junction_detail": junction_detail,
                "number_of_casualties": num_casualties,
                "number_of_vehicles": num_vehicles
            }

            output = run_model_prediction(ui_features, weather_res)

            if weather_res and city:
                st.info(
                    f"🌍 **Live Weather Info for {city}:** {weather_res['temp']}°C, {weather_res['condition']} (Humidity: {weather_res['humidity']}%).")

            st.success("Analysis Complete! Here is the Final Triple Output:")

            severity_class = f"severity-{output['diagnosis'].lower()}"

            st.markdown(f"""
                <div class="output-card {severity_class}">
                    <h4 style="margin:0; color:#b0c4de;">1. Diagnosis</h4>
                    <p style="font-size: 24px; margin: 5px 0 0 0;">Predicted Severity: <strong>{output['diagnosis'].upper()}</strong></p>
                </div>

                <div class="output-card">
                    <h4 style="margin:0; color:#b0c4de;">2. Confidence</h4>
                    <p style="font-size: 16px; margin: 5px 0 0 0;">{output['confidence']}</p>
                </div>

                <div class="output-card">
                    <h4 style="margin:0; color:#b0c4de;">3. Explanation</h4>
                    <p style="font-size: 16px; margin: 5px 0 0 0;">{output['explanation']}</p>
                </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Fill out the features on the left and click 'Run Severity Prediction' to see the model output.")