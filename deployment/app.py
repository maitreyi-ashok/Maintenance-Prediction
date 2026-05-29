
# ENGINE PREDICTIVE MAINTENANCE - STREAMLIT APP

import streamlit as st
import pandas as pd
import joblib

from huggingface_hub import hf_hub_download

# PAGE CONFIG

st.set_page_config(
    page_title="Engine Predictive Maintenance",
    layout="centered"
)

# LOAD MODEL FROM HUGGING FACE

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="MaitreyiAshok/maintenance-prediction-model",
        filename="model.pkl"
    )

    model = joblib.load(model_path)

    return model

model = load_model()

# TITLE

st.title(" Engine Predictive Maintenance System")

st.write(
    """
    Enter engine sensor values below to predict whether
    the engine is operating normally or requires maintenance.
    """
)

# USER INPUTS

engine_rpm = st.number_input(
    "Engine RPM",
    min_value=0.0,
    max_value=5000.0,
    value=800.0
)

lub_oil_pressure = st.number_input(
    "Lubricating Oil Pressure",
    min_value=0.0,
    max_value=10.0,
    value=3.0
)

fuel_pressure = st.number_input(
    "Fuel Pressure",
    min_value=0.0,
    max_value=25.0,
    value=6.0
)

coolant_pressure = st.number_input(
    "Coolant Pressure",
    min_value=0.0,
    max_value=10.0,
    value=2.0
)

lub_oil_temp = st.number_input(
    "Lubricating Oil Temperature (°C)",
    min_value=0.0,
    max_value=200.0,
    value=80.0
)

coolant_temp = st.number_input(
    "Coolant Temperature (°C)",
    min_value=0.0,
    max_value=250.0,
    value=85.0
)

# PREDICTION BUTTON

if st.button("Predict Engine Condition"):

    try:

        # CREATE INPUT DATAFRAME

        input_df = pd.DataFrame([{

            "Engine_RPM": engine_rpm,

            "Lub_Oil_Pressure": lub_oil_pressure,

            "Fuel_Pressure": fuel_pressure,

            "Coolant_Pressure": coolant_pressure,

            "Lub_Oil_Temperature": lub_oil_temp,

            "Coolant_Temperature": coolant_temp
        }])

        # PREDICTION

        prediction = model.predict(input_df)[0]

        # PROBABILITY

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(input_df)[0][1]

            st.info(
                f" Failure Probability: "
                f"{round(probability * 100, 2)}%"
            )

        # RESULT DISPLAY

        if prediction == 1:

            st.error(
                "Maintenance Required!\n\n"
                "The engine is likely experiencing abnormal conditions."
            )

        else:

            st.success(
                " Engine Operating Normally"
            )

        # DISPLAY INPUT SUMMARY

        st.subheader(" Input Summary")

        st.dataframe(input_df)

    except Exception as e:

        st.error(
            f" Error during prediction:\n{str(e)}"
        )

# FOOTER

st.markdown("---")

st.caption(
    "Engine Predictive Maintenance App using Machine Learning & XGBoost"
)
