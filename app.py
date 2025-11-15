import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
#   LOAD MODEL & FEATURES
# ---------------------------
model = joblib.load("churn_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="MTN Churn Prediction", layout="wide")

st.title("📱 MTN Customer Churn Prediction App")
st.write("Fill in customer details below to predict churn probability.")

# ---------------------------------------
#    USER INPUT FORM
# ---------------------------------------
user_input = {}

st.sidebar.header("Input Customer Data")

for col in feature_columns:
    user_input[col] = st.sidebar.text_input(col)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Convert empty fields to 0
input_df = input_df.replace("", 0)

# Convert all numeric-like columns to numbers
input_df = input_df.apply(pd.to_numeric, errors='ignore')

# ---------------------------------------
#    PREDICT
# ---------------------------------------
if st.button("Predict Churn"):
    try:
        prediction_prob = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.subheader("🔮 Prediction Results")

        st.write(f"**Churn Probability:** `{prediction_prob:.2f}`")

        if prediction == 1:
            st.error("⚠️ This customer is likely to churn.")
        else:
            st.success("✅ This customer is NOT likely to churn.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.info("Tip: Make sure your input values match the encoded format used during model training.")
