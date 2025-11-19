# app.py (updated - robust categorical widgets + encoder handling)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="MTN Churn Prediction", layout="wide")

# --- Helpers -----------------------------------------------------------------
def load_model_files():
    model = joblib.load("churn_model.pkl") if os.path.exists("churn_model.pkl") else joblib.load("heart_disease_model.pkl")
    features = joblib.load("feature_columns.pkl")
    encoders = None
    if os.path.exists("encoders.pkl"):
        encoders = joblib.load("encoders.pkl")
    return model, features, encoders

def build_encoders_from_csv(csv_path, feature_columns):
    """
    If encoders.pkl missing, try to read the training CSV from repo
    and build LabelEncoders for categorical features.
    """
    if not os.path.exists(csv_path):
        return {}
    df_ref = pd.read_csv(csv_path)
    # normalize columns names to match
    df_ref.columns = [c.strip() for c in df_ref.columns]
    enc_map = {}
    for col in feature_columns:
        if col not in df_ref.columns:
            continue
        if df_ref[col].dtype == 'object' or df_ref[col].dtype.name == 'category':
            # take unique non-null values
            uniques = df_ref[col].dropna().astype(str).str.strip().str.title().unique().tolist()
            # create label encoder and fit to uniques
            le = LabelEncoder()
            # If only numeric strings, convert to numeric first
            try:
                le.fit(uniques)
            except Exception:
                le.fit([str(u) for u in uniques])
            enc_map[col] = {"encoder": le, "classes": uniques}
    return enc_map

def preprocess_user_input(user_dict, feature_columns, encoders_map):
    df = pd.DataFrame([user_dict])
    # strip and title for strings
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            # unify date-like strings to YYYYMMDD if possible
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notna().any():
                    df[col] = parsed.dt.strftime("%Y%m%d").astype(float)
            except Exception:
                pass

    # map Yes/No to 1/0
    binary_map = {"yes": 1, "no": 0, "y": 1, "n": 0}
    for col in df.columns:
        df[col] = df[col].apply(lambda x: binary_map.get(str(x).strip().lower(), x))

    # apply encoders if available
    for col, enc_info in encoders_map.items():
        if col in df.columns:
            le = enc_info["encoder"]
            # handle unseen labels by adding them if possible, otherwise map to -1
            try:
                df[col] = df[col].astype(str).apply(lambda v: v.strip().title())
                df[col] = le.transform(df[col])
            except Exception:
                # unseen labels -> try to handle gracefully
                mapped = []
                for val in df[col].astype(str).tolist():
                    v = val.strip().title()
                    if v in getattr(le, "classes_", []):
                        mapped.append(int(np.where(le.classes_ == v)[0][0]))
                    else:
                        # fallback: index -1 (will be coerced to 0 later)
                        mapped.append(-1)
                df[col] = mapped

    # convert numeric-like to numbers
    df = df.apply(pd.to_numeric, errors='coerce')

    # fillna with 0
    df = df.fillna(0)

    # reindex columns to feature order
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df

# --- Load model + features + encoders ---------------------------------------
try:
    model, feature_columns, encoders = load_model_files()
except Exception as e:
    st.error(f"Failed to load model or feature files: {e}")
    st.stop()

# If encoders missing, try to build from csv in repo
if encoders is None:
    encoders_map = build_encoders_from_csv("mtn_customer_churn.csv", feature_columns)
    # wrap LabelEncoders into same structure as encoders.pkl would have used
    encoders = {}
    for k, v in encoders_map.items():
        encoders[k] = {"encoder": v["encoder"], "classes": list(v["encoder"].classes_)}
else:
    # if loaded encoders dict stored encoder objects, use directly
    # expected structure: {col: {"encoder": LabelEncoder(), "classes": [...]} }
    pass

# --- Build UI ----------------------------------------------------------------
st.title("📱 MTN Customer Churn Prediction")
st.write("Provide customer details. Use the dropdowns where available to match training categories.")

st.sidebar.header("Input Customer Data")
user_input = {}

# Attempt to load the CSV to infer option lists (fallback)
csv_df = None
if os.path.exists("mtn_customer_churn.csv"):
    try:
        csv_df = pd.read_csv("mtn_customer_churn.csv")
        csv_df.columns = [c.strip() for c in csv_df.columns]
    except Exception:
        csv_df = None

for col in feature_columns:
    # prefer selectbox if we have encoder/class list or csv unique values
    options = None
    if encoders and col in encoders and "classes" in encoders[col]:
        options = encoders[col]["classes"]
    elif csv_df is not None and col in csv_df.columns:
        # get unique values from csv for this column, dropna and stringify
        uniq = csv_df[col].dropna().astype(str).str.strip().str.title().unique().tolist()
        if 1 < len(uniq) <= 200:  # if many uniques, don't show huge list
            options = uniq

    # Choose widget type
    if options:
        # Prepend a default prompt
        prompt = f"Select {col}"
        # Ensure options are strings
        opts = [str(x) for x in options]
        sel = st.sidebar.selectbox(prompt, options=["--Choose--"] + opts, index=0)
        user_input[col] = "" if sel == "--Choose--" else sel
    else:
        # If column name suggests date
        if "date" in col.lower():
            d = st.sidebar.date_input(col)
            # format as YYYYMMDD integer
            user_input[col] = int(d.strftime("%Y%m%d"))
        else:
            # numeric entry by default, but allow text
            val = st.sidebar.text_input(col, value="")
            # keep as string and let preprocessing coerce
            user_input[col] = val

# --- Predict button ----------------------------------------------------------
if st.button("Predict Churn"):
    try:
        # Preprocess user input to numeric vector matching training features
        processed = preprocess_user_input(user_input, feature_columns, encoders or {})
        pred_prob = model.predict_proba(processed)[0][1] if hasattr(model, "predict_proba") else None
        pred = model.predict(processed)[0]

        st.subheader("🔮 Prediction Results")
        if pred_prob is not None:
            st.write(f"**Churn Probability:** {pred_prob:.2f}")
        st.write("**Prediction:** " + ("Likely to Churn" if int(pred) == 1 else "Not Likely to Churn"))
    except Exception as e:
        st.error(f"Error during prediction: {e}")
