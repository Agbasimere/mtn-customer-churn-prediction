# app.py - MTN Customer Churn Prediction Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sqlite3
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="MTN Churn Prediction", layout="wide")

# ---------------------------
# Helper: load model and feature columns
# ---------------------------
MODEL_FILENAMES = ["churn_model.pkl", "churn_model.sav", "model.pkl", "model/churn_model.pkl"]
FEATURES_FILENAME = "feature_columns.pkl"
CSV_FILENAME = "mtn_customer_churn.csv"

def try_load_model():
    for fn in MODEL_FILENAMES:
        if os.path.exists(fn):
            try:
                return joblib.load(fn), fn
            except Exception as e:
                st.error(f"Found model file {fn} but failed to load it: {e}")
                return None, fn
    return None, None

model, found_model_file = try_load_model()
if model is None:
    st.warning("No trained model file found in repository root. Please upload 'churn_model.pkl' (or a supported filename). App will still show the input form but predictions won't run.")
else:
    st.success(f"Loaded model file: {found_model_file}")

# Load feature columns if available; else infer from CSV (excluding obvious non-features)
if os.path.exists(FEATURES_FILENAME):
    try:
        feature_columns = joblib.load(FEATURES_FILENAME)
        st.info("Loaded feature_columns from file.")
    except Exception as e:
        st.error(f"Could not load {FEATURES_FILENAME}: {e}")
        feature_columns = None
else:
    feature_columns = None

# If CSV available, read it to build encoders and also to infer features if needed
encoders = {}
csv_df = None
if os.path.exists(CSV_FILENAME):
    try:
        csv_df = pd.read_csv(CSV_FILENAME)
        st.info(f"Loaded {CSV_FILENAME} to recreate label encoders (recommended).")
    except Exception as e:
        st.error(f"Error loading {CSV_FILENAME}: {e}")
        csv_df = None
else:
    st.warning(f"{CSV_FILENAME} not found in repo. To ensure categorical inputs map exactly to training encodings, add your CSV or save encoders during training.")

# Infer default feature list if feature_columns.pkl not present
if feature_columns is None:
    if csv_df is not None:
        # Heuristic: use all columns except identifiers and the target column
        exclude = {"Customer ID","Full Name","Customer Churn Status","Reasons for Churn"}
        feature_columns = [c for c in csv_df.columns if c not in exclude]
        st.info("Feature columns inferred from CSV (place feature_columns.pkl to skip this inference).")
    else:
        # fallback default (common fields) - adjust if yours differ
        feature_columns = [
            "Date of Purchase","Age","State","MTN Device","Gender",
            "Satisfaction Rate","Customer Review","Customer Tenure in months",
            "Subscription Plan","Unit Price","Number of Times Purchased",
            "Total Revenue","Data Usage"
        ]
        st.warning("Using fallback feature list. This may not match the training-time features.")

# Build encoders for categorical columns using CSV (so mapping matches training)
categorical_columns = []
if csv_df is not None:
    for col in csv_df.select_dtypes(include=['object']).columns:
        if col in feature_columns:
            vals = csv_df[col].fillna("<<MISSING>>").astype(str).str.strip().str.title().unique().tolist()
            # Create and fit LabelEncoder on CSV values (ensures mapping as during training)
            le = LabelEncoder()
            le.fit([str(v) for v in vals])
            encoders[col] = le
            categorical_columns.append(col)

# SQLite logging setup
DB_FILE = "churn_predictions.db"
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        input_json TEXT,
        predicted INT,
        probability REAL,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------------------
# Build the input form
# ---------------------------
st.title("📱 MTN Customer Churn Prediction")
st.markdown("Fill the form on the left (sidebar) and click **Predict**. Categorical fields are shown as dropdowns to match training encodings.")

with st.sidebar.form("input_form"):
    st.header("Customer Input")
    user_input = {}
    # We'll treat date column specially and numeric columns specially
    for col in feature_columns:
        # Standardize display name
        disp = col
        # If CSV encoder exists for this column, show dropdown
        if col in encoders:
            options = list(encoders[col].classes_)
            # Put missing sentinel as first if present
            if "<<Missing>>" in options:
                options = options
            user_choice = st.selectbox(disp, options, index=0)
            user_input[col] = user_choice
        else:
            # handle possible date column
            if "date" in col.lower():
                d = st.date_input(disp, value=None)
                if d is None:
                    user_input[col] = ""
                else:
                    # convert to numeric format YYYYMMDD as used in many pipelines
                    user_input[col] = int(d.strftime("%Y%m%d"))
            else:
                # numeric input with some guesses: if 'Rate', 'Price', 'Revenue', 'Usage', use number_input
                if any(x in col.lower() for x in ["age","tenure","price","unit","revenue","usage","number","count","satisfaction","rate","chol","bp"]):
                    # set sensible defaults and ranges
                    user_input[col] = st.number_input(disp, value=0.0, format="%.2f")
                else:
                    # fallback to text input
                    user_input[col] = st.text_input(disp, value="")
    submitted = st.form_submit_button("Predict")

# ---------------------------
# Preprocess the single input row to match training encodings
# ---------------------------
def preprocess_single(input_dict):
    # Convert to DataFrame
    df_input = pd.DataFrame([input_dict])
    # Clean strings
    for c in df_input.columns:
        if df_input[c].dtype == object:
            df_input[c] = df_input[c].astype(str).str.strip().str.title()
    # Apply encoders if available
    for col, le in encoders.items():
        if col in df_input.columns:
            val = df_input.at[0, col]
            # handle missing
            if val is None or (isinstance(val, float) and np.isnan(val)) or str(val).strip()=="":
                df_input.at[0,col] = 0
            else:
                # If value not seen in encoder, add it (assign new label)
                if str(val) not in le.classes_:
                    # extend classes_ and transform
                    new_classes = list(le.classes_) + [str(val)]
                    le.classes_ = np.array(new_classes, dtype=object)
                df_input.at[0,col] = int(le.transform([str(val)])[0])
    # Ensure numeric types where possible
    df_input = df_input.apply(pd.to_numeric, errors="ignore")
    # Reorder to feature_columns and fill missing with 0
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    # Finally convert all to numeric (coerce)
    df_input = df_input.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df_input

# ---------------------------
# Prediction & Logging
# ---------------------------
if submitted:
    if model is None:
        st.error("No trained model loaded. Upload 'churn_model.pkl' to the repo and redeploy.")
    else:
        try:
            processed = preprocess_single(user_input)
            pred_prob = float(model.predict_proba(processed)[0][1]) if hasattr(model, "predict_proba") else None
            pred = int(model.predict(processed)[0])

            # Display results
            st.subheader("🔮 Prediction Results")
            if pred_prob is not None:
                st.write(f"**Churn probability:** {pred_prob:.3f}")
            st.write(f"**Churn prediction (1 = churn, 0 = not churn):** {pred}")

            # Log to sqlite DB
            conn = sqlite3.connect(DB_FILE)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO predictions (input_json, predicted, probability, created_at) VALUES (?, ?, ?, ?)",
                (str(user_input), pred, pred_prob if pred_prob is not None else -1.0, datetime.utcnow().isoformat())
            )
            conn.commit()
            conn.close()

            st.success("Prediction logged to local sqlite database (churn_predictions.db).")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Optional: show recent logs
if st.checkbox("Show recent predictions (local DB)"):
    try:
        conn = sqlite3.connect(DB_FILE)
        df_logs = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC LIMIT 50", conn)
        conn.close()
        st.dataframe(df_logs)
    except Exception as e:
        st.error(f"Could not read local DB: {e}")

st.markdown("---")
st.caption("Note: This UI reconstructs label encoders from the CSV file (mtn_customer_churn.csv). For production, save encoders during training and load them directly to guarantee exact mapping.")
