# app.py (improved) - MTN Customer Churn Prediction Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sqlite3
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import json

st.set_page_config(page_title="MTN Churn Prediction", layout="wide")

# ---------- Config ----------
MODEL_FILENAMES = ["churn_model.pkl", "churn_model.sav", "model.pkl", "model/churn_model.pkl"]
FEATURES_FILENAME = "feature_columns.pkl"
CSV_FILENAME = "mtn_customer_churn.csv"
DB_FILENAME = "churn_predictions.db"  # relative; will be converted to absolute path

# For production persistent DB use (recommended), set env var DATABASE_URL (Postgres/Supabase)
DATABASE_URL = os.environ.get("DATABASE_URL", None)

# ---------- Helpers ----------
def get_repo_root():
    # return absolute folder where app.py lives
    return os.path.abspath(os.path.dirname(__file__))

def get_db_path():
    root = get_repo_root()
    return os.path.join(root, DB_FILENAME)

def try_load_model():
    root = get_repo_root()
    for fn in MODEL_FILENAMES:
        path = os.path.join(root, fn)
        if os.path.exists(path):
            try:
                mdl = joblib.load(path)
                return mdl, path
            except Exception as e:
                st.error(f"Found model file {path} but failed to load it: {e}")
                return None, path
    return None, None

def init_sqlite_db(db_path):
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
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
        return True, None
    except Exception as e:
        return False, str(e)

def log_prediction_sqlite(db_path, input_dict, predicted, probability):
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predictions (input_json, predicted, probability, created_at) VALUES (?, ?, ?, ?)",
            (json.dumps(input_dict, default=str), int(predicted), float(probability) if probability is not None else -1.0, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
        return True, None
    except Exception as e:
        return False, str(e)

# ---------- Load model & features ----------
model, model_path = try_load_model()
if model is None:
    st.warning("No trained model found. Upload churn_model.pkl (or supported filename) to the repo root. Predictions will not be available.")
else:
    st.success(f"Loaded model from: {model_path}")

# load feature columns
root = get_repo_root()
feature_columns = None
features_path = os.path.join(root, FEATURES_FILENAME)
if os.path.exists(features_path):
    try:
        feature_columns = joblib.load(features_path)
        st.info("Loaded feature_columns.pkl")
    except Exception as e:
        st.error(f"Could not load feature_columns.pkl: {e}")

# load CSV to learn encodings if feature_columns missing or to build encoders
encoders = {}
csv_df = None
csv_path = os.path.join(root, CSV_FILENAME)
if os.path.exists(csv_path):
    try:
        csv_df = pd.read_csv(csv_path)
        st.info(f"Loaded CSV: {CSV_FILENAME} (used to build encoders).")
        # create label encoders for categorical features present in feature_columns
        if feature_columns is None:
            exclude = {"Customer ID","Full Name","Customer Churn Status","Reasons for Churn"}
            feature_columns = [c for c in csv_df.columns if c not in exclude]
            st.info("Inferred feature list from CSV.")
        for col in csv_df.select_dtypes(include=['object']).columns:
            if col in feature_columns:
                vals = csv_df[col].fillna("<<MISSING>>").astype(str).str.strip().str.title().unique().tolist()
                le = LabelEncoder()
                le.fit([str(v) for v in vals])
                encoders[col] = le
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
else:
    if feature_columns is None:
        # fallback
        feature_columns = [
            "Date of Purchase","Age","State","MTN Device","Gender",
            "Satisfaction Rate","Customer Review","Customer Tenure in months",
            "Subscription Plan","Unit Price","Number of Times Purchased",
            "Total Revenue","Data Usage"
        ]
        st.warning("CSV not found and no feature_columns.pkl present; using fallback features. This may not match training.")

# ---------- Initialize DB ----------
db_path = get_db_path()

# If using cloud DB (DATABASE_URL), skip sqlite init
if DATABASE_URL:
    st.warning("DATABASE_URL environment variable set. Using external DB. Make sure insert logic is implemented.")
    sqlite_ready = False
else:
    ok, err = init_sqlite_db(db_path)
    sqlite_ready = ok
    if ok:
        st.info(f"SQLite DB initialized at: {db_path}")
    else:
        st.error(f"Failed to initialize SQLite DB: {err}")

# ---------- Build input form ----------
st.title("📱 MTN Customer Churn Prediction")
st.markdown("Fill the form in the sidebar and click **Predict**. Dropdowns are generated from CSV values to match training encodings when possible.")

with st.sidebar.form("input_form"):
    st.header("Customer Input")
    user_input = {}
    for col in feature_columns:
        # pretty display name
        disp = col
        if col in encoders:
            options = list(encoders[col].classes_)
            # add an empty sentinel at front if not present
            if "" not in options:
                options = [""] + options
            choice = st.selectbox(disp, options, index=0)
            user_input[col] = choice
        else:
            # detect date-like columns
            if "date" in col.lower():
                d = st.date_input(disp)
                user_input[col] = int(d.strftime("%Y%m%d")) if d is not None else ""
            else:
                # heuristic numeric fields
                if any(k in col.lower() for k in ["age","tenure","price","unit","revenue","usage","number","count","satisfaction","rate","chol","bp"]):
                    # set sensible defaults
                    val = st.number_input(disp, value=0.0)
                    user_input[col] = val
                else:
                    user_input[col] = st.text_input(disp, value="")
    submitted = st.form_submit_button("Predict")

# ---------- Preprocess single input ----------
def preprocess_single(input_dict):
    df_input = pd.DataFrame([input_dict])
    # normalize strings
    for c in df_input.columns:
        if df_input[c].dtype == object:
            df_input[c] = df_input[c].astype(str).str.strip().str.title()
    # apply encoders
    for col, le in encoders.items():
        if col in df_input.columns:
            val = df_input.at[0, col]
            if val is None or str(val).strip()=="":
                df_input.at[0, col] = 0
            else:
                s = str(val)
                if s not in le.classes_:
                    # extend classes safely
                    new_classes = list(le.classes_) + [s]
                    le.classes_ = np.array(new_classes, dtype=object)
                df_input.at[0, col] = int(le.transform([s])[0])
    # coerce numerics
    df_input = df_input.apply(pd.to_numeric, errors="coerce").fillna(0)
    # reorder and fill
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    return df_input

# ---------- Prediction & logging ----------
if submitted:
    if model is None:
        st.error("No model loaded. Upload model file to the repo root and redeploy.")
    else:
        try:
            processed = preprocess_single(user_input)
            # ensure processed columns align
            st.write("Processed input (first row):")
            st.write(processed.iloc[0].to_dict())

            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(processed)[0][1])
            else:
                prob = None
            pred = int(model.predict(processed)[0])

            st.subheader("🔮 Prediction Results")
            if prob is not None:
                st.write(f"**Churn probability:** {prob:.3f}")
            st.write(f"**Churn prediction (1=churn,0=not):** {pred}")

            # Log result (prefer external DB in production)
            if DATABASE_URL:
                # TODO: implement external DB insert with psycopg2 or supabase
                st.info("DATABASE_URL set: implement external DB logging (not implemented in this script).")
            else:
                ok, err = log_prediction_sqlite(db_path, user_input, pred, prob if prob is not None else -1.0)
                if ok:
                    st.success(f"Prediction logged to SQLite DB at {db_path}")
                else:
                    st.error(f"Failed to log prediction to SQLite DB: {err}")

        except Exception as e:
            st.exception(e)

# ---------- Show recent logs ----------
if st.checkbox("Show recent predictions (local DB)"):
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            df_logs = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC LIMIT 50", conn)
            conn.close()
            st.dataframe(df_logs)
            # show DB file link (only works locally)
            if "localhost" in st.experimental_get_query_params() or not st._is_running_with_streamlit:
                st.write(f"DB path: {db_path}")
        except Exception as e:
            st.error(f"Could not read DB: {e}")
    else:
        st.warning("DB file not found.")

st.markdown("---")
st.caption("For production-grade persistence use a managed DB (Supabase, Postgres). SQLite here is intended for local testing and demo only.")
