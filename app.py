
# app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import plotly.express as px

st.set_page_config(page_title="AI Fraud Detection & Risk Scoring", layout="wide", page_icon="🛡️")

# Sidebar
st.sidebar.title("🛠️ Controls")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use bundled sample dataset", value=not bool(uploaded))

contamination = st.sidebar.slider("Anomaly % (IsolationForest contamination)", 0.0, 0.2, 0.03, 0.01)
test_size = st.sidebar.slider("Test size (LogReg)", 0.1, 0.5, 0.2, 0.05)
threshold = st.sidebar.slider("Fraud threshold (probability)", 0.1, 0.9, 0.5, 0.05)

@st.cache_data
def load_sample():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "user_id": [f"U{i:04d}" for i in range(n)],
        "age": rng.integers(18, 80, size=n),
        "country": rng.choice(["India","USA","UK","China","Germany","Brazil"], size=n),
        "amount": rng.lognormal(mean=4, sigma=1.0, size=n).round(2),
        "payment_method": rng.choice(["UPI","Debit Card","Credit Card","Netbanking"], size=n),
        "device": rng.choice(["Android","iPhone","Windows","Mac"], size=n),
        "transaction_time": pd.Timestamp("2025-07-01") + pd.to_timedelta(rng.integers(0, 60*24*30, size=n), unit="m"),
        "is_fraud": rng.choice([0,1], p=[0.9,0.1], size=n)
    })
    return df

if uploaded is not None and not use_sample:
    df = pd.read_csv(uploaded)
else:
    df = load_sample()

st.title("🛡️ AI-Powered Fraud Detection & Risk Scoring")
st.dataframe(df.head(), use_container_width=True)

time_col_candidates = [c for c in df.columns if "time" in c.lower()]
time_col = time_col_candidates[0] if time_col_candidates else None

work = df.copy()
if time_col:
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work["hour"] = work[time_col].dt.hour
    work["dayofweek"] = work[time_col].dt.dayofweek
    work["month"] = work[time_col].dt.month
else:
    work["hour"] = 0
    work["dayofweek"] = 0
    work["month"] = 0

target = "is_fraud" if "is_fraud" in work.columns else None
num_cols = [c for c in work.select_dtypes(include=np.number).columns if c != target]
cat_cols = [c for c in work.columns if c not in num_cols + ([target] if target else [])]

# Isolation Forest
if num_cols:
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(work[num_cols])
    work["anomaly_score"] = iso.decision_function(work[num_cols])
    work["anomaly_flag"] = (iso.predict(work[num_cols]) == -1).astype(int)

    st.subheader("Anomaly Score Distribution")
    st.plotly_chart(px.histogram(work, x="anomaly_score", nbins=50))

# Logistic Regression
if target:
    X = work.drop(columns=[target])
    y = work[target].astype(int)

    preproc = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols)
        ]
    )

    model = Pipeline(steps=[
        ("prep", preproc),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    model.fit(X_train, y_train)

    proba_all = model.predict_proba(X)[:,1]
    work["fraud_probability"] = proba_all
    work["risk_score"] = work["fraud_probability"] * work["amount"].fillna(0)
    work["predicted_fraud"] = (work["fraud_probability"] >= threshold).astype(int)

    st.subheader("Fraud Probability Distribution")
    st.plotly_chart(px.histogram(work, x="fraud_probability", nbins=50))

    st.subheader("Top High-Risk Transactions")
    st.dataframe(work.sort_values("risk_score", ascending=False).head(20))

    if "country" in work.columns:
        st.subheader("Risk by Country")
        by_country = work.groupby("country", as_index=False).agg(
            avg_risk=("risk_score","mean"),
            fraud_rate=("predicted_fraud","mean"),
            txns=("country","count")
        )
        st.plotly_chart(px.scatter(by_country, x="fraud_rate", y="avg_risk", size="txns", color="country", hover_name="country"))

# Download
csv = work.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV with anomaly & risk scores", csv, file_name="scored_transactions.csv")
