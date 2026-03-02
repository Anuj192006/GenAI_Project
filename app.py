import os
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnPredictor AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0e1117 0%, #161b27 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #13161f !important;
    border-right: 1px solid #2a2d3e;
}

/* Title */
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #ff4b4b, #ff8c42);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-sub {
    color: #8892a4;
    font-size: 1.05rem;
    margin-top: 0;
}

/* Metric Cards */
.metric-container {
    background: #1a1c2e;
    border: 1px solid #2a2d3e;
    border-left: 4px solid #ff4b4b;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
}
.metric-label {
    font-size: 0.85rem;
    color: #8892a4;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Predict Button */
div[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    background: linear-gradient(90deg, #ff4b4b, #ff8c42);
    color: white;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 8px;
    height: 3.2em;
    cursor: pointer;
    transition: all 0.3s ease;
    letter-spacing: 1px;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(255, 75, 75, 0.4);
}

/* Divider Color */
hr {
    border-color: #2a2d3e;
}

/* Input styling */
.stSelectbox > div, .stNumberInput > div {
    background-color: #1a1c2e !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Train Model (cached so it only runs once per session) ─────────────────────
@st.cache_resource(show_spinner=False)
def train_model():
    # Resolve CSV path relative to this script so it works both locally and on Streamlit Cloud
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "7 churn.csv")
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Use assignment instead of inplace=True (deprecated in newer pandas)
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train_sc, y_train)

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_sc, y_train)

    metrics = {
        "Logistic Regression": {
            "Accuracy":  round(accuracy_score(y_test, log_model.predict(X_test_sc)), 4),
            "Precision": round(precision_score(y_test, log_model.predict(X_test_sc)), 4),
            "Recall":    round(recall_score(y_test, log_model.predict(X_test_sc)), 4),
            "F1 Score":  round(f1_score(y_test, log_model.predict(X_test_sc)), 4),
        },
        "Decision Tree": {
            "Accuracy":  round(accuracy_score(y_test, dt_model.predict(X_test_sc)), 4),
            "Precision": round(precision_score(y_test, dt_model.predict(X_test_sc)), 4),
            "Recall":    round(recall_score(y_test, dt_model.predict(X_test_sc)), 4),
            "F1 Score":  round(f1_score(y_test, dt_model.predict(X_test_sc)), 4),
        },
    }

    return log_model, dt_model, scaler, encoders, feature_names, metrics


# ─── Load & Train ──────────────────────────────────────────────────────────────
with st.spinner("🔄 Initializing AI Models... please wait"):
    log_model, dt_model, scaler, encoders, feature_names, metrics = train_model()


# ─── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🛡️ ChurnPredictor AI")
st.sidebar.markdown("*Enterprise-grade Customer Retention Intelligence*")
st.sidebar.divider()

selected_model_name = st.sidebar.radio(
    "**Select Model**",
    ["Logistic Regression", "Decision Tree"],
    index=0
)

st.sidebar.divider()
st.sidebar.markdown("### 📊 Model Performance")
m = metrics[selected_model_name]
for k, v in m.items():
    st.sidebar.metric(k, f"{v*100:.2f}%")

st.sidebar.divider()
st.sidebar.caption("Telco Customer Churn Dataset · Scikit-Learn · Streamlit")


# ─── Main Header ───────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🛡️ ChurnPredictor AI</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Predict customer churn risk with machine learning — powered by Logistic Regression & Decision Trees.</p>', unsafe_allow_html=True)
st.divider()


# ─── Input Form ────────────────────────────────────────────────────────────────
with st.form("churn_form"):
    st.markdown("### 📋 Customer Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        st.markdown("**📶 Services**")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("**💳 Billing**")
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0, step=1.0)

    st.divider()
    submitted = st.form_submit_button("🔥 CALCULATE CHURN RISK")


# ─── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    input_dict = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = pd.DataFrame([input_dict])

    for col, le in encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)

    active_model = log_model if selected_model_name == "Logistic Regression" else dt_model
    prediction = active_model.predict(input_scaled)[0]
    probability = active_model.predict_proba(input_scaled)[0]

    st.divider()
    st.markdown("### 🔍 Prediction Result")

    res_col1, res_col2, res_col3 = st.columns([1, 1, 1])

    with res_col1:
        if prediction == 1:
            st.error("## ⚠️ HIGH CHURN RISK")
        else:
            st.success("## ✅ CUSTOMER RETAINED")

    with res_col2:
        churn_pct = f"{probability[1]*100:.1f}%"
        retain_pct = f"{probability[0]*100:.1f}%"
        st.metric("Churn Probability", churn_pct)
        st.metric("Retention Probability", retain_pct)

    with res_col3:
        confidence = max(probability)
        st.write("**Model Confidence**")
        st.progress(float(confidence))
        st.caption(f"{confidence*100:.1f}% confident — using {selected_model_name}")

    st.divider()
    st.markdown("### 💡 Recommended Actions")
    if prediction == 1:
        st.warning("""
- 🎯 **Offer a loyalty discount** on a long-term contract to improve retention.
- 📞 **Schedule a proactive support call** to address any dissatisfaction.
- 🎁 **Provide an upgrade incentive** for premium services like Fiber optic or streaming bundles.
- 📧 **Send a personalised retention email** with an exclusive offer within 48 hours.
        """)
    else:
        st.info("""
- 📈 **Target for upselling** — ideal candidate for premium add-ons.
- 🏷️ **Early renewal program** — lock in a longer contract with a small incentive.
- ⭐ **Loyalty rewards** — enrol in a customer satisfaction program.
        """)
