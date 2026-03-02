import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# Set Page Config
st.set_page_config(
    page_title="ChurnPredictor AI | Premium Enterprise Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }
    .metric-card {
        background-color: #1a1c24;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
    }
    h1 {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    p {
        color: #a0a0a0;
    }
    </style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    try:
        with open("telco_churn_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file 'telco_churn_model.pkl' not found. Please run the training script first.")
        return None

model_data = load_model()

if model_data:
    model = model_data["model"]
    scaler = model_data["scaler"]
    encoders = model_data["encoders"]
    feature_names = model_data["feature_names"]

    # Sidebar Header
    st.sidebar.image("https://img.icons8.com/plasticine/100/000000/shield.png", width=80)
    st.sidebar.title("ChurnPredictor AI")
    st.sidebar.info("Enterprise-grade Customer Retention Intelligence")

    # Main Area
    st.title("🛡️ ChurnPredictor AI")
    st.markdown("### Predicting Customer Retention with Precision Neural Analytics")
    st.write("Fill in the customer details below to calculate the churn probability.")

    # Input Sections using Expander for Cleaner UI
    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("👤 Demographic Info")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        with col2:
            st.subheader("📶 Service Usage")
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

        with col3:
            st.subheader("💳 Billing Details")
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

        st.divider()
        col4, col5 = st.columns(2)
        with col4:
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        with col5:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)

        submitted = st.form_submit_button("🔥 CALCULATE RETENTION RISK")

    if submitted:
        with st.status("Analyzing Customer Behavior...", expanded=True) as status:
            time.sleep(1)
            st.write("Retrieving historical patterns...")
            time.sleep(1)
            st.write("Running Neural Logistic Regression...")
            
            # Prepare Input
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
                "TotalCharges": total_charges
            }

            input_df = pd.DataFrame([input_dict])

            # Apply Encoders
            for col in encoders:
                if col in input_df.columns:
                    input_df[col] = encoders[col].transform(input_df[col])

            # Ensure Column Order
            input_df = input_df[feature_names]
            
            # Scale
            input_scaled = scaler.transform(input_df)

            # Predict
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # Output Display
        st.divider()
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if prediction == 1:
                st.error("### ⚠️ HIGH RISK OF CHURN")
                st.metric("Churn Probability", f"{probability[1]*100:.2f}%")
            else:
                st.success("### ✅ LOW RISK / RETAINED")
                st.metric("Retention Probability", f"{probability[0]*100:.2f}%")

        with res_col2:
            st.write("#### Confidence Score")
            st.progress(max(probability))
            st.info(f"The model is {max(probability)*100:.1f}% confident in this prediction.")

        # Recommendations
        st.markdown("---")
        st.subheader("💡 Strategic Recommendations")
        if prediction == 1:
            st.warning("- Offer a loyalty discount on a longer-term contract.\n- Schedule a preemptive support call to ensure satisfaction.\n- Consider hardware upgrade incentives.")
        else:
            st.info("- Customer is stable. Target for upselling premium services.\n- Excellent candidate for early renewal programs.")

else:
    st.warning("⚠️ Application is waiting for the model. Please make sure 'telco_churn_model.pkl' exists in the project root.")
