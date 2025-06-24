'''import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Sustainable Banking", layout="wide")
st.title("🌿 Sustainable Banking - Carbon Intelligence Dashboard")

# Load models
models = {
    "Random Forest": joblib.load("rf_best_model.pkl"),
    "Decision Tree": joblib.load("dt_model.pkl"),
    "KNN Regressor": joblib.load("knn_model.pkl")
}
scaler = joblib.load("scaler.pkl")
customer_features = joblib.load("customer_features.pkl")  # must include Customer_ID, Segment, Loyalty_Score

model_name = st.selectbox("Select Prediction Model", list(models.keys()))
model = models[model_name]

uploaded_file = st.file_uploader("📤 Upload your transaction CSV", type="csv")

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Raw Data")
    st.write(df_raw.head())

    if 'Customer_ID' not in df_raw.columns:
        st.error("🚫 Your CSV must contain 'Customer_ID' column.")
    else:
        original_ids = df_raw[['Customer_ID']].copy()  # to keep for later

        # Preprocessing
        df = df_raw.copy()
        df['Amount_Spent'] = np.log1p(df['Amount_Spent'])
        df = pd.get_dummies(df, columns=['Transaction_Type', 'Payment_Method', 'Merchant_Category', 'Location'], drop_first=True)
        df['Spending_Frequency_Interaction'] = df['Amount_Spent'] * df['Frequency']

        # Define or load expected columns
        try:
            required_cols = model.feature_names  # if saved in model
        except:
            # Define manually here if model.feature_names not saved
            required_cols = df.columns.tolist()  # fallback, adjust if needed

        for col in required_cols:
            if col not in df.columns:
                df[col] = 0  # fill missing columns
        df = df[required_cols]

        # Predict
        X_scaled = scaler.transform(df)
        df['Predicted_Emission'] = model.predict(X_scaled)

        # Merge Customer_ID back
        df = pd.concat([original_ids.reset_index(drop=True), df], axis=1)

        st.subheader(f"📈 Predicted Emissions using {model_name}")
        st.write(df[['Customer_ID', 'Predicted_Emission']])
        st.line_chart(df['Predicted_Emission'])

        # Regulatory Compliance
        thresholds = {
            "Electricity": 0.50, "Flights": 2.00, "Fuel": 3.00,
            "Groceries": 0.20, "Hotel Stay": 1.20, "Online Shopping": 0.30,
            "Public Transport": 0.10, "Dining": 0.35, "Ride Sharing": 0.90
        }

        def get_type(row):
            for key in thresholds:
                col = f'Transaction_Type_{key}'
                if col in row and row[col] == 1:
                    return key
            return "Unknown"

        df["Transaction_Type"] = df.apply(get_type, axis=1)
        df["Expected_Emission"] = df["Amount_Spent"] * df["Transaction_Type"].map(thresholds).fillna(0)
        df["Excess_Emission"] = df["Predicted_Emission"] - df["Expected_Emission"]

        def compliance_score(excess):
            if excess <= -15: return 5
            elif excess <= -7: return 4
            elif excess <= -2: return 3
            elif excess <= 0: return 2
            else: return 1

        df["Regulatory_Compliance_Score"] = df["Excess_Emission"].apply(compliance_score)

        # Merge with customer_features
        merged_df = pd.merge(df, customer_features[['Customer_ID', 'Segment', 'Loyalty_Score']], on='Customer_ID', how='left')

        st.subheader("🏷 Segment, Loyalty & Compliance Scores")
        st.write(merged_df[['Customer_ID', 'Segment', 'Loyalty_Score', 'Regulatory_Compliance_Score']])

        with st.expander("📊 Segment Distribution"):
            st.bar_chart(merged_df['Segment'].value_counts())

        with st.expander("🏅 Loyalty Score Breakdown"):
            st.bar_chart(merged_df['Loyalty_Score'].value_counts())

        with st.expander("⚖️ Compliance Score Overview"):
            st.bar_chart(merged_df['Regulatory_Compliance_Score'].value_counts())

else:
    st.info("Upload a CSV file with transaction data including 'Customer_ID'.")'''

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Sustainable Banking", layout="wide")

# 🌿 Stylish Title and Subtitle
st.markdown("""
    <h1 style='text-align: center; color: #2E8B57;'>🌿 Sustainable Banking Dashboard</h1>
    <h4 style='text-align: center; color: gray;'>Track Emissions, Segment Customers, and Ensure Compliance</h4>
    <hr style="border:2px solid #2E8B57">
""", unsafe_allow_html=True)

# Load model and data
model = joblib.load("rf_best_model.pkl")
scaler = joblib.load("scaler.pkl")
customer_features = joblib.load("customer_features.pkl")

uploaded_file = st.file_uploader("📄 Upload your transaction CSV", type="csv")

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Raw Data")
    st.write(df_raw.head())

    if 'Customer_ID' not in df_raw.columns:
        st.error("🛘 Your CSV must contain 'Customer_ID' column.")
    else:
        original_ids = df_raw[['Customer_ID']].copy()

        # Preprocessing
        df = df_raw.copy()
        df['Amount_Spent'] = np.log1p(df['Amount_Spent'])
        df = pd.get_dummies(df, columns=['Transaction_Type', 'Payment_Method', 'Merchant_Category', 'Location'], drop_first=True)
        df['Spending_Frequency_Interaction'] = df['Amount_Spent'] * df['Frequency']

        required_cols = model.feature_names if hasattr(model, 'feature_names') else df.columns.tolist()
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[required_cols]

        # Predict
        X_scaled = scaler.transform(df)
        df['Predicted_Emission'] = model.predict(X_scaled)
        df = pd.concat([original_ids.reset_index(drop=True), df], axis=1)

        # Tabs for Navigation
        tab1, tab2, tab3 = st.tabs(["📈 Predictions", "📊 Clustering", "⚖ Compliance"])

        with tab1:
            st.subheader("📈 Emissions - Random Forest")
            st.write(df[['Customer_ID', 'Predicted_Emission']])
            st.line_chart(df['Predicted_Emission'])

        # Add Compliance Info
        thresholds = {
            "Electricity": 0.50, "Flights": 2.00, "Fuel": 3.00,
            "Groceries": 0.20, "Hotel Stay": 1.20, "Online Shopping": 0.30,
            "Public Transport": 0.10, "Dining": 0.35, "Ride Sharing": 0.90
        }

        def get_type(row):
            for key in thresholds:
                col = f'Transaction_Type_{key}'
                if col in row and row[col] == 1:
                    return key
            return "Unknown"

        df["Transaction_Type"] = df.apply(get_type, axis=1)
        df["Expected_Emission"] = df["Amount_Spent"] * df["Transaction_Type"].map(thresholds).fillna(0)
        df["Excess_Emission"] = df["Predicted_Emission"] - df["Expected_Emission"]

        def compliance_score(excess):
            if excess <= -15: return 5
            elif excess <= -7: return 4
            elif excess <= -2: return 3
            elif excess <= 0: return 2
            else: return 1

        df["Regulatory_Compliance_Score"] = df["Excess_Emission"].apply(compliance_score)

        # Merge with customer_features
        merged_df = pd.merge(df, customer_features[['Customer_ID', 'Segment', 'Loyalty_Score']], on='Customer_ID', how='left')

        # Show metrics at the top
        st.markdown("<hr style='border:1px dashed #bbb'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric("Top Eco Score", merged_df['Loyalty_Score'].max())
        col2.metric("Compliant %", f"{(merged_df['Regulatory_Compliance_Score'] >= 3).mean()*100:.1f}%")
        st.markdown("<hr style='border:1px dashed #bbb'>", unsafe_allow_html=True)

        with tab2:
            st.subheader("📊 Customer Segments & Loyalty")
            st.write(merged_df[['Customer_ID', 'Segment', 'Loyalty_Score']])
            st.bar_chart(merged_df['Segment'].value_counts())
            st.bar_chart(merged_df['Loyalty_Score'].value_counts())

        with tab3:
            st.subheader("⚖ Regulatory Compliance Scores")
            st.write(merged_df[['Customer_ID', 'Regulatory_Compliance_Score']])
            score_counts = merged_df['Regulatory_Compliance_Score'].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
            st.bar_chart(score_counts)

        # Footer
        st.markdown("""
            <hr>
            <center>👩‍💻 Created by Deepika | Sustainable AI Banking ✨</center>
        """, unsafe_allow_html=True)
else:
    st.info("Upload a CSV file with transaction data including 'Customer_ID'.")

