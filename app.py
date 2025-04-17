import numpy as np
import pandas as pd
import streamlit as st
import bz2
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load model from compressed pickle
with bz2.BZ2File("liver_disease_model.pbz2", 'rb') as f:
    model = pickle.load(f)

# Load sample data (used to scale new inputs)
df = pd.read_csv("train.csv", encoding='unicode_escape')
df.columns = df.columns.str.strip()
df['Gender'] = df['Gender'].str.strip().str.capitalize()

# Define features
num_features = ['Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'AAP', 'SAA_1',
                'SAA_2', 'Total_Protiens', 'ALB_Albumin', 'A/G_RATIO']
cat_features = ['Gender']

# Drop rows with missing values for consistency
df_cleaned = df.dropna(subset=num_features + cat_features + ['Result'])

# Preprocessing: encode 'Gender' manually
df_cleaned['Gender'] = df_cleaned['Gender'].map({'Male': 0, 'Female': 1})
X = df_cleaned[num_features + cat_features].copy()

# Fit scaler for numerical columns
scaler = MinMaxScaler()
X[num_features] = scaler.fit_transform(X[num_features])

# Streamlit UI
st.set_page_config(page_title="Liver Disease Predictor", layout="centered")
st.title("🧬 Liver Disease Prediction System")
st.markdown("Enter the values below to predict liver disease status:")

# Input fields
age = st.number_input("Age", min_value=0.0, format="%.2f")
total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, format="%.2f")
direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, format="%.2f")
aap = st.number_input("Alkaline Amino Phosphatase (AAP)", min_value=0.0, format="%.2f")
saa_1 = st.number_input("SAA_1", min_value=0.0, format="%.2f")
saa_2 = st.number_input("SAA_2", min_value=0.0, format="%.2f")
total_proteins = st.number_input("Total Proteins", min_value=0.0, format="%.2f")
alb_albumin = st.number_input("Albumin", min_value=0.0, format="%.2f")
ag_ratio = st.number_input("A/G Ratio", min_value=0.0, format="%.2f")
gender = st.selectbox("Gender", options=["Male", "Female"])

if st.button("Predict"):
    # Map gender to numerical value
    gender_val = 0 if gender == "Male" else 1

    # Create input array and scale
    input_data = np.array([[age, total_bilirubin, direct_bilirubin, aap, saa_1,
                            saa_2, total_proteins, alb_albumin, ag_ratio, gender_val]])
    input_df = pd.DataFrame(input_data, columns=num_features + cat_features)

    # Scale numeric columns only
    input_df[num_features] = scaler.transform(input_df[num_features])

    # Make prediction
    prediction = model.predict(input_df)

    # Output result
    if prediction[0]  < 0.33:
        st.success("🟢 Low Risk – You're likely healthy.")
    elif prediction[0] < 0.66:
        st.warning("🟠 Moderate Risk – Consider lifestyle changes and regular check-ups.")
    else:
        st.error("🔴 High Risk – Please consult a liver specialist immediately.")
