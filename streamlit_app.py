import streamlit as st

st.title('Breast cancer prediaction')

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Introduction
st.title("Breast Cancer Classification")
st.write("This application uses machine learning to classify breast cancer as malignant or benign based on input features.")

# Load Data
@st.cache
def load_data():
    # Replace with the actual path or method to load your dataset
    data = pd.read_csv('breast_cancer_data.csv')  # Example path
    return data

data = load_data()

# Data Exploration
st.header("Data Exploration")
if st.checkbox("Show raw data"):
    st.write(data.head())

st.write("Summary Statistics:")
st.write(data.describe())

# Preprocessing and Model Training
st.header("Model Training")
target_column = st.selectbox("Select Target Column", data.columns, index=len(data.columns) - 1)
features = st.multiselect("Select Features", data.columns[:-1].tolist(), default=data.columns[:-1].tolist())

if features and target_column:
    X = data[features]
    y = data[target_column]

    # Data splitting
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

# Predictions
st.header("Make Predictions")
input_data = {}
for feature in features:
    input_data[feature] = st.number_input(f"Input {feature}", value=float(data[feature].mean()))

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.write(f"Prediction: {'Malignant' if prediction[0] == 1 else 'Benign'}")
