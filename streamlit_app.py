import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle  # For loading the predictive model

# Streamlit App Title
st.title("Breast Cancer Data Analysis and Prediction")

# File Upload Section
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load the dataset
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Overview")
        st.write("First 5 Rows of the Dataset:")
        st.dataframe(df.head())

        # Dataset Information
        st.subheader("Dataset Information")
        with st.expander("Click to view dataset info"):
            buffer = pd.io.formats.format.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

        # Checking for duplicates
        duplicates = df.duplicated().sum()
        st.write(f"Number of duplicate rows: {duplicates}")

        # Checking for missing values
        missing_values = df.isnull().sum()
        st.write("Missing Values:")
        st.dataframe(missing_values)

        # Summary Statistics
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())

        # Value Counts for Target Variable
        if 'diagnosis' in df.columns:
            st.subheader("Target Variable Distribution")
            diagnosis_counts = df['diagnosis'].value_counts()
            st.bar_chart(diagnosis_counts)

        # Data Visualization
        st.subheader("Data Visualization")
        st.write("Correlation Heatmap")

        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            st.write("No numeric columns available for correlation heatmap.")
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.write("Please upload a dataset to begin analysis.")

# Predictive System
st.sidebar.header("Breast Cancer Prediction")

# Load the predictive model
model_file = st.sidebar.file_uploader("Upload your trained model (e.g., .pkl file)", type=["pkl"])
if model_file is not None:
    try:
        # Load the model
        model = pickle.load(model_file)

        # Input form for predictions
        st.subheader("Make a Prediction")
        st.write("Provide the input features below:")

        # Input fields for each feature
        mean_radius = st.number_input("Mean Radius")
        mean_texture = st.number_input("Mean Texture")
        mean_perimeter = st.number_input("Mean Perimeter")
        mean_area = st.number_input("Mean Area")
        mean_smoothness = st.number_input("Mean Smoothness")
        mean_compactness = st.number_input("Mean Compactness")
        mean_concavity = st.number_input("Mean Concavity")
        mean_concave_points = st.number_input("Mean Concave Points")
        mean_symmetry = st.number_input("Mean Symmetry")
        mean_fractal_dimension = st.number_input("Mean Fractal Dimension")

        # Collect input features
        input_data = pd.DataFrame({
            "mean_radius": [mean_radius],
            "mean_texture": [mean_texture],
            "mean_perimeter": [mean_perimeter],
            "mean_area": [mean_area],
            "mean_smoothness": [mean_smoothness],
            "mean_compactness": [mean_compactness],
            "mean_concavity": [mean_concavity],
            "mean_concave_points": [mean_concave_points],
            "mean_symmetry": [mean_symmetry],
            "mean_fractal_dimension": [mean_fractal_dimension],
        })

        # Make prediction
        if st.button("Predict"):
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]

            st.subheader("Prediction Result")
            if prediction == 1:
                st.write("The model predicts: **Malignant**")
            else:
                st.write("The model predicts: **Benign**")

            st.write(f"Prediction Probabilities: {prediction_proba}")
    except Exception as e:
        st.error(f"Failed to load the model: {e}")

