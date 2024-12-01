import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle  # For loading the predictive model
from sklearn.preprocessing import StandardScaler  # For scaling (if needed)

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
scaler_file = st.sidebar.file_uploader("Upload your scaler (e.g., .pkl file, optional)", type=["pkl"])

if model_file is not None:
    try:
        # Load the model
        model = pickle.load(model_file)

        # Load the scaler (if provided)
        scaler = None
        if scaler_file is not None:
            scaler = pickle.load(scaler_file)

        # Display expected feature names
        if hasattr(model, "feature_names_in_"):
            st.write("The model expects these feature names:")
            st.write(model.feature_names_in_)

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

        # Collect input features with correct names
        input_data = pd.DataFrame({
            "radius_mean": [mean_radius],
            "texture_mean": [mean_texture],
            "perimeter_mean": [mean_perimeter],
            "area_mean": [mean_area],
            "smoothness_mean": [mean_smoothness],
            "compactness_mean": [mean_compactness],
            "concavity_mean": [mean_concavity],
            "concave points_mean": [mean_concave_points],
            "symmetry_mean": [mean_symmetry],
            "fractal_dimension_mean": [mean_fractal_dimension],
        })

        # Apply scaler if available
        if scaler:
            input_data_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)
        else:
            input_data_scaled = input_data  # Use raw input if no scaler is provided

        # Make prediction
        if st.button("Predict"):
            prediction = model.predict(input_data_scaled)[0]
            prediction_proba = model.predict_proba(input_data_scaled)[0]

            st.subheader("Prediction Result")
            if prediction == 1:
                st.write("The model predicts: **Malignant**")
            else:
                st.write("The model predicts: **Benign**")

            st.write(f"Prediction Probabilities: {prediction_proba}")

    except Exception as e:
        st.error(f"Failed to load the model or scaler: {e}")
else:
    st.write("Please upload the model file to make predictions.")
    # Load the predictive model
model_file = st.sidebar.file_uploader("Upload your trained pipeline (e.g., model.pkl)", type=["pkl"])

if model_file is not None:
    try:
        # Load the pipeline
        model = pickle.load(model_file)

        # Display expected feature names
        if hasattr(model.named_steps["classifier"], "feature_names_in_"):
            st.write("The model expects these feature names:")
            st.write(model.named_steps["classifier"].feature_names_in_)

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
            "radius_mean": [mean_radius],
            "texture_mean": [mean_texture],
            "perimeter_mean": [mean_perimeter],
            "area_mean": [mean_area],
            "smoothness_mean": [mean_smoothness],
            "compactness_mean": [mean_compactness],
            "concavity_mean": [mean_concavity],
            "concave points_mean": [mean_concave_points],
            "symmetry_mean": [mean_symmetry],
            "fractal_dimension_mean": [mean_fractal_dimension],
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


