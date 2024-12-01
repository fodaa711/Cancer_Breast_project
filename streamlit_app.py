import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit App Title
st.title("Breast Cancer Data Analysis")

# File Upload Section
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Attempt to load the dataset
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

        # Convert non-numeric columns to numeric or exclude them
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
