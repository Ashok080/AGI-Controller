import streamlit as st
import pandas as pd
import os

st.title("AGI & AGS Dataset Explorer")

# Dataset selector
dataset_choice = st.selectbox(
    "Select a dataset:",
    ["AGI Dataset", "AGS Dataset"]
)

# Map dataset choice to file paths in your repo
dataset_paths = {
    "AGI Dataset": "aos/agi/agi_dataset.csv",  # change to your actual AGI dataset path
    "AGS Dataset": "aos/ags/ags_dataset.csv"   # change to your actual AGS dataset path
}

selected_path = dataset_paths[dataset_choice]

# Load dataset
if os.path.exists(selected_path):
    df = pd.read_csv(selected_path)
    st.write(f"### Preview of {dataset_choice}")
    st.dataframe(df.head())

    st.write(f"**Shape:** {df.shape}")
else:
    st.error(f"Dataset not found: {selected_path}")
