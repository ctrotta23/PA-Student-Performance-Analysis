import streamlit as st
import pandas as pd
import numpy as np
from model_utils import load_model
from data_utils import clean_grade_columns, load_course_catalog
import os

# === Paths ===
MODEL_PATH = "model.pkl"
COURSE_CATALOG_PATH = "data/Course_Catalog.xlsx"

@st.cache_resource
def load_app_components():
    model = load_model(MODEL_PATH)
    course_codes = load_course_catalog(COURSE_CATALOG_PATH)
    return model, course_codes

model, course_codes = load_app_components()

st.title("📊 PANCE Outcome Prediction App")
st.markdown("""
Upload student grade data (PAS course grades only) to predict **Pass/Fail** outcomes
on the PANCE certification exam.
""")

# === File Upload ===
uploaded_file = st.file_uploader("📁 Upload student grade file (.xlsx or .csv)", type=["xlsx", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_excel(uploaded_file)

        st.success(f"✅ Successfully loaded file with {len(df_input)} students")

        # === Clean PAS features ===
        df_features = clean_grade_columns(df_input, course_codes)

        # === Validate columns ===
        missing_cols = [col for col in course_codes if col not in df_features.columns]
        if missing_cols:
            st.error(f"❌ Uploaded file is missing required PAS course columns:\n{missing_cols}")
            st.stop()

        # === Restrict and reorder columns to match model input ===
        df_model_input = df_features.reindex(columns=course_codes)

        # === Tolerate missing grades: fill with mean
        df_model_input = df_model_input.fillna(df_model_input.mean())

        # === Predict
        y_pred = model.predict(df_model_input)
        y_prob = model.predict_proba(df_model_input)[:, 1]

        # === Append predictions to original input
        df_input = df_input.reset_index(drop=True)
        df_input['Prediction'] = np.where(y_pred == 1, 'Pass', 'Fail')
        df_input['Probability (Pass)'] = np.round(y_prob, 3)

        st.subheader("📈 Prediction Results")
        st.dataframe(df_input[['Unique Masked ID', 'Prediction', 'Probability (Pass)']])

        # === Download button
        csv = df_input.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="pance_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
