import streamlit as st
import pandas as pd
import numpy as np
from data_utils import clean_grade_columns, load_course_catalog
import os
from model_utils import load_model, predict_outcome
from data_utils import load_course_catalog
from model_utils import save_regression_model 
from joblib import load as joblib_load

st.set_page_config(page_title="PANCE Pass/Fail Predictor", layout="wide")
st.title("🩺 PA Student Performance Predictor")

# Create tabs
tabs = st.tabs(["User Guide", "Grade Template Download", "📂 Predict Pass/Fail", "🧑‍🏫 Manual Entry (Single Student)", "📈 PANCE Score Prediction"])


# Load model and scaler
@st.cache_resource
def get_model_and_scaler():
    model, scaler = load_model(path_model="model.pkl", path_scaler="scaler.pkl")
    return model, scaler

model, scaler = get_model_and_scaler()

@st.cache_resource
def get_regression_model():
    model = joblib_load("model_pance_score.pkl")
    scaler = joblib_load("scaler_pance_score.pkl")
    return model, scaler


# Load course catalog
course_codes_raw, course_credit_map = load_course_catalog("data/Course_Catalog.xlsx")
# course_codes = [code.strip() for code in course_codes_raw]
course_codes = sorted(course_credit_map.keys())


# Fix formatting if needed
if isinstance(course_codes_raw, (pd.Series, np.ndarray)):
    course_codes = [str(code).strip() for code in course_codes_raw.tolist()]
else:
    course_codes = [str(code).strip() for code in course_codes_raw]


with tabs[0]:
    st.title("📖 User Guide")
    st.markdown("This app allows you to upload student grade data and predict their risk of failing the PANCE exam.")
    st.markdown("1. **Download the Grade Input Template**: Use the template to prepare your student grade data for upload.")
    st.markdown("2. **Predict Pass/Fail**: Upload a CSV or Excel file containing student grades.")
    st.markdown("3. **Manual Entry (Single Student)**: Enter grades manually for 1-on-1 advising.")
    st.markdown("4. **Predict PANCE Scores**: Upload grades to predict PANCE scores based on historical data.")



with tabs[1]:
    st.markdown("Use this template to prepare your student grade data for upload.")
    st.markdown("### 📥 Download Grade Input Template")

    # Create an empty DataFrame with expected course columns (plus optional ID)
    template_df = pd.DataFrame(columns=["Unique Masked ID"] + course_codes)

    # Offer CSV download
    st.download_button(
        label="⬇️ Download Grade Template",
        data=template_df.to_csv(index=False).encode('utf-8'),
        file_name="grade_input_template.csv",
        mime="text/csv"
    )

# === Tab: File Upload (wrap existing logic) ===
with tabs[2]:
    st.subheader("📂 Upload Student Grades for Prediction")
    
    # File uploader
    uploaded_file = st.file_uploader("📂 Upload Student Grade File (.xlsx or .csv)", type=["xlsx", "csv"])

    def plot_dual_probability_histogram(model_probs, weighted_probs):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.hist(model_probs, bins=10, alpha=0.6, label="Model Probability", color="skyblue", edgecolor="black")
        plt.hist(weighted_probs, bins=10, alpha=0.6, label="Weighted GPA Probability", color="orange", edgecolor="black")
        plt.title("Probability Distributions")
        plt.xlabel("Probability")
        plt.ylabel("Number of Students")
        plt.legend()
        st.pyplot(plt)


    if uploaded_file:
        try:
            # Load file
            if uploaded_file.name.endswith(".xlsx"):
                input_data = pd.read_excel(uploaded_file)
            else:
                input_data = pd.read_csv(uploaded_file)

            # Normalize column names
            input_data.columns = [col[:7].strip() if col.startswith("PAS") else col.strip() for col in input_data.columns]

            # Extract student IDs
            if 'Unique Masked ID' in input_data.columns:
                ids = input_data['Unique Masked ID']
            else:
                ids = input_data.index.astype(str)

            # Extract just course data
            matched_cols = [col for col in course_codes if col in input_data.columns]
            if not matched_cols:
                raise ValueError("None of the expected PAS course columns were found.")

            input_data = input_data[matched_cols]

            # Run prediction using helper function
            results = predict_outcome(
                df=input_data,
                model=model,
                scaler=scaler,
                credit_weights=course_credit_map,
                features=course_codes,
                threshold=0.2  # Same threshold as train.py
            )

            results.insert(0, "Unique Masked ID", ids)

            # Display results
            st.subheader("📊 Prediction Results")
            st.dataframe(results, use_container_width=True)

            # NEW: Add probability histogram
            st.subheader("📈 Model Confidence Distribution")
            plot_dual_probability_histogram(
                results["Probability (Model)"],
                results["Probability (Weighted GPA)"]
            )




            # Download option
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results as CSV", data=csv, file_name="pance_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")

    else:
        st.info("👆 Please upload a file to begin.")


# === Tab: Manual Entry ===
with tabs[3]:
    st.subheader("🧑‍🏫 Manual Grade Entry for 1-on-1 Advising")

    # Collect grades using input boxes
    manual_input = {}
    for course in course_codes:
        manual_input[course] = st.number_input(
            label=f"{course} grade",
            min_value=0.0,
            max_value=100.0,
            step=0.5,
            format="%.1f"
        )

    # Add a Predict button
    if st.button("🔍 Predict Risk Category"):
        # Convert input to DataFrame
        input_df = pd.DataFrame([manual_input])

        # Predict
        result = predict_outcome(
            df=input_df,
            model=model,
            scaler=scaler,
            credit_weights=course_credit_map,
            features=course_codes,
            threshold=0.2
        )

        st.subheader("🧠 Model Prediction")
        st.dataframe(result, use_container_width=True)


# === Tab: Score Prediction ===
with tabs[4]:
    st.subheader("📈 Predict PANCE Scores from Grades")

    reg_model, reg_scaler = get_regression_model()

    uploaded_score_file = st.file_uploader("📂 Upload Student Grades for Score Prediction (.xlsx or .csv)", type=["xlsx", "csv"], key="score")

    if uploaded_score_file:
        try:
            if uploaded_score_file.name.endswith(".xlsx"):
                df_score = pd.read_excel(uploaded_score_file)
            else:
                df_score = pd.read_csv(uploaded_score_file)

            # Normalize column names
            df_score.columns = [col[:7].strip() if col.startswith("PAS") else col.strip() for col in df_score.columns]

            # Extract IDs
            ids = df_score['Unique Masked ID'] if 'Unique Masked ID' in df_score.columns else df_score.index.astype(str)

            # Force all expected columns to be present, in order
            df_score = df_score.reindex(columns=course_codes)
            df_score.columns = course_codes # force column names to match exactly

            # Convert everything to numeric (in case there are str/empty cells)
            df_score = df_score.apply(pd.to_numeric, errors='coerce')

            # Fill missing values with column means
            df_score = df_score.fillna(df_score.mean())

            # Debug: sanity check before scaling
            #st.write("Grades before scaling:", df_score.describe())

            # Confirm matching column names before transform
            if list(df_score.columns) != course_codes:
                st.write("Model expected:", course_codes)
                st.write("Input columns:", list(df_score.columns))
                st.error("Feature mismatch: column names don't match what model was trained on.")
                st.stop()

            # # Apply credit weights
            # for col, weight in course_credit_map.items():
            #     if col in df_score.columns:
            #         df_score[col] *= weight


            expected = course_codes
            actual = list(df_score.columns)

            #st.write("✅ Model was trained on:", expected)
            #st.write("📄 Uploaded file columns (after reindex):", actual)

            missing = [col for col in expected if col not in actual]
            extra = [col for col in actual if col not in expected]
            #st.write("❌ Missing columns:", missing)
            #st.write("🧯 Unexpected columns:", extra)
            #st.write("🟰 Column order match:", expected == actual)
            

            # Align columns in order, with valid names
            # df_score = df_score.reindex(columns=course_codes)
            # df_score.columns = course_codes  # ensure identical names
            # df_score = df_score.apply(pd.to_numeric, errors='coerce')
            # df_score = df_score.fillna(df_score.mean())

            # Debug: sanity check before scaling
            #            git filter-repo --path venv/Lib/site-packages/xgboost/lib/xgboost.dll --invert-paths            git filter-repo --path venv/Lib/site-packages/xgboost/lib/xgboost.dll --invert-pathsst.write("Grades before scaling:", df_score.describe())

            print("Using columns:")
            print(df_score.columns.tolist())

            # Scale and predict
            X_scaled = reg_scaler.transform(df_score)  # <- this will work now
            predictions = reg_model.predict(X_scaled)

            
            result_df = pd.DataFrame({
                "Unique Masked ID": ids,
                "Predicted PANCE Score": np.round(predictions, 1)
            })

            st.subheader("📊 Score Predictions")
            st.dataframe(result_df, use_container_width=True)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Score Predictions", data=csv, file_name="pance_score_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")
    else:
        st.info("👆 Upload a file to get started.")
