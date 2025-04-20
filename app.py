import streamlit as st
import pandas as pd
import numpy as np
from model_utils import load_model
from data_utils import clean_grade_columns, load_course_catalog
import os
from model_utils import load_model, predict_outcome
from data_utils import load_course_catalog

st.set_page_config(page_title="PANCE Pass/Fail Predictor", layout="wide")
st.title("🩺 PA Student Performance Predictor")

# Create tabs
tabs = st.tabs(["📂 Upload Grades", "🧑‍🏫 Manual Entry"])

# Load model and scaler
@st.cache_resource
def get_model_and_scaler():
    model, scaler = load_model(path_model="model.pkl", path_scaler="scaler.pkl")
    return model, scaler

model, scaler = get_model_and_scaler()

# Load course catalog
course_codes_raw, course_credit_map = load_course_catalog("data/Course_Catalog.xlsx")
course_codes = [code.strip() for code in course_codes_raw]

# === Tab 1: File Upload (wrap existing logic) ===
with tabs[0]:

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


# === Tab 2: Manual Entry ===
with tabs[1]:
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

# st.set_page_config(page_title="PANCE Pass/Fail Predictor", layout="wide")
# st.title("🩺 PA Student Performance Predictor")

# # Load model and scaler with correct unpacking
# @st.cache_resource
# def get_model_and_scaler():
#     model, scaler = load_model()
#     return model, scaler

# model, scaler = get_model_and_scaler()

# # Load expected course codes
# course_codes_raw, course_credit_map = load_course_catalog("data/Course_Catalog.xlsx")
# course_codes = [code.strip() for code in course_codes_raw]

# # # Ensure course_codes is a clean, flat list of strings
# # if isinstance(course_codes, pd.Series):
# #     course_codes = course_codes.tolist()
# # elif isinstance(course_codes, np.ndarray):
# #     course_codes = course_codes.flatten().tolist()

# # # Final fallback: remove any non-string or nested items
# # course_codes = [str(code).strip() for code in course_codes if isinstance(code, (str, int))]


# uploaded_file = st.file_uploader("📂 Upload Student Grade File (.xlsx or .csv)", type=["xlsx", "csv"])

# if uploaded_file:
#     try:
#         if uploaded_file.name.endswith(".xlsx"):
#             input_data = pd.read_excel(uploaded_file)
#         else:
#             input_data = pd.read_csv(uploaded_file)

#         # Normalize column headers by stripping after 6-character course codes
#         cleaned_columns = []
#         for col in input_data.columns:
#             if col.startswith("PAS"):
#                 cleaned_columns.append(col[:7].strip())  # e.g., PAS 610
#             else:
#                 cleaned_columns.append(col.strip())
#         input_data.columns = cleaned_columns

#         if 'Unique Masked ID' in input_data.columns:
#             ids = input_data['Unique Masked ID']
#         else:
#             ids = input_data.index.astype(str)
        
#         st.write("📋 Uploaded file columns:", input_data.columns.tolist())

#         st.write("📋 Cleaned uploaded column names:", input_data.columns.tolist())
#         st.write("🎯 Expected course codes:", course_codes)
#         st.write("✅ Type of course_codes:", type(course_codes))
#         matched_cols = [col for col in course_codes if col in input_data.columns]
#         st.write("✅ Matched columns:", matched_cols)
#         st.write(type(course_codes), course_codes)
#         st.write(type(input_data.columns), input_data.columns.tolist())
#         st.write(type(matched_cols), matched_cols)

#         input_data = input_data[matched_cols]

#         # Filter and align columns
#         input_data = input_data[[col for col in course_codes if col in input_data.columns]]
#         if input_data.empty:
#             raise ValueError("None of the expected PAS course columns were found in the uploaded file.")

#         input_data = input_data.reindex(columns=course_codes, fill_value=np.nan)
#         input_data = input_data.fillna(input_data.mean())

#         # # Scale the input
#         # input_scaled = scaler.transform(input_data)

#         # # Predict
#         # y_pred = model.predict(input_scaled)
#         # y_proba = model.predict_proba(input_scaled)[:, 1]

#         # Prepare results
#         results = pd.DataFrame({
#             "Unique Masked ID": ids,
#             "Predicted Result": ["Pass" if pred == 1 else "Fail" for pred in y_pred],
#             "Probability of Passing": y_proba
#         })


#         st.subheader("📊 Prediction Results")
#         st.dataframe(results, use_container_width=True)

#         csv = results.to_csv(index=False).encode('utf-8')
#         st.download_button("⬇️ Download Results as CSV", data=csv, file_name="pance_predictions.csv", mime="text/csv")

#     except Exception as e:
#         st.error(f"⚠️ Error processing file: {e}")

# else:
#     st.info("👆 Please upload a file to begin.")




# # === Paths ===
# MODEL_PATH = "model.pkl"
# COURSE_CATALOG_PATH = "data/Course_Catalog.xlsx"

# @st.cache_resource
# def load_app_components():
#     model = load_model(MODEL_PATH)
#     course_codes = load_course_catalog(COURSE_CATALOG_PATH)
#     return model, course_codes

# model, course_codes = load_app_components()

# st.title("📊 PANCE Outcome Prediction App")
# st.markdown("""
# Upload student grade data (PAS course grades only) to predict **Pass/Fail** outcomes
# on the PANCE certification exam.
# """)

# # === File Upload ===
# uploaded_file = st.file_uploader("📁 Upload student grade file (.xlsx or .csv)", type=["xlsx", "csv"])

# if uploaded_file:
#     try:
#         if uploaded_file.name.endswith(".csv"):
#             df_input = pd.read_csv(uploaded_file)
#         else:
#             df_input = pd.read_excel(uploaded_file)

#         st.success(f"✅ Successfully loaded file with {len(df_input)} students")

#         # === Clean PAS features ===
#         df_features = clean_grade_columns(df_input, course_codes)

#         # === Validate columns ===
#         missing_cols = [col for col in course_codes if col not in df_features.columns]
#         if missing_cols:
#             st.error(f"❌ Uploaded file is missing required PAS course columns:\n{missing_cols}")
#             st.stop()

#         # === Restrict and reorder columns to match model input ===
#         df_model_input = df_features.reindex(columns=course_codes)

#         # === Tolerate missing grades: fill with mean
#         df_model_input = df_model_input.fillna(df_model_input.mean())

#         # === Predict
#         y_pred = model.predict(df_model_input)
#         y_prob = model.predict_proba(df_model_input)[:, 1]

#         # === Append predictions to original input
#         df_input = df_input.reset_index(drop=True)
#         df_input['Prediction'] = np.where(y_pred == 1, 'Pass', 'Fail')
#         df_input['Probability (Pass)'] = np.round(y_prob, 3)

#         st.subheader("📈 Prediction Results")
#         st.dataframe(df_input[['Unique Masked ID', 'Prediction', 'Probability (Pass)']])

#         # === Download button
#         csv = df_input.to_csv(index=False).encode('utf-8')
#         st.download_button(
#             label="📥 Download Results",
#             data=csv,
#             file_name="pance_predictions.csv",
#             mime="text/csv"
#         )

#     except Exception as e:
#         st.error(f"⚠️ Error processing file: {e}")
