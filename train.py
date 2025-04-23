import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from data_utils import load_all_cohort_data, clean_grade_columns, load_course_catalog
from model_utils import train_model, evaluate_model, save_model
import streamlit as st
from model_utils import train_linear_regression, save_regression_model
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == "__main__":
    print("🚀 Loading training data...")
    df = load_all_cohort_data("data/cohort_didactic_PANCE_records")

    if 'Result' not in df.columns:
        raise ValueError("'Result' column is required in training data.")

    print("\n📘 Loading course catalog...")
    course_codes, course_credit_map = load_course_catalog("data/Course_Catalog.xlsx")

    print("\n🧼 Cleaning and preparing grade columns...")
    X_full = clean_grade_columns(df, course_codes)
    y_raw = df['Result'].astype(str).str.strip().str.title()
    y_mapped = y_raw.map({'Pass': 1, 'Fail': 0})

    df_combined = pd.concat([
        df[['Unique Masked ID', 'Cohort Source']].reset_index(drop=True),
        X_full.reset_index(drop=True),
        y_mapped.rename('Result').reset_index(drop=True)
    ], axis=1).dropna(subset=['Result'])

    # Keep only course columns
    X = df_combined[course_codes]
    y = df_combined.loc[X.index, 'Result']
    X = X.dropna(thresh=10)  # keep students with at least 10 grades
    y = y.loc[X.index]

    # df_cleaned = df_combined.dropna(subset=['Result'])
    # X = df_cleaned[course_codes]
    # X = X.dropna(thresh=15)
    # y = df_cleaned.loc[X.index, 'Result']

    print(f"\n✅ Final training set size: {len(X)} students")
    print("📊 Included students by cohort:")
    print(df_combined.loc[X.index, 'Cohort Source'].value_counts())

    # print(f"\n✅ Final training set size: {len(X)} students")
    # print("📊 Included students by cohort:")
    # print(df_cleaned.loc[X.index, 'Cohort Source'].value_counts())

    print("\n🧪 Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fill missing values with column means (needed before SMOTE)
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    print(f"Any NaNs in X_train? {X_train.isnull().any().any()}")
    print(f"Any NaNs in y_train? {y_train.isnull().any()}")

    st.write("DEBUG - course_codes type:", type(course_codes))
    st.write("DEBUG - course_codes item type:", type(course_codes[0]) if len(course_codes) > 0 else "Empty")
    st.write("DEBUG - course_credit_map type:", type(course_credit_map))

   # Train model (XGBoost + SMOTE + scaling + credit weighting)
    model, scaler = train_model(X_train, y_train, features=course_codes, credit_weights=course_credit_map)

    # Apply credit weights to test set
    X_test_weighted = X_test.copy()
    for col, weight in course_credit_map.items():
        if col in X_test_weighted.columns:
            X_test_weighted[col] *= weight

   # Evaluate model
    metrics = evaluate_model(model, scaler, X_test_weighted, y_test, features=course_codes, threshold=0.2)

    print(f"\n🎯 Final Accuracy: {metrics['accuracy']:.4f}")
    print("📉 Confusion Matrix:")
    print(metrics['confusion_matrix'])

    # Save model + scaler
    print("\n💾 Saving trained model...")
    save_model(model, scaler, path_model='model.pkl', path_scaler='scaler.pkl')
    print("✅ Training complete.")

# LINEAR REGRESSION: PANCE SCORE PREDICTION
if 'Score' in df.columns:
    print("\n📈 Starting linear regression training for PANCE score prediction...")


    df_reg = df.dropna(subset=['Score']).copy()
    X_reg = clean_grade_columns(df_reg, course_codes)
    X_reg = X_reg.reindex(columns=course_codes)
    X_reg = X_reg.fillna(X_reg.mean())
    y_reg = df_reg.loc[X_reg.index, 'Score']

    reg_model, reg_scaler, reg_metrics = train_linear_regression(
        X_reg, y_reg, course_codes
    )


    # df_reg = df.dropna(subset=['Score']).copy()
    # # X_reg = clean_grade_columns(df_reg, course_codes)
    # # X_reg = X_reg.fillna(X_reg.mean())
    # # y_reg = df_reg.loc[X_reg.index, 'Score']

    # X_reg = clean_grade_columns(df_reg, course_codes) # Extract grade columns

    # # Apply credit weighting to the grades
    # for course in course_codes:
    #     if course in X_reg.columns and course in course_credit_map:
    #         X_reg[course] *= course_credit_map[course]

    # # Ensure order matches training
    # X_reg = X_reg.reindex(columns=course_codes)
    # X_reg = X_reg.fillna(X_reg.mean())

    # y_reg = df_reg.loc[X_reg.index, 'Score']

    # # X_reg = X_reg[course_codes]  # force column order
    # # X_reg = X_reg.fillna(X_reg.mean())
    
    # # y_reg = df_reg.loc[X_reg.index, 'Score']


    # print(f"🧮 Regression dataset size: {len(X_reg)} students with PANCE scores")

    # # Train regression model
    # #reg_model, reg_scaler, reg_metrics = train_linear_regression(X_reg, y_reg, course_codes, course_credit_map)
    # # Train linear regression model
    # reg_model, reg_scaler, reg_metrics = train_linear_regression(
    #     X_reg, y_reg, course_codes, course_credit_map
    # )
    print("Fitting scaler on columns:")
    print(X_reg.columns.tolist())

    print(f"📊 Regression R²: {reg_metrics['r2']:.4f}")
    print(f"📉 Regression RMSE: {reg_metrics['rmse']:.2f}")

    # Save the regression model + scaler
    save_regression_model(reg_model, reg_scaler, 'model_pance_score.pkl', 'scaler_pance_score.pkl')
    print("✅ Regression model saved.")
else:
    print("⚠️ No PANCE_Score column found. Skipping regression.")
