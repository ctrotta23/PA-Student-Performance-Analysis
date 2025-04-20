import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from data_utils import load_all_cohort_data, clean_grade_columns, load_course_catalog
from model_utils import train_model, evaluate_model, save_model
import streamlit as st

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

    # # Apply credit hour weights
    # credit_weights = pd.Series(course_credit_map)
    # X_train_weighted = X_train[course_codes] * credit_weights
    # X_test_weighted = X_test[course_codes] * credit_weights

    # print("\n📊 Class distribution in y_train:")
    # print(y_train.value_counts(normalize=True))


    # print("\n🧠 Training Random Forest model...")
    # #model, scaler = train_model(X_train, y_train, features=course_codes, credit_weights=course_credit_map)
    # model, scaler = train_model(X_train_weighted, y_train, features=course_codes, credit_weights=course_credit_map)
    


    # print("\n📊 Evaluating model...")
    # #metrics = evaluate_model(model, scaler, X_test, y_test, features=course_codes)
    # #metrics = evaluate_model(model, scaler, X_test_weighted, y_test, features=course_codes)
    # metrics = evaluate_model(model, scaler, X_test_weighted, y_test, features=course_codes)
    # print(f"🎯 Accuracy: {metrics['accuracy']:.4f}")
    # print("📉 Confusion Matrix:")
    # print(metrics['confusion_matrix'])

    # print("\n💾 Saving trained model to 'model.pkl'...")
    # save_model(model, scaler)
    # print("✅ Training complete.")
