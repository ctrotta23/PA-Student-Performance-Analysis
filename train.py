import pandas as pd
from sklearn.model_selection import train_test_split
from data_utils import load_all_cohort_data, clean_grade_columns, load_course_catalog
from model_utils import train_model, evaluate_model, save_model

if __name__ == "__main__":
    print("Loading training data from cohort files...")
    df = load_all_cohort_data("data/cohort_didactic_PANCE_records")

    if 'Result' not in df.columns:
        raise ValueError("'Result' column is required in training data.")

    print("Loading valid PAS course codes from course catalog...")
    course_codes = load_course_catalog("data/Course_Catalog.xlsx")

    # === Clean features and result labels ===
    X_full = clean_grade_columns(df, course_codes)
    y_raw = df['Result'].astype(str).str.strip().str.title()
    y_mapped = y_raw.map({'Pass': 1, 'Fail': 0})

    # === Combine everything for filtering ===
    df_combined = pd.concat([
        df[['Unique Masked ID', 'Cohort Source']].reset_index(drop=True),
        X_full.reset_index(drop=True),
        y_mapped.rename('Result').reset_index(drop=True)
    ], axis=1)

    # === Filter rows with valid result, allow up to 3 missing grades ===
    df_cleaned = df_combined.dropna(subset=['Result'])
    X = df_cleaned[course_codes]
    X = X.dropna(thresh=15)  # Keep students with at least 15/18 grades
    y = df_cleaned.loc[X.index, 'Result']

    print(f"✅ Final training set size: {len(X)} students")
    print("📊 Included students by cohort:")
    print(df_cleaned.loc[X.index, 'Cohort Source'].value_counts())

    # === Split and impute ===
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Impute missing values with column means (based on training set)
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())  # Always use training mean for test set

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Any NaNs in X_train? {X_train.isnull().any().any()}")
    print(f"Any NaNs in y_train? {y_train.isnull().any()}")

    # === Train and evaluate model ===
    print("Training logistic regression model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"🎯 Accuracy: {metrics['accuracy']:.4f}")
    print("📉 Confusion Matrix:")
    print(metrics['confusion_matrix'])

    print("💾 Saving trained model to 'model.pkl'...")
    save_model(model)
    print("✅ Training complete.")
