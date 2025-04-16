import os
import pandas as pd


def load_course_catalog(path='data/course_catalog.xlsx'):
    catalog = pd.read_excel(path)
    # Filter only graded (not pass/fail) courses
    graded = catalog[catalog['Pass/Fail'] == False]
    valid_codes = graded['Course Code'].str.strip().unique().tolist()
    print(f"Found {len(valid_codes)} valid PAS course codes in the catalog.")
    print(f"Valid PAS course codes: {valid_codes}")
    return valid_codes


def clean_grade_columns(df, valid_codes):
    # Clean up column names
    df.columns = df.columns.str.strip()

    # Rename each course column to its first 7 chars
    renamed_cols = {}
    for col in df.columns:
        if col.upper().startswith('PAS') and len(col) >= 7:
            short_code = col[:7].strip()
            renamed_cols[col] = short_code

    df = df.rename(columns=renamed_cols)

    print("🔍 All cleaned column names:")
    print(df.columns.tolist())

    print("✅ Course codes expected:", valid_codes)


    # Drop duplicate PAS codes after renaming
    df = df.loc[:, ~df.columns.duplicated()]

    # Filter only PAS codes that are in the course catalog
    #valid_cols = [col for col in df.columns if col in valid_codes]
    valid_cols = [col for col in df.columns if col in valid_codes and col.startswith("PAS")]

    print(f"Matched PAS course columns: {valid_cols}")

    if not valid_cols:
        raise ValueError("No valid PAS course columns found based on course catalog.")

    cleaned = df[valid_cols].copy()

    for col in cleaned.columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')

    # print(f"✅ Final PAS course columns used: {course_cols}")
    # print(f"⚠️ Missing from cleaned df: {[c for c in valid_codes if c not in df.columns]}")

    return cleaned


def load_all_cohort_data(folder_path='data/cohort_didactic_PANCE_records'):
    print(f"Looking in folder: {folder_path}")
    print(f"Files found: {os.listdir(folder_path)}")
    all_dfs = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.xlsx') or filename.endswith('.csv'):
            full_path = os.path.join(folder_path, filename)
            df = pd.read_excel(full_path) if filename.endswith('.xlsx') else pd.read_csv(full_path)

            if 'Result' not in df.columns:
                print(f"⚠️ Skipping {filename} — no 'Result' column found.")
                continue

            print(f"✅ Loaded {filename} — {len(df)} rows")
            df['Cohort Source'] = filename
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No valid cohort training files found.")

    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total merged training records: {len(full_df)}")
    return full_df


# def load_all_cohort_data(folder_path='data/cohort_didactic_grades'):
#     """
#     Loads all cohort grade files from the specified folder.
#     Returns one concatenated DataFrame for model training.
#     Assumes each file contains: Unique ID, PAS grades, Result (Pass/Fail)
#     """
#     all_dfs = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.xlsx') or filename.endswith('.csv'):
#             full_path = os.path.join(folder_path, filename)
#             df = pd.read_excel(full_path) if filename.endswith('.xlsx') else pd.read_csv(full_path)
#             if 'Result' not in df.columns:
#                 continue  # skip prediction-only files
#             df['Cohort Source'] = filename  # track source cohort
#             all_dfs.append(df)

#     if not all_dfs:
#         raise ValueError("No valid cohort training files found.")

#     full_df = pd.concat(all_dfs, ignore_index=True)
#     return full_df

def load_grades(file):
    """
    Loads a single grade file (used for predictions).
    Returns tuple: (features X, student_ids)
    """
    df = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)
    student_ids = df['Unique Masked ID'] if 'Unique Masked ID' in df.columns else None
    X = clean_grade_columns(df)
    return X, student_ids