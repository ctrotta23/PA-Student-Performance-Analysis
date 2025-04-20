# model_utils.py

import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump, load
from imblearn.over_sampling import SMOTE  # Import SMOTE
from xgboost import XGBClassifier
from scipy.special import expit




def train_model(X, y, features, credit_weights):
    print("\n🧠 Applying SMOTE to balance the dataset...")
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"✅ Resampled dataset shape: {X_resampled.shape}")
    print(pd.Series(y_resampled).value_counts())

    # Apply credit hour weights
    print("\n🔢 Applying credit hour weights...")
    X_weighted = X_resampled.copy()
    for feature, weight in credit_weights.items():
        if feature in X_weighted.columns:
            X_weighted[feature] *= weight

    # Scale features
    print("\n🧠 Scaling the features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_weighted)

    # Train XGBoost model with a fixed scale_pos_weight
    print("\n🧠 Training XGBoost model...")
    num_neg = sum(y_resampled == 0)
    num_pos = sum(y_resampled == 1)
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1  # Avoid div by 0

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,  # Use imbalance ratio
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_scaled, y_resampled)

    # Show probability distributions (for debugging)
    print("🔍 Sample predicted probabilities (training set):")
    print(model.predict_proba(X_scaled)[:10])

    return model, scaler





# def train_model(X, y, features):
#     print("\n🧠 Applying SMOTE to balance the dataset...")
    
#     # Apply SMOTE
#     smote = SMOTE(random_state=42)
#     X_resampled, y_resampled = smote.fit_resample(X, y)
#     print(f"✅ Resampled dataset shape: {X_resampled.shape}")

#     print("\n🧠 Training Random Forest model with class weights...")
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_resampled)  # Use resampled data for scaling

#     model = RandomForestClassifier(
#         class_weight='balanced',
#         n_estimators=100,
#         min_samples_split=5,
#         random_state=42
#     )
#     model.fit(X_scaled, y_resampled)  # Train model with resampled data
#     return model, scaler

# def train_model(X, y, features):
#     print("\n🧠 Training Random Forest model with class weights...")

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     model = RandomForestClassifier(
#         class_weight='balanced',
#         n_estimators=100,
#         min_samples_split=5,
#         random_state=42
#     )

#     model.fit(X_scaled, y)
#     return model, scaler

def evaluate_model(model, scaler, X, y, features, threshold=0.2):
    print("\n📊 Evaluating XGBoost model performance...")

    # Scale features
    X_scaled = scaler.transform(X)

    # Predict probabilities for positive class (Pass = 1)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    # Apply custom threshold to get final predictions
    y_pred = (y_proba >= threshold).astype(int)
    print(f"📏 Using custom threshold: {threshold}")

    # Show prediction stats
    print("\n🔍 Predicted class distribution:")
    print(pd.Series(y_pred).value_counts())

    print("\n🔍 Probability summary stats:")
    print(pd.Series(y_proba).describe())

    # Performance metrics
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)

    print("\n📊 Model Performance:")
    print(report)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fail', 'Pass'],
                yticklabels=['Fail', 'Pass'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

    # Feature importances
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\n🌟 Feature Importances:")
    print(importance_df)

    return {
        'accuracy': acc,
        'confusion_matrix': cm,
        'feature_importance': importance_df
    }

def predict_outcome(df, model, scaler, credit_weights, features, threshold=0.2):
    """
    Given raw grade data, applies credit weighting, scaling, and predicts pass/fail with a custom threshold.
    
    Parameters:
    - df: DataFrame of raw student grades
    - model: Trained model
    - scaler: Trained scaler
    - credit_weights: Dict mapping course to credit weight
    - features: List of expected PAS course columns (ordered)
    - threshold: Probability threshold for classification (default: 0.2)
    
    Returns:
    - DataFrame with 'Predicted Result' and 'Probability of Passing'
    """

    df_copy = df.copy()

    # Apply credit hour weights
    for course, weight in credit_weights.items():
        if course in df_copy.columns:
            df_copy[course] *= weight

    # Align with expected features
    df_copy = df_copy.reindex(columns=features, fill_value=np.nan)
    df_copy = df_copy.fillna(df_copy.mean())

    # Scale features
    X_scaled = scaler.transform(df_copy)

    # Predict probabilities and apply threshold
    y_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Return results DataFrame
    results = pd.DataFrame({
        "Predicted Result": ["Pass" if pred == 1 else "Fail" for pred in y_pred],
        "Probability of Passing": y_proba
    })

    return results

def predict_outcome(df, model, scaler, credit_weights, features, threshold=0.2):

    df_copy = df.copy()

    # Apply credit hour weights
    for course, weight in credit_weights.items():
        if course in df_copy.columns:
            df_copy[course] *= weight

    # Align and clean
    df_copy = df_copy.reindex(columns=features, fill_value=np.nan)
    df_copy = df_copy.fillna(df_copy.mean())

    # Scale features
    X_scaled = scaler.transform(df_copy)

    # Predict pass probabilities
    y_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Feature importances from model
    importance_array = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importance_array})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Calculate Weighted Grade Average
    top_features = importance_df.head(10)['Feature'].tolist()
    weights = importance_df.set_index('Feature')['Importance']
    weights = weights.loc[top_features]
    normalized_weights = weights / weights.sum()

    weighted_grades = pd.DataFrame()
    for feature in normalized_weights.index:
        weighted_grades[feature] = df_copy[feature] * normalized_weights[feature]
    
    weighted_avg = weighted_grades.sum(axis=1)

    # Adjust these based on your grade scale
    midpoint = 275  # center of your typical weighted GPA distribution
    scale = 15

    prob_weighted = expit((weighted_avg - midpoint) / scale)

    # Define thresholds using quantiles
    threshold_at_risk = weighted_avg.quantile(0.25)
    threshold_borderline = weighted_avg.quantile(0.50)

    def assign_weighted_risk(score):
        if score < threshold_at_risk:
            return "At Risk"
        elif score < threshold_borderline:
            return "Borderline"
        else:
            return "Safe"

    weighted_risk = weighted_avg.apply(assign_weighted_risk)

    # Return result DataFrame
    results = pd.DataFrame({
    "Predicted Result": ["Pass" if pred == 1 else "Fail" for pred in y_pred],
    "Probability (Model)": y_proba,
    "Probability (Weighted GPA)": prob_weighted,
    "Weighted Grade Average": weighted_avg.round(2),
    "Risk Category": weighted_risk
})

    return results



# def evaluate_model(model, scaler, X, y, features):
#     print("\n📊 Evaluating XGBoost model performance...")

#     X_scaled = scaler.transform(X)
#     y_pred = model.predict(X_scaled)
#     y_proba = model.predict_proba(X_scaled)[:, 1]  # Probability for positive class

#     # Performance metrics
#     acc = accuracy_score(y, y_pred)
#     cm = confusion_matrix(y, y_pred)
#     report = classification_report(y, y_pred)

#     print("🔍 Predicted class distribution:")
#     print(pd.Series(y_pred).value_counts())

#     print("🔍 Probability summary stats:")
#     print(pd.Series(y_proba).describe())

#     print("\nModel Performance:")
#     print(report)

#     # Plot confusion matrix
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
#     plt.title('Confusion Matrix')
#     plt.ylabel('Actual')
#     plt.xlabel('Predicted')
#     plt.tight_layout()
#     plt.show()

#     # Feature importances
#     importances = model.feature_importances_
#     importance_df = pd.DataFrame({
#         'Feature': features,
#         'Importance': importances
#     }).sort_values(by='Importance', ascending=False)

#     print("\nFeature Importances:")
#     print(importance_df)

#     return {
#         'accuracy': acc,
#         'confusion_matrix': cm,
#         'feature_importance': importance_df
#     }
#---
# def evaluate_model(model, scaler, X, y, features):
#     print("\n📊 Evaluating model performance...")

#     X_scaled = scaler.transform(X)
#     y_pred = model.predict(X_scaled)

#     # Performance metrics
#     acc = accuracy_score(y, y_pred)
#     cm = confusion_matrix(y, y_pred)
#     report = classification_report(y, y_pred)

#     print("\nModel Performance:")
#     print(report)

#     # Plot confusion matrix
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
#     plt.title('Confusion Matrix')
#     plt.ylabel('Actual')
#     plt.xlabel('Predicted')
#     plt.tight_layout()
#     plt.show()

#     # Feature importances
#     importances = model.feature_importances_
#     importance_df = pd.DataFrame({
#         'Feature': features,
#         'Importance': importances
#     }).sort_values(by='Importance', ascending=False)

#     print("\nFeature Importances:")
#     print(importance_df)

#     return {
#         'accuracy': acc,
#         'confusion_matrix': cm,
#         'feature_importance': importance_df
#     }

def save_model(model, scaler, path_model='model.xgb', path_scaler='scaler.pkl'):
    joblib.dump(model, path_model)  # Save XGBoost model using joblib
    joblib.dump(scaler, path_scaler)

def load_model(path_model='model.xgb', path_scaler='scaler.pkl'):
    model = joblib.load(path_model)
    scaler = joblib.load(path_scaler)
    return model, scaler


# def save_model(model, scaler, path_model='model.pkl', path_scaler='scaler.pkl'):
#     joblib.dump(model, path_model)
#     joblib.dump(scaler, path_scaler)


# def load_model(path_model='model.pkl', path_scaler='scaler.pkl'):
#     model = joblib.load(path_model)
#     scaler = joblib.load(path_scaler)
#     return model, scaler


# def train_model(X, y, features, course_credit_map=None):
#     # Apply course weighting
#     if course_credit_map:
#         for col in features:
#             if col in X.columns:
#                 X[col] = X[col] * course_credit_map.get(col, 1)

#     # Fill missing values
#     X = X.fillna(X.mean())

#     # Scale features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # Grid search with RandomForestClassifier
#     param_grid = {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [None, 10, 20],
#         'min_samples_split': [2, 5]
#     }

#     grid_search = GridSearchCV(
#         RandomForestClassifier(class_weight='balanced', random_state=42),
#         param_grid,
#         cv=StratifiedKFold(n_splits=5),
#         scoring='balanced_accuracy',
#         n_jobs=-1
#     )

#     grid_search.fit(X_scaled, y)
#     best_model = grid_search.best_estimator_

#     return best_model, scaler

# def evaluate_model(model, scaler, X, y, features):
#     X = X.fillna(X.mean())
#     X_scaled = scaler.transform(X)
#     y_pred = model.predict(X_scaled)

#     print("\nModel Performance:")
#     print(classification_report(y, y_pred))

#     cm = confusion_matrix(y, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Fail', 'Pass'],
#                 yticklabels=['Fail', 'Pass'])
#     plt.title('Confusion Matrix')
#     plt.ylabel('Actual')
#     plt.xlabel('Predicted')
#     plt.tight_layout()
#     plt.savefig("confusion_matrix.png")

#     # Feature importance
#     importances = model.feature_importances_
#     feature_len = len(features)
#     importances = importances[:feature_len]  # ensure alignment
#     importance_df = pd.DataFrame({
#         'Feature': features,
#         'Importance': importances
#     }).sort_values('Importance', ascending=False)

#     print("\nFeature Importances:")
#     print(importance_df)

#     plt.figure(figsize=(12, 8))
#     sns.barplot(x='Importance', y='Feature', data=importance_df)
#     plt.title('Random Forest Feature Importances')
#     plt.tight_layout()
#     plt.savefig("feature_importances.png")

#     return {
#         'accuracy': accuracy_score(y, y_pred),
#         'confusion_matrix': cm,
#         'feature_importances': importance_df
#     }

# def save_model(model, scaler, model_path='model.pkl', scaler_path='scaler.pkl'):
#     dump(model, model_path)
#     dump(scaler, scaler_path)


# def load_model(model_path='model.pkl', scaler_path='scaler.pkl'):
#     model = joblib.load(model_path)
#     scaler = joblib.load(scaler_path)
#     return model, scaler


