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

# model_utils.py

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # Import SMOTE
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

def train_model(X, y, features, credit_weights):
    print("\n🧠 Applying SMOTE to balance the dataset...")
    
    # Apply SMOTE for class balance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"✅ Resampled dataset shape: {X_resampled.shape}")

    # Apply credit hour weights to resampled data
    print("\n🔢 Applying credit hour weights...")
    X_weighted = X_resampled.copy()
    for feature, weight in credit_weights.items():
        if feature in X_weighted.columns:
            X_weighted[feature] *= weight

    # Scale the weighted features
    print("\n🧠 Scaling the features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_weighted)

    # Train the XGBoost model
    print("\n🧠 Training XGBoost model...")
    model = XGBClassifier(
        scale_pos_weight=len(y_resampled[y_resampled == 0]) / len(y_resampled[y_resampled == 1]),  # Handle imbalance
        n_estimators=200,  # Number of trees
        max_depth=6,  # Tree depth
        learning_rate=0.1,  # Shrinkage step size
        random_state=42, 
        use_label_encoder=False  # Avoid warnings
    )
    model.fit(X_scaled, y_resampled)
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

def evaluate_model(model, scaler, X, y, features):
    print("\n📊 Evaluating XGBoost model performance...")

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]  # Probability for positive class

    # Performance metrics
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)

    print("\nModel Performance:")
    print(report)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
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

    print("\nFeature Importances:")
    print(importance_df)

    return {
        'accuracy': acc,
        'confusion_matrix': cm,
        'feature_importance': importance_df
    }

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


