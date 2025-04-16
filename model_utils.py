import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

MODEL_PATH = 'model.pkl'

def train_model(X, y):
    """
    Trains a logistic regression model on X and y.
    Returns the trained model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using accuracy and confusion matrix.
    Returns a dictionary of results.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return {
        'accuracy': acc,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }

def save_model(model, path=MODEL_PATH):
    """
    Saves the model to disk using pickle.
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path=MODEL_PATH):
    """
    Loads a model from disk.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)