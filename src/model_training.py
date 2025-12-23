import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
import time

def load_preprocessed_data():
    """
    Load preprocessed training data.

    Returns:
    tuple: (X_train, y_train)
    """
    try:
        X_train = pd.read_csv('../data/X_train_preprocessed.csv')
        y_train = pd.read_csv('../data/y_train.csv').values.ravel()
        print(f"Loaded training data: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
        return X_train, y_train
    except FileNotFoundError:
        print("Preprocessed data not found. Please run data_preprocessing.py first.")
        return None, None

def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model.

    Parameters:
    X_train (pd.DataFrame): Training features
    y_train (pd.Series): Training labels

    Returns:
    LogisticRegression: Trained model
    """
    print("Training Logistic Regression...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    print("Logistic Regression training completed.")
    return model

def train_random_forest(X_train, y_train):
    """
    Train Random Forest Classifier.

    Parameters:
    X_train (pd.DataFrame): Training features
    y_train (pd.Series): Training labels

    Returns:
    RandomForestClassifier: Trained model
    """
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Random Forest training completed.")
    return model

def save_model(model, filename):
    """
    Save trained model to disk.

    Parameters:
    model: Trained model object
    filename (str): Path to save the model
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def explain_models():
    """
    Explain why Logistic Regression and Random Forest are suitable for intrusion detection.
    """
    print("\n" + "="*60)
    print("WHY THESE MODELS FOR INTRUSION DETECTION?")
    print("="*60)

    print("\nLOGISTIC REGRESSION:")
    print("- Interpretable: Easy to understand feature importance")
    print("- Fast training and prediction: Suitable for real-time detection")
    print("- Provides probability scores: Useful for threshold tuning")
    print("- Works well with linearly separable data")

    print("\nRANDOM FOREST:")
    print("- Handles complex, non-linear relationships in network data")
    print("- Robust to overfitting with proper tuning")
    print("- Feature importance analysis helps identify key network features")
    print("- Ensemble method reduces variance and improves generalization")

    print("\nWHY NOT OTHER MODELS?")
    print("- No deep learning: Project requirement (simple, interview-friendly)")
    print("- SVM/Neural Networks: More complex, harder to explain")
    print("- KNN: Slow for large datasets, not suitable for real-time")

if __name__ == "__main__":
    print("="*60)
    print("MODEL TRAINING - TIMING ANALYSIS")
    print("="*60)

    start_time = time.time()

    # Load preprocessed data
    print("\n[STEP 1] Loading preprocessed data...")
    load_start = time.time()
    X_train, y_train = load_preprocessed_data()
    load_time = time.time() - load_start
    print(".2f")

    if X_train is not None:
        # Train models
        print("\n[STEP 2] Training Logistic Regression...")
        lr_start = time.time()
        lr_model = train_logistic_regression(X_train, y_train)
        lr_time = time.time() - lr_start
        print(".2f")

        print("\n[STEP 3] Training Random Forest...")
        rf_start = time.time()
        rf_model = train_random_forest(X_train, y_train)
        rf_time = time.time() - rf_start
        print(".2f")

        # Save models
        print("\n[STEP 4] Saving models...")
        save_start = time.time()
        save_model(lr_model, '../models/logistic_regression.pkl')
        save_model(rf_model, '../models/random_forest.pkl')
        save_time = time.time() - save_start
        print(".2f")

        # Explain model choices
        explain_models()

        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("TIMING SUMMARY:")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print("="*60)
        print("\nModel training completed successfully!")
