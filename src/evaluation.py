import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

def load_models():
    """
    Load trained models from disk.

    Returns:
    tuple: (lr_model, rf_model)
    """
    try:
        with open('../models/logistic_regression.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        with open('../models/random_forest.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        print("Models loaded successfully.")
        return lr_model, rf_model
    except FileNotFoundError:
        print("Trained models not found. Please run model_training.py first.")
        return None, None

def load_test_data():
    """
    Load test data for evaluation.

    Returns:
    tuple: (X_test, y_test)
    """
    try:
        X_test = pd.read_csv('../data/X_test_preprocessed.csv')
        y_test = pd.read_csv('../data/y_test.csv').values.ravel()
        return X_test, y_test
    except FileNotFoundError:
        print("Test data not found. Please run data_preprocessing.py first.")
        return None, None

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model using various metrics.

    Parameters:
    model: Trained model object
    X_test (pd.DataFrame): Test features
    y_test (pd.Series): Test labels
    model_name (str): Name of the model for display
    """
    print(f"\n--- {model_name} Evaluation ---")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(".4f")
    print(".4f")
    print(".4f")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'../results/{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.show()

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

    return accuracy, precision, recall

def explain_metrics():
    """
    Explain the importance of evaluation metrics in the context of intrusion detection.
    """
    print("\n" + "="*60)
    print("METRICS EXPLANATION FOR INTRUSION DETECTION")
    print("="*60)

    print("\n1. ACCURACY:")
    print("   - Measures overall correctness of predictions")
    print("   - In security: High accuracy is good, but not sufficient alone")

    print("\n2. PRECISION:")
    print("   - Measures how many predicted attacks are actually attacks")
    print("   - In security: High precision means fewer false alarms")
    print("   - Formula: TP / (TP + FP)")

    print("\n3. RECALL:")
    print("   - Measures how many actual attacks were detected")
    print("   - In security: High recall means fewer missed attacks")
    print("   - Formula: TP / (TP + FN)")

    print("\n4. CONFUSION MATRIX:")
    print("   - TP: True Positive (Attack correctly detected)")
    print("   - TN: True Negative (Normal traffic correctly identified)")
    print("   - FP: False Positive (Normal traffic flagged as attack)")
    print("   - FN: False Negative (Attack missed as normal)")

    print("\nSECURITY CONTEXT:")
    print("- False Negatives (FN) are more dangerous than False Positives (FP)")
    print("- Missing an attack is worse than investigating a false alarm")
    print("- Balance between Precision and Recall is crucial")

def compare_models(lr_metrics, rf_metrics):
    """
    Compare performance of both models.

    Parameters:
    lr_metrics (tuple): (accuracy, precision, recall) for Logistic Regression
    rf_metrics (tuple): (accuracy, precision, recall) for Random Forest
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)

    metrics_names = ['Accuracy', 'Precision', 'Recall']
    lr_values = lr_metrics
    rf_values = rf_metrics

    for i, metric in enumerate(metrics_names):
        print(".4f")

        if lr_values[i] > rf_values[i]:
            print(f"   Logistic Regression performs better in {metric.lower()}")
        elif rf_values[i] > lr_values[i]:
            print(f"   Random Forest performs better in {metric.lower()}")
        else:
            print(f"   Both models have equal {metric.lower()}")

def discuss_limitations():
    """
    Discuss limitations and future improvements.
    """
    print("\n" + "="*60)
    print("LIMITATIONS AND FUTURE IMPROVEMENTS")
    print("="*60)

    print("\nCURRENT LIMITATIONS:")
    print("- Dataset is from 1999, may not represent modern attacks")
    print("- No real-time processing capability")
    print("- Binary classification may miss attack types")
    print("- Potential overfitting on training data")

    print("\nOVERFITTING CONSIDERATIONS:")
    print("- Random Forest can overfit if not tuned properly")
    print("- Cross-validation should be used for better generalization")
    print("- Regularization in Logistic Regression helps prevent overfitting")

    print("\nFUTURE ENHANCEMENTS:")
    print("- Implement cross-validation for robust evaluation")
    print("- Add feature selection to reduce dimensionality")
    print("- Explore other algorithms (SVM, Neural Networks)")
    print("- Integrate with real-time packet capture")
    print("- Add anomaly detection for unknown attacks")

if __name__ == "__main__":
    # Load models and test data
    lr_model, rf_model = load_models()
    X_test, y_test = load_test_data()

    if lr_model is not None and X_test is not None:
        # Evaluate Logistic Regression
        lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

        # Evaluate Random Forest
        rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")

        # Explain metrics
        explain_metrics()

        # Compare models
        compare_models(lr_metrics, rf_metrics)

        # Discuss limitations
        discuss_limitations()

        print("\nEvaluation completed. Results saved in results/ directory.")
