import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Define column names for NSL-KDD dataset (41 features + label + difficulty)
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

def load_data(train_path, test_path):
    """
    Load NSL-KDD dataset from text files.

    Parameters:
    train_path (str): Path to training data file
    test_path (str): Path to testing data file

    Returns:
    tuple: (train_df, test_df) pandas DataFrames
    """
    try:
        train_df = pd.read_csv(train_path, header=None, names=columns)
        test_df = pd.read_csv(test_path, header=None, names=columns)
        print(f"Training data shape: {train_df.shape}")
        print(f"Testing data shape: {test_df.shape}")
        print("\nFirst few rows of training data:")
        print(train_df.head())
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure dataset files are in the data/ directory.")
        return None, None

def preprocess_data(train_df, test_df):
    """
    Preprocess the NSL-KDD dataset for machine learning.

    Steps:
    1. Convert labels to binary (normal=0, attack=1)
    2. Handle missing values
    3. Encode categorical features
    4. Scale numerical features
    5. Split data

    Parameters:
    train_df (pd.DataFrame): Training data
    test_df (pd.DataFrame): Testing data

    Returns:
    tuple: (X_train, X_test, y_train, y_test)
    """
    if train_df is None or test_df is None:
        return None, None, None, None

    # Combine datasets for consistent preprocessing
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # Step 1: Convert labels to binary
    # Normal traffic = 0, All attacks = 1
    combined_df['label'] = combined_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

    print("\nClass distribution:")
    print(combined_df['label'].value_counts())
    print(f"Normal: {combined_df['label'].value_counts()[0]}")
    print(f"Attack: {combined_df['label'].value_counts()[1]}")

    # Step 2: Handle missing values (NSL-KDD has no missing values, but check anyway)
    print(f"\nMissing values: {combined_df.isnull().sum().sum()}")

    # Step 3: Identify categorical and numerical features
    categorical_features = ['protocol_type', 'service', 'flag']
    numerical_features = [col for col in columns[:-1] if col not in categorical_features]

    # Step 4: Encode categorical features using One-Hot Encoding
    # One-Hot Encoding converts categorical variables into binary vectors
    # This is necessary because ML algorithms work with numerical data
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_categorical = encoder.fit_transform(combined_df[categorical_features])
    encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features))

    # Combine encoded categorical with numerical features
    X = pd.concat([combined_df[numerical_features], encoded_df], axis=1)
    y = combined_df['label']

    # Step 5: Feature scaling using StandardScaler
    # StandardScaler normalizes features to have mean=0 and variance=1
    # This is important for algorithms like Logistic Regression that are sensitive to feature scales
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Step 6: Split back into train and test sets
    # Since we combined them, split based on original sizes
    train_size = len(train_df)
    X_train = X_scaled[:train_size]
    X_test = X_scaled[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    print(f"\nPreprocessed data shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print("="*60)
    print("NSL-KDD DATA PREPROCESSING - TIMING ANALYSIS")
    print("="*60)

    start_time = time.time()

    # Load data
    print("\n[STEP 1] Loading data...")
    load_start = time.time()
    train_df, test_df = load_data('../data/KDDTrain+.txt', '../data/KDDTest+.txt')
    load_time = time.time() - load_start
    print(".2f")

    # Preprocess data
    print("\n[STEP 2] Preprocessing data...")
    preprocess_start = time.time()
    X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df)
    preprocess_time = time.time() - preprocess_start
    print(".2f")

    # Save preprocessed data (optional)
    if X_train is not None:
        print("\n[STEP 3] Saving preprocessed data...")
        save_start = time.time()
        X_train.to_csv('../data/X_train_preprocessed.csv', index=False)
        X_test.to_csv('../data/X_test_preprocessed.csv', index=False)
        y_train.to_csv('../data/y_train.csv', index=False)
        y_test.to_csv('../data/y_test.csv', index=False)
        save_time = time.time() - save_start
        print(".2f")
        print("\nPreprocessed data saved to data/ directory.")

    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("TIMING SUMMARY:")
    print(".2f")
    print(".2f")
    print(".2f")
    print(".2f")
    print("="*60)
