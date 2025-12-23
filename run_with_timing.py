#!/usr/bin/env python3
"""
Complete Network Intrusion Detection System Runner with Timing Analysis

This script runs the entire NIDS pipeline with detailed timing measurements:
1. Data preprocessing
2. Model training
3. Model evaluation

Usage:
    python run_with_timing.py

Requirements:
- NSL-KDD dataset files in data/ directory
- All dependencies installed (pip install -r requirements.txt)
"""

import subprocess
import sys
import time
import os

def run_script_with_timing(script_name, description):
    """
    Run a Python script and measure its execution time.

    Parameters:
    script_name (str): Name of the script to run (without .py extension)
    description (str): Description of what the script does

    Returns:
    tuple: (success, execution_time)
    """
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}.py")
    print('='*60)

    start_time = time.time()

    try:
        # Change to src directory and run the script
        result = subprocess.run([sys.executable, f'src/{script_name}.py'],
                              cwd=os.getcwd(),
                              capture_output=True,
                              text=True,
                              timeout=3600)  # 1 hour timeout

        execution_time = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ SUCCESS: {script_name}.py completed in {execution_time:.2f} seconds")
            print("Sample output:")
            # Print last 10 lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[-10:]:
                if line.strip():
                    print(f"  {line}")
            return True, execution_time
        else:
            print(f"❌ FAILED: {script_name}.py failed with return code {result.returncode}")
            print("Error output:")
            print(result.stderr)
            return False, execution_time

    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        print(f"⏰ TIMEOUT: {script_name}.py exceeded 1 hour limit")
        return False, execution_time
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"💥 ERROR: Failed to run {script_name}.py - {str(e)}")
        return False, execution_time

def check_prerequisites():
    """
    Check if all prerequisites are met before running the pipeline.
    """
    print("🔍 CHECKING PREREQUISITES...")

    # Check if data files exist
    train_file = 'data/KDDTrain+.txt'
    test_file = 'data/KDDTest+.txt'

    if not os.path.exists(train_file):
        print(f"❌ MISSING: {train_file}")
        print("   Please download NSL-KDD dataset and place KDDTrain+.txt in data/ directory")
        print("   Download from: https://www.unb.ca/cic/datasets/nsl.html")
        return False

    if not os.path.exists(test_file):
        print(f"❌ MISSING: {test_file}")
        print("   Please download NSL-KDD dataset and place KDDTest+.txt in data/ directory")
        print("   Download from: https://www.unb.ca/cic/datasets/nsl.html")
        return False

    # Check file sizes (rough check for real data vs placeholder)
    train_size = os.path.getsize(train_file)
    test_size = os.path.getsize(test_file)

    if train_size < 1000000:  # Less than 1MB, probably placeholder
        print("⚠️  WARNING: KDDTrain+.txt seems small. Please ensure you have the full dataset.")
        print("   Expected size: ~20MB, Current size: ~{:.1f}KB".format(train_size/1024))

    if test_size < 1000000:  # Less than 1MB, probably placeholder
        print("⚠️  WARNING: KDDTest+.txt seems small. Please ensure you have the full dataset.")
        print("   Expected size: ~20MB, Current size: ~{:.1f}KB".format(test_size/1024))

    print("✅ Data files found")
    print(f"   Training data: {train_size/1024/1024:.1f} MB")
    print(f"   Testing data: {test_size/1024/1024:.1f} MB")

    # Check if required directories exist
    required_dirs = ['models', 'results']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"📁 Created directory: {dir_name}")

    print("✅ All prerequisites met!")
    return True

def main():
    """
    Main function to run the complete NIDS pipeline with timing.
    """
    print("🚀 NETWORK INTRUSION DETECTION SYSTEM - COMPLETE PIPELINE")
    print("="*70)
    print("This will run the entire ML pipeline with detailed timing analysis:")
    print("1. Data Preprocessing (loading, cleaning, feature engineering)")
    print("2. Model Training (Logistic Regression + Random Forest)")
    print("3. Model Evaluation (metrics, confusion matrices, analysis)")
    print("="*70)

    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above and try again.")
        sys.exit(1)

    # Initialize timing variables
    total_start_time = time.time()
    step_times = {}

    # Step 1: Data Preprocessing
    success, exec_time = run_script_with_timing('data_preprocessing', 'Data Preprocessing')
    step_times['preprocessing'] = exec_time
    if not success:
        print("\n❌ Pipeline failed at data preprocessing step.")
        sys.exit(1)

    # Step 2: Model Training
    success, exec_time = run_script_with_timing('model_training', 'Model Training')
    step_times['training'] = exec_time
    if not success:
        print("\n❌ Pipeline failed at model training step.")
        sys.exit(1)

    # Step 3: Model Evaluation
    success, exec_time = run_script_with_timing('evaluation', 'Model Evaluation')
    step_times['evaluation'] = exec_time
    if not success:
        print("\n❌ Pipeline failed at evaluation step.")
        sys.exit(1)

    # Final summary
    total_time = time.time() - total_start_time

    print(f"\n{'='*70}")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print('='*70)
    print("EXECUTION SUMMARY:")
    print(".2f")
    print(".2f")
    print(".2f")
    print(".2f")
    print(".2f")
    print('='*70)

    print("\n📊 PERFORMANCE BREAKDOWN:")
    print(".1f")
    print(".1f")
    print(".1f")

    print("\n📁 OUTPUT FILES CREATED:")
    print("   Preprocessed data: data/X_train_preprocessed.csv, data/X_test_preprocessed.csv")
    print("   Trained models: models/logistic_regression.pkl, models/random_forest.pkl")
    print("   Evaluation plots: results/logistic_regression_confusion_matrix.png")
    print("                     results/random_forest_confusion_matrix.png")

    print("\n💡 NEXT STEPS:")
    print("   1. Review the evaluation metrics and confusion matrices")
    print("   2. Open notebooks/intrusion_detection.ipynb for interactive analysis")
    print("   3. Consider the limitations and future improvements mentioned in the output")

    print("\n🎯 PROJECT COMPLETE!")
    print("   This NIDS system is ready for portfolio presentation and interviews.")

if __name__ == "__main__":
    main()
