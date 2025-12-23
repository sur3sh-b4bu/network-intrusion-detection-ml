@echo off
echo ========================================
echo Network Intrusion Detection System
echo Windows CMD Runner
echo ========================================

cd /d "%~dp0"

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Checking if data files exist...
if not exist "data\KDDTrain+.txt" (
    echo ERROR: KDDTrain+.txt not found in data\ directory
    echo Please download NSL-KDD dataset from:
    echo https://www.unb.ca/cic/datasets/nsl.html
    pause
    exit /b 1
)

if not exist "data\KDDTest+.txt" (
    echo ERROR: KDDTest+.txt not found in data\ directory
    echo Please download NSL-KDD dataset from:
    echo https://www.unb.ca/cic/datasets/nsl.html
    pause
    exit /b 1
)

echo Data files found. Starting pipeline...
echo.

echo ========================================
echo STEP 1: Data Preprocessing
echo ========================================
cd src
python data_preprocessing.py
if %errorlevel% neq 0 (
    echo ERROR: Data preprocessing failed
    cd ..
    pause
    exit /b 1
)
cd ..

echo.
echo ========================================
echo STEP 2: Model Training
echo ========================================
cd src
python model_training.py
if %errorlevel% neq 0 (
    echo ERROR: Model training failed
    cd ..
    pause
    exit /b 1
)
cd ..

echo.
echo ========================================
echo STEP 3: Model Evaluation
echo ========================================
cd src
python evaluation.py
if %errorlevel% neq 0 (
    echo ERROR: Model evaluation failed
    cd ..
    pause
    exit /b 1
)
cd ..

echo.
echo ========================================
echo SUCCESS: All steps completed!
echo ========================================
echo.
echo Output files created:
echo - Preprocessed data: data\X_train_preprocessed.csv
echo - Trained models: models\logistic_regression.pkl
echo - Evaluation results: results\*.png
echo.
echo Next steps:
echo 1. Check the evaluation metrics above
echo 2. Open notebooks\intrusion_detection.ipynb for analysis
echo.
pause
