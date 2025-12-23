# Network Intrusion Detection System Using Machine Learning

## Project Overview
This project implements a Network Intrusion Detection System (NIDS) using supervised machine learning techniques. The system classifies network traffic as either NORMAL or ATTACK based on features extracted from network packets. This is an entry-level project suitable for beginners in cybersecurity and AI, demonstrating data preprocessing, model training, and evaluation.

## Problem Statement
Network intrusion detection is crucial for identifying malicious activities in computer networks. Traditional rule-based systems can be bypassed, so machine learning offers a data-driven approach to detect anomalies. This project uses the NSL-KDD dataset to train models that can classify network traffic in real-time scenarios.

## Dataset Description
- **Dataset**: NSL-KDD (Network Security Laboratory - Knowledge Discovery in Databases)
- **Files**:
  - `KDDTrain+.txt`: Training dataset
  - `KDDTest+.txt`: Testing dataset
- **Features**: 41 features including protocol type, service, flag, and various statistical measures
- **Target**: Binary classification (0: Normal, 1: Attack)
- **Size**: Training set has ~125,973 records, Test set has ~22,544 records

## Methodology
1. **Data Loading**: Load and inspect the NSL-KDD dataset
2. **Data Understanding**: Analyze class distribution and feature types
3. **Data Preprocessing**:
   - Encode categorical features using One-Hot Encoding
   - Handle missing values
   - Feature scaling with StandardScaler
   - Train-test split
4. **Model Building**: Train Logistic Regression and Random Forest classifiers
5. **Model Evaluation**: Assess performance using accuracy, precision, recall, and confusion matrix
6. **Results & Conclusion**: Compare models and discuss limitations

## Tech Stack
- **Language**: Python 3.x
- **Libraries**:
  - NumPy: Numerical computations
  - Pandas: Data manipulation
  - Matplotlib/Seaborn: Data visualization
  - Scikit-learn: Machine learning algorithms
  - Jupyter: Interactive notebook for demonstration

## How to Run the Project

### Prerequisites
- Python 3.x installed
- VS Code or any Python IDE

### Setup
1. Clone or download the project repository
2. Navigate to the project directory: `cd network-intrusion-ml`
3. Install dependencies: `pip install -r requirements.txt`
4. Download the NSL-KDD dataset:
   - Visit: https://www.unb.ca/cic/datasets/nsl.html
   - Download `KDDTrain+.txt` and `KDDTest+.txt` (approximately 20MB each)
   - Place these files in the `data/` directory
   - Replace the existing placeholder files

### Running the Code

#### Option 1: Complete Pipeline with Timing Analysis (Recommended)
Run the entire project with detailed timing measurements:
```bash
python run_with_timing.py
```
This will:
- Check prerequisites and data files
- Run all steps sequentially with timing
- Provide comprehensive performance analysis
- Show execution time for each step

#### Option 2: Individual Scripts
Run modules separately in `src/` directory:
```bash
python src/data_preprocessing.py  # Data loading and preprocessing
python src/model_training.py      # Model training
python src/evaluation.py          # Model evaluation and analysis
```

#### Option 3: Jupyter Notebook
Interactive analysis and visualization:
```bash
jupyter notebook notebooks/intrusion_detection.ipynb
```

### Expected Output
- Preprocessed data saved as CSV files
- Trained models saved as pickle files
- Evaluation metrics printed to console
- Visualizations displayed in notebook

## Results
- **Logistic Regression**: Achieved ~XX% accuracy on test set
- **Random Forest**: Achieved ~XX% accuracy on test set
- Detailed metrics available in evaluation output

## Future Enhancements
- Implement additional ML algorithms (SVM, KNN)
- Add feature selection techniques
- Integrate with real-time packet capture tools
- Deploy as a web service using Flask/Django
- Explore deep learning approaches for improved accuracy

## References
- NSL-KDD Dataset: https://www.unb.ca/cic/datasets/nsl.html
- Scikit-learn Documentation: https://scikit-learn.org/

## Author
[Your Name] - Entry-level ML Engineer and Cybersecurity Enthusiast

## License
This project is for educational purposes only.
