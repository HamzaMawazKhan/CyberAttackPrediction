# Cyber Attack Prediction and Analysis using AI

A comprehensive machine learning project that predicts cyber attack categories using various AI models and compares their performance on a network intrusion dataset.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Models Implemented](#models-implemented)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)

## 🔍 Overview

This project implements multiple machine learning algorithms to predict cyber attack categories based on network traffic features. The analysis includes comprehensive data preprocessing, feature engineering, model training, and performance comparison across different classification algorithms.

## 📊 Dataset

The project uses a cyber attack dataset containing **125,973 records** with **43 features** including:
- Network connection features (duration, protocol_type, service, flag)
- Traffic statistics (src_bytes, dst_bytes, count, srv_count)
- Content features (hot, num_failed_logins, logged_in)
- Time-based features (same_srv_rate, diff_srv_rate)
- Host-based features (dst_host_count, dst_host_srv_count)

**Target Variable**: `attack_category` with 5 classes:
- Normal traffic
- DOS (Denial of Service)
- Probe attacks
- R2L (Remote to Local)
- U2R (User to Root)

## ✨ Features

### Data Preprocessing
- **Missing Value Analysis**: Comprehensive data quality assessment
- **Feature Selection**: Removal of low-variance features using statistical analysis
- **Correlation Analysis**: Identification and handling of highly correlated features
- **Encoding**: Label encoding for categorical variables
- **Normalization**: Feature scaling using sklearn preprocessing

### Feature Engineering
- Histogram analysis for feature distribution
- Correlation heatmap visualization
- Removal of redundant features (num_root, num_compromised, etc.)
- One-hot encoding for target variable

### Model Training & Evaluation
- Train-test split (80-20)
- Cross-validation setup
- Model performance comparison
- Comprehensive evaluation metrics

## 🤖 Models Implemented

### 1. Artificial Neural Network (ANN)
- **Architecture**: Sequential model with 4 layers
- **Layers**: 
  - Input: 20 neurons (ReLU + LeakyReLU)
  - Hidden: 30 neurons (ReLU + LeakyReLU)
  - Hidden: 40 neurons (ReLU)
  - Output: 5 neurons (Softmax)
- **Optimizer**: Adam (lr=0.01)
- **Loss**: Categorical crossentropy
- **Callbacks**: Early stopping, Model checkpoint
- **Accuracy**: 99.36%

### 2. Random Forest Classifier
- **Estimators**: 100 trees
- **Performance**: Best performing model
- **Accuracy**: 99.88%

### 3. Multinomial Logistic Regression
- **Solver**: LBFGS
- **Multi-class**: Multinomial approach
- **Accuracy**: 92.00%

### 4. Support Vector Machine (SVM)
- **Kernel**: RBF (default)
- **Accuracy**: 96.50%

### 5. K-Nearest Neighbors (KNN)
- **Default parameters**: k=5
- **Accuracy**: 99.33%

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cyber-attack-prediction.git
cd cyber-attack-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Ensure you have the dataset file `CyberAttack_Dataset` in the project directory.

## 💻 Usage

1. **Run the complete analysis**:
```bash
python cyber_attack_analysis.py
```

2. **Individual model training**:
```python
# Example for Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(train_x, train_y)
predictions = clf.predict(test_x)
```

3. **Visualization**:
```python
# Generate model comparison plot
plt.plot(['ANN', 'Random Forest', 'Logistic Regression', 'SVM', 'KNN'], 
         [accuracy_scores])
plt.title('Model Accuracy Comparison')
plt.show()
```

## 📈 Results

### Model Performance Ranking:
1. **Random Forest**: 99.88% accuracy
2. **K-Nearest Neighbors**: 99.33% accuracy  
3. **Artificial Neural Network**: 99.36% accuracy
4. **Support Vector Machine**: 96.50% accuracy
5. **Multinomial Logistic Regression**: 92.00% accuracy

### Key Insights:
- Random Forest achieved the highest accuracy due to its ensemble approach
- Neural Network showed excellent performance with proper architecture
- Class imbalance affects performance on minority classes (U2R, R2L)
- All models struggle with classes 3 and 4 due to limited samples

### Classification Challenges:
- **Class Imbalance**: Classes 3 and 4 have very few samples
- **Feature Correlation**: High correlation between some features required removal
- **Preprocessing Impact**: Normalization significantly improved model performance

## 📁 Project Structure

```
cyber-attack-prediction/
│
├── cyber_attack_analysis.py    # Main analysis script
├── CyberAttack_Dataset         # Dataset file
├── best_model3.hdf5           # Saved neural network model
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
│
├── visualizations/           # Generated plots and charts
│   ├── correlation_heatmap.png
│   ├── feature_histograms.png
│   └── model_comparison.png
│
└── models/                   # Saved model files
    ├── random_forest_model.pkl
    ├── svm_model.pkl
    └── knn_model.pkl
```

## 📦 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
keras>=2.8.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

