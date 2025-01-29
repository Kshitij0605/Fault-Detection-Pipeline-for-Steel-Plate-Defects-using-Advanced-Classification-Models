
Fault Detection Pipeline for Steel Plate Defects

Overview

This project builds a machine learning pipeline to classify steel plate defects into seven categories, improving quality control in manufacturing.

Dataset & Defects

The dataset includes labeled defects:

Pastry, Z-Scratch, K-Scratch, Stains, Dirtiness, Bumps, Other Faults.


Workflow

1. Data Preprocessing – Handling missing values, normalization, feature encoding.


2. Model Training – Logistic Regression, KNN, Naïve Bayes, Decision Trees, Random Forest, SVM.


3. Evaluation – Accuracy, Precision, Recall, F1 Score.


4. Optimization & Feature Selection – PCA, Hyperparameter tuning (Future Scope).



Installation

pip install numpy pandas scikit-learn matplotlib seaborn

Running the Project

python preprocess.py  # Data cleaning  
python train.py       # Model training

Future Improvements

Hyperparameter tuning.

Deep learning models (CNNs).

Real-time defect detection system.
