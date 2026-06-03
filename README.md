Breast Cancer Prediction

Machine learning project for predicting breast cancer-related outcomes using clinical data.

This project was developed as part of a machine learning hackathon. It includes data preprocessing, baseline models, decision-tree-based feature extraction, XGBoost models, neural networks, and exploratory data analysis.

Project Overview:

The goal of this project is to build and evaluate machine learning models for breast cancer prediction tasks, including metastasis-related classification.

The project compares baseline approaches with more advanced models that combine tree-based methods and neural networks.

The original dataset is private clinical data and is therefore not included in this repository.

Dataset:

This project was developed using private clinical breast cancer data.

Due to privacy restrictions, the original train and test CSV files are not included in this repository.
To run the code, compatible train and test CSV files should be provided in the expected format.

Prediction CSV files and analysis output files are also not included, since they may be derived from private data.

Project Structure:
breast-cancer-prediction/
│
├── subtaskI/
│   ├── preprocess.py
│   ├── Decision_tree.py
│   ├── Baseline.py
│   ├── BoostedMetastasisNN.py
│   ├── Predicting_Metastases.py
│   └── requirements.txt
│
├── subtaskII/
│   ├── part2.py
│   ├── part2_baseline.py
│   └── requirements.txt
│
├── subtaskIII/
│   ├── Data_analysis.py
│   ├── hospital_by_age_part3.py
│   ├── part_3.py
│   └── requirements.txt
│
├── README.md
├── project.pdf
└── .gitignore
Subtasks
subtaskI — Metastasis Prediction

This part focuses on preprocessing the data and predicting metastasis using decision-tree-based features and a neural network model.

Main files:

preprocess.py — preprocesses the data for the project.
Decision_tree.py — trains a separate decision tree for each label and returns both the ensemble model and per-label positive-class probabilities.
Baseline.py — preprocesses the data, fits per-label decision trees with multiple depths, predicts on a test split, and saves predictions and true labels.
BoostedMetastasisNN.py — trains a dropout-regularized feedforward neural network on the decision-tree probability features using Adam optimization and binary cross-entropy loss.
Predicting_Metastases.py — main file for running the metastasis prediction pipeline.

The train and test CSV files required for this part are not included due to privacy restrictions.

subtaskII — XGBoost and Neural Network Model

This part combines an XGBoost baseline with a neural network model.

Main files:

part2_baseline.py — baseline model for this part, implemented using XGBoost.
part2.py — main model, a neural network that uses features produced by the XGBoost model.

The train and test CSV files required for this part are not included due to privacy restrictions.

subtaskIII — Data Analysis and Final Report

This part includes exploratory data analysis and the final hackathon report.

Main files:

Data_analysis.py — exploratory data analysis and initial research direction.
hospital_by_age_part3.py — analysis related to hospital and age-based patterns.
part_3.py — main implementation for part 3.
project.pdf — final report for the hackathon and part 3.

Any CSV output files generated during the analysis are not included due to privacy restrictions.

Technologies Used
Python
Pandas
NumPy
Scikit-learn
XGBoost
Neural Networks
Decision Trees
Data preprocessing
Model evaluation
Exploratory data analysis
Machine Learning Methods

The project uses several machine learning approaches:

Decision Tree models
XGBoost baseline
Neural Networks
Dropout regularization
Binary cross-entropy loss
Per-label classification
Feature extraction from model probabilities
Feature importance analysis
How to Run

Because the original dataset is private, the data files are not included in this repository.

To run the project:
Add compatible train and test CSV files in the expected locations.
Install the required Python packages from the relevant requirements.txt file.
Run the relevant main file for each subtask.

Example:

pip install -r requirements.txt
python Predicting_Metastases.py

The exact running command may vary depending on the specific subtask and file structure.

Privacy Notice:
This repository does not include the original clinical dataset due to privacy restrictions.
Private data, personal information, sensitive medical information, prediction CSV files, and derived analysis CSV files should not be uploaded to this repository.

Goal:
The main goal of this project is to explore machine learning techniques for breast cancer prediction and evaluate how different models perform on clinical features.
