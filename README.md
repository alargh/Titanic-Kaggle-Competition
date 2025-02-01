# Titanic: Machine Learning from Disaster

This repository contains a solution for the Titanic: Machine Learning from Disaster problem, a well-known dataset used for learning machine learning techniques. The objective is to predict which passengers survived the Titanic disaster based on various features such as age, sex, class, etc.

## Problem Description

The Titanic dataset consists of information about the passengers aboard the RMS Titanic, including whether they survived or not. This dataset is used to predict survival based on the following features:

- **PassengerId**: Unique ID for each passenger
- **Pclass**: The class of the passenger (1st, 2nd, 3rd)
- **Name**: Name of the passenger
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **SibSp**: Number of siblings or spouses aboard
- **Parch**: Number of parents or children aboard
- **Ticket**: Ticket number
- **Fare**: The fare the passenger paid for the ticket
- **Cabin**: Cabin where the passenger stayed
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Approach

The goal of this project is to create a machine learning model that predicts whether a passenger survived based on the features provided. To achieve this, multiple models were trained and evaluated to select the best-performing one. The approach includes the following steps:

### Data Exploration & Preprocessing
- **Loading and Cleaning Data**: The dataset is loaded, and preprocessing steps such as handling missing values, encoding categorical variables, and feature scaling are performed.
- **Visualizing Feature Distributions**: Various visualizations help to explore the distribution of features and identify trends related to passenger survival.
- **Feature Engineering** Added new features by combining old ones in optimal way looking at certain features distribution.

### Model Training
- **Train-Test Split**: The dataset is split into training and validation sets.
- **Model Training**: A variety of machine learning models were trained to predict survival, including:
  - **XGBoost**: A powerful gradient boosting model
  - **Logistic Regression**
  - **Random Forest Classifier**
  - **K-Nearest Neighbors (KNN)**
  - **Support Vector Classifier (SVC)**
- **Hyperparameter Tuning**: Hyperparameter tuning was performed for each model using `RandomizedSearchCV` to find the best set of parameters for each model.

### Model Evaluation
- **Cross-Validation**: Models were evaluated using cross-validation to assess their generalization ability and performance.
- **Selection of Best Model**: After training, the models' performances were compared based on cross-validation accuracy, and the best-performing model was chosen for predictions.

### Final Predictions
- **Making Predictions**: After training, the best model was used to make predictions on the test dataset.
- **Submission**: The results were saved into a CSV file (`submission.csv`) which can be submitted for evaluation.

## Results

The model achieved an accuracy score of **0.8227** on the validation set.

- **Best Model**: The best-performing model based on cross-validation accuracy was selected and used for final predictions.
- **Hyperparameters**: Hyperparameter optimization was performed for models like XGBoost, Random Forest, and Logistic Regression, leading to improved performance.

## Notebook

- **titanic.ipynb**

## Dependencies

- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
