# Titanic: Machine Learning from Disaster

This repository contains a solution for the Titanic: Machine Learning from Disaster problem, which is a popular dataset for learning machine learning techniques. The goal of this problem is to predict which passengers survived the Titanic disaster based on various features like age, sex, class, etc.

## Problem Description

The Titanic dataset consists of information about the passengers aboard the RMS Titanic, including whether they survived or not. This dataset is used to predict survival based on these features:

- `PassengerId`: Unique ID for each passenger
- `Pclass`: The class of the passenger (1st, 2nd, 3rd)
- `Name`: Name of the passenger
- `Sex`: Gender of the passenger
- `Age`: Age of the passenger
- `SibSp`: Number of siblings or spouses aboard
- `Parch`: Number of parents or children aboard
- `Ticket`: Ticket number
- `Fare`: The fare the passenger paid for the ticket
- `Cabin`: Cabin where the passenger stayed
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

The goal is to create a machine learning model that predicts whether a passenger survived based on these features.


### Notebooks

- **TitanicNotebook.ipynb**

### Data Exploration & Preprocessing

- Loading and cleaning the training data.
- Handling missing values and encoding categorical variables.
- Visualizing the distribution of features.

### Model Training

- Spliting the data into training and validation sets.
- Training multiple models (XGBoosting) to predict survival.
- Evaluating models performance

### Making Predictions

After training the model, predictions can be made on the test dataset, and the results can be saved as a submission file.

## Results

Model achieved accuracy score 0.8316
