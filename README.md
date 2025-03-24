# Loan-Default-Prediction
This project focuses on predicting whether a loan applicant is likely to default on their loan using machine learning. The goal is to help financial institutions make informed decisions about loan approvals.

## Table of Contents

  - Project Overview

  - Dataset

  - Usage

  - Results

  - Model Interpretability

  - Future Improvements

  - Contributing

  - License


## Project Overview

The project involves building and evaluating machine learning models to predict loan default based on features such as:

- Applicant income

- Co-applicant income

- Loan amount

- Loan term

- Credit history

- Property area

- Education level

- Employment status

Three models were implemented and compared:

  * Support Vector Machine (SVM)

  * K-Nearest Neighbors (KNN)

  * Decision Tree

The Support Vector Machine model achieved the best performance with an accuracy of 80%.


## Dataset

The dataset consists of two files:

  - Train Dataset: train_u6lujuX_CVtuZ9i.csv (614 rows)

  - Test Dataset: test_Y3wMUE5_7gLdaTN.csv (367 rows)

### Features

  - Numerical Features: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History

  - Categorical Features: Gender, Married, Dependents, Education, Self_Employed, Property_Area

### Target Variable

  - Loan_Status: Whether the loan was approved (Y) or not (N).


## Usage

  1. Data Preprocessing:

        Handle missing values.

        Encode categorical variables.

        Scale numerical features.

   2. Model Training:

        Train SVM, KNN, and Decision Tree models on the training dataset.

   3. Model Evaluation:

        Evaluate models on the validation set using accuracy, precision, recall, and F1-score.

   4. Model Interpretation:

        Use SHAP and LIME to interpret the model's predictions.

   5. Make Predictions:

        Predict loan approval for the test dataset.


## Results

Model Performance
Model	Accuracy	Precision	Recall	F1-Score
SVM	0.78	0.79	0.78	0.78
KNN	0.75	0.76	0.75	0.75
Decision Tree	0.80	0.81	0.80	0.80

## Model Interpretability

LIME (Local Interpretable Model-agnostic Explanations)

LIME was used to explain individual predictions. For example:

    Prediction: 80% probability of loan approval.

    Key Features:

        Loan_Amount_Term: Positive contribution (supports approval).

        LoanAmount, Gender, Education, Self_Employed: Negative contributions (support rejection).

## Future Improvements

  Feature Engineering: Creating new features like debt-to-income ratio.

  Hyperparameter Tuning: Optimizing model hyperparameters using Grid Search or Random Search.

  Advanced Models: Experiment with ensemble methods like Random Forest or Gradient Boosting.

License

This project is licensed under the MIT License. See the LICENSE file for details.





