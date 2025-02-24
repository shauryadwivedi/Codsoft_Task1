# 🚢 Titanic Survival Prediction
## 📌 Overview
This project predicts the survival of passengers on the Titanic using machine learning techniques. The dataset contains information such as passenger class, age, gender, fare, and number of family members aboard. The goal is to build a classification model to determine whether a passenger would survive based on these features.

## 🔍 Features
Data Preprocessing: Handling missing values, feature encoding, feature scaling, and data cleaning.
Exploratory Data Analysis (EDA): Analyzing survival trends based on different factors like class, age, and gender using visualizations.
Model Training & Evaluation: Implementing machine learning models such as Logistic Regression, Decision Trees, Random Forest, and Support Vector Machine (SVM).
Performance Metrics: Accuracy, Precision, Recall, F1-score, and Confusion Matrix for model evaluation.

## 📂 Dataset
The dataset used is the Titanic - Machine Learning from Disaster dataset from Kaggle. It consists of passenger details, including survival status (0 = No, 1 = Yes).

## ⚙️ Tech Stack
Programming Language: Python
Development Environment: Jupyter Notebook (Anaconda)
Libraries Used:
Data Handling: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Machine Learning: Scikit-Learn

## 🚀 How to Run
Clone the Repository
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
Set Up the Environment (Anaconda Recommended)
conda create --name titanic-survival python=3.8  
conda activate titanic-survival  
pip install -r requirements.txt  
Run Jupyter Notebook
jupyter notebook  
Open the Notebook (titanic_survival_prediction.ipynb) and execute the cells step by step.

## 📊 Results & Insights
Gender and Passenger Class are strong indicators of survival.
Feature engineering and hyperparameter tuning improve model performance.
The best-performing model is selected based on accuracy and recall to minimize false negatives.

# 🛡️ Future Enhancements
Implementing deep learning models for better prediction accuracy.
Deploying the model using Flask/Django for real-time survival predictions.
Performing advanced feature engineering for improved insights.
