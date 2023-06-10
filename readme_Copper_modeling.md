Copper Industry ML Models
This repository contains Python code for building two machine learning models for the copper industry:


A regression model that predicts the selling price of copper based on various features such as quantity, thickness, width, etc.
A classification model that predicts whether a lead will result in a successful sale or not, based on features such as customer, country, product type, etc.
The data used for these models is included in the repository, along with Jupyter notebooks that walk through the data preprocessing, feature engineering, model training, and evaluation steps. Additionally, a Streamlit app is provided that allows users to input feature values and get a predicted selling price or lead status.

Installation:
To run the code in this repository, you will need to have Python 3 installed, along with several Python packages such as Pandas, Scikit-learn, and Streamlit. You can install these packages using pip or conda:

code:
pip install pandas scikit-learn streamlit

Usage:
To use the Streamlit app, navigate to the project directory in your terminal and run:


code:
streamlit run app.py

This will start the app and open it in your browser. From there, you can select the task (regression or classification), input feature values, and get a predicted selling price or lead status.

Contributing
If you have any suggestions or improvements for this project, feel free to open an issue or submit a pull request.

 continuous variable ‘Selling_Price’ and an ML Classification model that predicts Status: WON or LOST. A streamlit page is created where each column value can be inserted, and the Selling_Price predicted value or Status(Won/Lost) will be displayed.

Steps:
The following steps are involved in the project:

Data Understanding:
Identify the types of variables (continuous, categorical) and their distributions. Some rubbish values are present in ‘Material_Reference’ which starts with ‘00000’ value, which should be converted into null. Treat both reference columns as categorical variables. INDEX may not be useful.

Data Preprocessing:
Handle missing values with mean/median/mode. Treat Outliers using IQR or Isolation Forest from the sklearn library. Identify Skewness in the dataset and treat skewness with appropriate data transformations such as log transformation, boxcox transformation, or other techniques to handle high skewness in continuous variables. Encode categorical variables using suitable techniques, such as one-hot encoding, label encoding, or ordinal encoding based on their nature and relationship with the target variable.

EDA:
Try visualizing outliers and skewness (before and after treating skewness) using Seaborn’s boxplot, distplot, violinplot. Feature Engineering: Engineer new features if applicable, such as aggregating or transforming existing features to create more informative representations of the data. And drop highly correlated columns using SNS HEATMAP.

Model Building and Evaluation:
Split the dataset into training and testing/validation sets. Train and evaluate different classification models such as ExtraTreesClassifier, XGBClassifier, or Logistic Regression, using appropriate evaluation metrics such as accuracy, precision, recall, F1 score, and AUC curve. Optimize model hyperparameters using techniques such as cross-validation and grid search to find the best-performing model. Interpret the model results and assess its performance based on the defined problem statement. Same steps for Regression modelling. Note: the dataset contains more noise and linearity between independent variables so it'll perform well only with tree-based models.

Model GUI:
Using the streamlit module, create an interactive page with (1) task input (Regression or Classification) and (2) an input field where each column value except ‘Selling_Price’ for the regression model and except ‘Status’ for the classification model can be entered. (3) Perform the same feature engineering, scaling factors, log/any transformation steps used for training the ml model, predict this new data from streamlit, and display the output.

Conclusion:
The copper industry can benefit from machine learning models to address sales and lead prediction challenges. This project provides an end-to-end solution to create a regression model for Selling_Price prediction and a classification model for Lead prediction. With the streamlit page, it is easier to predict values using the models with new data inputs.





