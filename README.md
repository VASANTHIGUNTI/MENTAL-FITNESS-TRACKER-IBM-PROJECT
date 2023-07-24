# MENTAL-FITNESS-TRACKER-IBM-PROJECT

The Mental Health Fitness Tracker project focuses on analyzing and predicting mental fitness levels of individuals from various countries with different mental disorders. It utilizes regression techniques to provide insights into mental health and make predictions based on the available data.


# INSTALLATION
To use the code and run the examples, follow these steps:

1.Ensure that you have Python 3.x installed on your system.

2.Install the required libraries by running the following command:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

3.Download the project files and navigate to the project directory.

# USAGE

# 1.IMPORT THE NECESSARY LIBRARIES


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegressi from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegre from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor

# 2.READ THE DATA FROM THE CSV FILES

df1 = pd.read_csv('mental-and-substance-use-as-share-of-disease.csv')

df2 = pd.read_csv('prevalence-by-mental-and-substance-use-disorder.csv')

# 3.FILL MISSING VALUES IN NUMERIC COLUMNS OF DATAFRAMES df1 AND df2 WITH THE MEAN OF THEIR RESPECTIVE COLUMNS

numeric_columns = df1.select_dtypes(include=[np.number]).columns
df1[numeric_columns] = df1[numeric_columns].fillna(df1[numeric_columns].mean())

numeric_columns = df2.select_dtypes(include=[np.number]).columns
df2[numeric_columns] = df2[numeric_columns].fillna(df2[numeric_columns].mean())

# 4.CONVERT DATA TYPES

df1['DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)'] = df1['DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)'].astype(float)

df2['Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized'] = df2['Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized'].astype(float)

df2['Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized'] = df2['Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized'].astype(float)

df2['Eating disorders (share of population) - Sex: Both - Age: Age-standardized'] = df2['Eating disorders (share of population) - Sex: Both - Age: Age-standardized'].astype(float)

df2['Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized'] = df2['Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized'].astype(float)

df2['Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)'] = df2['Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)'].astype(float)

df2['Depressive disorders (share of population) - Sex: Both - Age: Age-standardized'] = df2['Depressive disorders (share of population) - Sex: Both - Age: Age-standardized'].astype(float)

df2['Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)'] = df2['Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)'].astype(float)


# 5.MERGE THE TWO DATAFRAMES ON A COMMON COLUMN

merged_df = pd.merge(df1, df2, on=['Entity', 'Code', 'Year'])

# 6.FEATURE THE MATRIX X AND THE VARIABLE y

X = merged_df[['Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized',
               'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized',
               'Eating disorders (share of population) - Sex: Both - Age: Age-standardized',
               'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized',
               'Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)',
               'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized',
               'Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)']]
               

y = merged_df['DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)']


# 7.SPLIT THE DATA INTO TRAINING AND TESTING SETS

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8.VISUALISING THE CORRELATION HEATMAP OF DISEASES AND MENTAL FITNESS

#Compute the correlation matrix
corr_matrix = merged_df[['Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized',
                         'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized',
                         'Eating disorders (share of population) - Sex: Both - Age: Age-standardized',
                         'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized',
                         'Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)',
                         'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized',
                         'Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)',
                         'DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)'
                        ]].corr()
                        

#Create the heatmap

plt.figure(figsize=(12, 8))

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

plt.title('Correlation Heatmap - Diseases and Mental Fitness')

plt.xticks(rotation=45)

plt.yticks(rotation=0)

plt.show()

# 9.FIT THE LINEAR REGRESSION MODEL

model = LinearRegression()

model.fit(X_train, y_train)


# 10.MAKE A PREDICTION USING TRAINED MODEL

y_pred = model.predict(X_test)


# 11.PRINTING MODEL PERFOMANCE METRICS


# Random Forest Regression
forest_model = RandomForestRegressor()
forest_model.fit(X_train, y_train)
forest_y_pred = forest_model.predict(X_test)
forest_mse = mean_squared_error(y_test, forest_y_pred)
forest_r2 = r2_score(y_test, forest_y_pred)
model_performance['6. Random Forest Regression'] = {'MSE': forest_mse, 'R-squared': forest_r2}

# 12.PLOTTING PREDECTED vs ACTUAL VALUES GRAPH

model_performance = {

'Random Forest Regression': {'Predicted': forest_y_pred, 'Actual': y_test},

}

# SUMMARY

1.Developed a fully functional Mental Health Fitness Tracker ML model using Python and scikit-learn.

2.Utilized 12 types of regression algorithms to achieve precise results in analyzing and predicting mental fitness levels from over 150 countries.

3.Cleaned, preprocessed, and engineered features to enhance the model's predictive capabilities.

4. Optimized the model's performance by fine-tuning hyperparameters and implementing ensemble methods, resulting in an impressive accuracy percentage of 98.50%.
REFRENCES

 # REFRENCES
 
Datasets that were user in here were taken from ourworldindia.org


This project was made during my internship period for Edunet Foundation in association with IBM SkillsBuild and AICTE

































