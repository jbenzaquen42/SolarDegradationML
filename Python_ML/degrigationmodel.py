#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

# Load the dataset
df = pd.read_csv("../Data/pvfleet.csv")

# Display the first few rows of the dataframe
print(df.head(5))

# Display general information about the dataframe
df.info()

# Cleaning Data

### Missing Values
# Identifying columns with missing values and the count of missing values in each column
missing_values = df.isnull().sum()
print("Missing values by column:\n", missing_values)

# Removing rows where any of the specified columns ('technology1', 'technology2', 'power_dc', 'pv_climate_zone', 'type_mounting', 'tracking') are missing data
columns_with_critical_info = ['technology1', 'technology2', 'power_dc', 'pv_climate_zone', 'type_mounting', 'tracking']
df = df.dropna(subset=columns_with_critical_info, how='any')

# Correcting Data Types

# Make sure 'df' is a direct copy to operate on
df = df.copy()

# Converting 'length_years_rounded' to numerical type (float) explicitly
df['length_years_rounded'] = pd.to_numeric(df['length_years_rounded'], errors='coerce')

# Assuming 'tracking' column contains boolean-like values stored as objects, this converts them to actual booleans
df['tracking'] = df['tracking'].astype(bool)

# Convert 'tracking' from boolean to integer (binary encoding)
df['tracking'] = df['tracking'].astype(int)

# Identifying numeric and categorical features
numeric_features = ['length_years_rounded']
categorical_features = ['technology1', 'technology2', 'pv_climate_zone', 'type_mounting']

# Defining preprocessing for numeric features: imputation + scaling
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Defining preprocessing for categorical features: imputation + one-hot encoding
# Here we set handle_unknown='ignore' to address the issue with unknown categories
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combining preprocessing steps
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)])

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Preparing the data
X = df.drop(['plr_median', 'plr_confidence_low', 'plr_confidence_high'], axis=1)
y = df['plr_median']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
pipeline.fit(X_train, y_train)

# Evaluating the model
y_pred = pipeline.predict(X_test)
print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
print(f'MSE: {mean_squared_error(y_test, y_pred)}')
print(f'R^2: {r2_score(y_test, y_pred)}')

# Saving the model
model_path = '../Model/deg_modelv1.joblib'
joblib.dump(pipeline, model_path)
print(f"Model saved to {model_path}")
