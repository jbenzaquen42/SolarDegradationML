#!/usr/bin/env python
# coding: utf-8

# # Solar Panel Degradation Analysis
# 
# ## Project Overview
# This project aims to analyze solar panel performance and degradation over time using data science techniques. The goal is to identify key factors that influence solar panel degradation and to build predictive models to forecast future performance. This analysis is crucial for optimizing solar panel maintenance and maximizing efficiency and longevity.
# 
# ## Objectives
# - Perform data cleaning and preprocessing to prepare the dataset for analysis.
# - Conduct exploratory data analysis (EDA) to uncover trends and insights.
# - Build and evaluate predictive models to forecast solar panel degradation rates.
# 
# ## Dataset Description
# The dataset contains various parameters related to solar panel performance, including types of analysis conducted, median system degradation rates, confidence intervals, and more, across different climate zones and te 
# 
# ## Dataset Source and Purpose
# 
# ### Source
# The dataset for this project is sourced from the **Photovoltaic Fleet Degradation Insights Data** available through DuraMAT's Data Hub, provided by the National Renewable Energy Laboratory (NREL). The dataset can be accessed directly [here](https://datahub.duramat.org/dataset/photovoltaic-fleet-degradation-insights-data).
# 
# ### Purpose
# The purpose of the dataset, as outlined by the [NREL's PV Fleet Performance Data Initiative](https://www.nrel.gov/pv/fleet-performance-data-initiative.html), is to enhance the understanding of solar panel performance and degradation across various conditions over time. This initiative aims to aggregate and analyze data from a wide array of solar panel installations to identify key factors influencing solar panel health, efficiency, and longevity. Insights gained from this data are crucial for improving solar panel maintenance strategies, optimizing performance, and extending the operational life of solar energy systems.
# chnologies.
# 

# ## Variables
# 
# 1. **plr_type**: Type of RdTools analysis conducted based on available weather data. 
#    - `sensor`: on-site irradiance and temperature sensors used. 
#    - `nsrdb`: satellite weather data used. 
#    - `clearsky`: no on-site weather data available. Modeled irradiance and average temperature was used under clear-sky conditions only.
# 2. **plr_median**: Median system degradation rate in %/yr.
# 3. **plr_confidence_low**: 95% confidence interval for PLR, lower value.
# 4. **plr_confidence_high**: 95% confidence interval for PLR, upper value.
# 5. **length_years_rounded**: Length of performance data set in years (may not be the same as the age of the system if we only have access to some of the system data).
# 6. **power_dc**: Approximate DC nameplate power of the system (grouped by 2MW).
# 7. **pv_climate_zone**: Karin PV climate temperature zone, scale of 1-7.
# 8. **technology1**: PV cell main technology (e.g., c-Si, CdTe, CIGS).
# 9. **technology2**: PV cell technology sub-category (e.g., PERC, Al-BSF).
# 10. **type_mounting**: Ground, roof, garage canopy, etc.
# 11. **tracking**: Boolean, is the system tacking the sun or fixed.
# n or fixed

# ## Loading the Dataset
# 
# The dataset, `pvfleet.csv`, includes detailed information on solar panel performance across different conditions and setups. Let's load this dataset and take a preliminary look at its structure and some of its entries.
#

# %%
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

# %% 


# Load the dataset
df = pd.read_csv("../Data/pvfleet.csv")

# Display the first few rows of the dataframe
print(df.head(5))


# # Display general information about the dataframe
df.info()

# %%  Cleaning Data

### Missing Values
# Identifying columns with missing values and the count of missing values in each column
missing_values = df.isnull().sum()
print("Missing values by column:\n", missing_values)

# Removing rows where any of the specified columns ('technology1', 'technology2', 'power_dc', 'pv_climate_zone', 'type_mounting', 'tracking') are missing data
columns_with_critical_info = ['technology1', 'technology2', 'power_dc', 'pv_climate_zone', 'type_mounting', 'tracking']

df = df.dropna(subset=columns_with_critical_info, how='any')

# Now, df should have rows only where all of these columns have data
## Recheck
# missing_values = df.isnull().sum()
# print("Missing values by column:\n", missing_values)


# ### Correcting Data Types

# Correcting data types explicitly to avoid DeprecationWarning
# Make sure 'df' is a direct copy to operate on
df = df.copy()

# Converting 'length_years_rounded' to numerical type (float) explicitly
df['length_years_rounded'] = pd.to_numeric(df['length_years_rounded'], errors='coerce')

# Assuming 'tracking' column contains boolean-like values stored as objects, this converts them to actual booleans
# This example assumes the 'tracking' column's data can be mapped directly to booleans without an explicit map
df['tracking'] = df['tracking'].astype(bool)

# Convert 'tracking' from boolean to integer (binary encoding)
df['tracking'] = df['tracking'].fillna(False).astype(int)

### Outliers
# Identifying outliers in 'plr_median' using IQR
Q1 = df['plr_median'].quantile(0.25)
Q3 = df['plr_median'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtering out the outliers
df = df[(df['plr_median'] >= lower_bound) & (df['plr_median'] <= upper_bound)]
df.head()
df.shape


# %%
# # Feature Engineering 

# %%

# One-Hot Encoding for Categorical Variables
categorical_cols = ['technology1', 'technology2', 'type_mounting', 'pv_climate_zone']
one_hot_encoder = OneHotEncoder(sparse=False)
encoded_features = one_hot_encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names_out(categorical_cols))
df_transformed = pd.concat([df.drop(categorical_cols + ['tracking'], axis=1), encoded_df, df['tracking']], axis=1)


# %%


# Creating a Confidence Range Feature
df_transformed['plr_confidence_range'] = df_transformed['plr_confidence_high'] - df_transformed['plr_confidence_low']

# Example Interaction Feature
df_transformed['tracking_x_pv_climate_zone_T3'] = df_transformed['tracking'] * df_transformed['pv_climate_zone_T3']


# %%


# Mapping 'power_dc' categories to numerical values
power_dc_mapping = {
    '< 0.5 MW': 1,
    '0.5-1 MW': 2,
    '1-5 MW': 3,
    '5-10 MW': 4,
    '> 10 MW': 5
}
df_transformed['power_dc_numeric'] = df['power_dc'].map(power_dc_mapping).fillna(0)  # Defaulting unknown categories to 0


# %%


df_transformed.head()


# # Split Dataset

# %%


# Drop rows with NaN values in the dataset before splitting
df_transformed = df_transformed.dropna()

#  'plr_median' is the target variable
X = df_transformed.drop(['plr_median', '_id', 'plr_type', 'plr_confidence_low', 'plr_confidence_high', 'power_dc'], axis=1)
y = df_transformed['plr_median']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# %% Rand Forest Model

# %%
# Assuming df is your DataFrame and 'plr_median' is the target variable
target = 'plr_median'

# Dropping the target variable and the high & low confidence intervals from the features
X = df.drop(columns=['_id', target, 'plr_confidence_high', 'plr_confidence_low'])
y = df[target]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# Identifying numeric and categorical features, excluding 'plr_type' from numerical features
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Ensure 'plr_type' is treated as a categorical feature
if 'plr_type' in numeric_features:
    numeric_features.remove('plr_type')
if 'plr_type' not in categorical_features:
    categorical_features.append('plr_type')

# Defining preprocessing for numeric features: imputation + scaling
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Defining preprocessing for categorical features: imputation + one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combining preprocessing steps
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)])


# %%
# Random Forest model pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])


# %%
# Defining the hyperparameter space for RandomizedSearchCV
param_grid = {
    'model__n_estimators': [100, 200, 300, 400, 500],
    'model__max_depth': [None, 10, 20, 30, 40],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['auto', 'sqrt', 'log2']
}


# %%
# Initialize the Randomized Search
neg_mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

random_search = RandomizedSearchCV(
    pipeline, 
    param_grid, 
    n_iter=40, 
    cv=5, 
    verbose=2, 
    random_state=42, 
    n_jobs=-1,
    scoring=neg_mse_scorer  # Use the negative MSE scorer
)


# %%
# Fit the model
random_search.fit(X_train, y_train)


# %%
# Output the best parameters and model score
best_params = random_search.best_params_
best_score = random_search.best_score_

print(f"Best parameters: {best_params}")
rmse = np.sqrt(-best_score)
print(rmse)

# %%
y_pred = random_search.predict(X_test)
# Calculate MAE, MSE, RMSE, and R²
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # RMSE is just the square root of MSE
r_squared = r2_score(y_test, y_pred)

# Print the metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r_squared}")




