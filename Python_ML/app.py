# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 00:18:27 2024

@author: jacob
"""
# 
import tkinter as tk
from tkinter import ttk
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Load the model
model = joblib.load('../Model/deg_modelv1.joblib')

# Placeholder options for dropdowns - these should match the categories used during model training
pv_climate_zone_options = ['T2', 'T3', 'T4', 'T5', 'T6']
technology1_options = ['multi-Si', 'mono-Si', 'CdTe', 'CIGS', 'a-Si', 'c-Si']
type_mounting_options = ['Ground', 'Roof', 'Parking', 'Garage Canopy', 'Canopy', 'Canopy / Ground']
tracking_options = ['0', '1']
power_dc_options = ['< 0.5 MW', '0.5-2 MW', '> 2 MW']

# OneHotEncoder - Setup encoders for each categorical feature based on your model training
# This is a simplified version; you'll need to adjust this to match the exact encoding from training
encoders = {
    'pv_climate_zone': OneHotEncoder(categories=[pv_climate_zone_options], drop='if_binary', sparse=False),
    'technology1': OneHotEncoder(categories=[technology1_options], drop='if_binary', sparse=False),
    'type_mounting': OneHotEncoder(categories=[type_mounting_options], drop='if_binary', sparse=False),
    'tracking': OneHotEncoder(categories=[tracking_options], drop='if_binary', sparse=False),
    # Assuming 'power_dc' is handled similarly; adjust as necessary
    'power_dc': OneHotEncoder(categories=[power_dc_options], drop='if_binary', sparse=False)
}

# Function to one-hot encode input data
def one_hot_encode_input(data):
    encoded_features = []
    for feature, encoder in encoders.items():
        # Reshape is necessary as the encoder expects a 2D array
        encoded = encoder.fit_transform(np.array(data[feature]).reshape(-1, 1))
        # Flatten the encoded features to a 1D array for simplicity
        encoded_features.extend(encoded.flatten())
    return encoded_features

def predict():
    input_data = {
        'pv_climate_zone': combo_climate_zone.get(),
        'technology1': combo_technology.get(),
        'length_years_rounded': entry_age.get(),
        'tracking': combo_tracking.get(),
        'type_mounting': combo_mounting.get(),
        'power_dc': entry_power.get()
    }
    
    # Adjust the preprocessing to include one-hot encoding
    processed_data = one_hot_encode_input(input_data)
    prediction = model.predict([processed_data])
    label_prediction.config(text=f'Predicted Rate: {prediction[0]:.4f}')
# %%
# Creating the main window
root = tk.Tk()
root.title("Solar Degradation Rate Prediction")

# %%
# Create input fields for the specified features
# Dropdown for Climate Zone
label_climate_zone = ttk.Label(root, text="Climate Zone:")
label_climate_zone.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
combo_climate_zone = ttk.Combobox(root, values=pv_climate_zone_options)
combo_climate_zone.grid(column=1, row=0, sticky=tk.E, padx=5, pady=5)

# Dropdown for Technology
label_technology = ttk.Label(root, text="Technology:")
label_technology.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
combo_technology = ttk.Combobox(root, values=technology1_options)
combo_technology.grid(column=1, row=1, sticky=tk.E, padx=5, pady=5)

# Entry for Age (Years)
label_age = ttk.Label(root, text="Age (Years):")
label_age.grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
entry_age = ttk.Entry(root)
entry_age.grid(column=1, row=2, sticky=tk.E, padx=5, pady=5)

# Dropdown for Tracking
label_tracking = ttk.Label(root, text="Tracking (0=No, 1=Yes):")
label_tracking.grid(column=0, row=3, sticky=tk.W, padx=5, pady=5)
combo_tracking = ttk.Combobox(root, values=tracking_options)
combo_tracking.grid(column=1, row=3, sticky=tk.E, padx=5, pady=5)

# Dropdown for Mounting Type
label_mounting = ttk.Label(root, text="Mounting Type:")
label_mounting.grid(column=0, row=4, sticky=tk.W, padx=5, pady=5)
combo_mounting = ttk.Combobox(root, values=type_mounting_options)
combo_mounting.grid(column=1, row=4, sticky=tk.E, padx=5, pady=5)

# Entry for Power
label_power = ttk.Label(root, text="Power:")
label_power.grid(column=0, row=5, sticky=tk.W, padx=5, pady=5)
entry_power = ttk.Entry(root)
entry_power.grid(column=1, row=5, sticky=tk.E, padx=5, pady=5)

# %%
# Prediction button
btn_predict = ttk.Button(root, text="Predict", command=predict)
btn_predict.grid(column=1, row=6, sticky=tk.E, padx=5, pady=5)

# %%
# Prediction display
label_prediction = ttk.Label(root, text="Predicted Rate: ")
label_prediction.grid(column=0, row=7, columnspan=2, sticky=tk.W, padx=5, pady=5)

# %%
# Start the GUI loop
root.mainloop()





# %%   # Check the type of the model to understand what kind of model or pipeline it is
try:     
    # Check the type of the model to understand what kind of model or pipeline it is
    model_type = type(model).__name__
    
    # If the model is a Pipeline, we can list its steps to understand its structure better
    if hasattr(model, 'steps'):
        model_steps = [step[0] for step in model.steps]
    else:
        model_steps = "N/A"
        
    result = {
        "model_type": model_type,
        "model_steps": model_steps
    }
except Exception as e:
    result = {"error": str(e)}

result