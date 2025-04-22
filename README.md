# ClearSkyAI

**ClearSkyAI** is an AI-powered system designed to predict the Air Quality Index (AQI) across different geographic regions using key environmental features. Built with TensorFlow and Keras, this tool supports real-time air quality monitoring through deep learning regression techniques.

## Table of Contents

- [Overview](#overview)
- [Project Summary](#project-summary)
- [Requirements](#requirements)
- [How It Works](#how-it-works)
  - [Data Loading and Preprocessing](#1-data-loading-and-preprocessing)
  - [Feature Selection](#2-feature-selection)
  - [Model Training](#3-model-training)
  - [Model Evaluation](#4-model-evaluation)
- [Model Visualization](#model-visualization)
- [Neural Network Architecture](#neural-network-architecture)
- [Conclusion](#conclusion)

---

## Overview

ClearSkyAI uses a neural network to estimate AQI values based on pollutant concentrations (PM2.5, NO2, O3), geographic data (latitude, longitude), and statistical summaries. The system is capable of providing AQI predictions with high accuracy, helping users monitor and respond to environmental health concerns.

---

## Project Summary

**Dataset**:  
Air quality data for 2024 (`daily_88101_2024.csv`)

**Target Variable**:  
- AQI (Air Quality Index)

**Features**:  
- Arithmetic Mean  
- Day of Year  
- Latitude  
- Longitude  
- 1st Max Value

**Model**:  
- Feedforward Neural Network using Keras

**Evaluation Metric**:  
- Mean Absolute Error (MAE)

---

## Requirements

Install the required packages:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib pydot graphviz

## How It Works

1. Data Loading and Preprocessing
python
Copy
Edit
import pandas as pd

# Load the dataset
df = pd.read_csv("daily_88101_2024.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Filter the dataset for relevant pollutants (PM2.5, NO2, O3)
df_pollutants = df[df['Parameter Name'].str.contains('PM2.5|NO2|O3', case=False, na=False)]

# Convert 'Date Local' to datetime format
df_pollutants['Date Local'] = pd.to_datetime(df_pollutants['Date Local'])

# Extract 'Day of Year' from the Date
df_pollutants['Day of Year'] = df_pollutants['Date Local'].dt.dayofyear
2. Feature Selection
python
Copy
Edit
from sklearn.preprocessing import StandardScaler

# Select relevant features for the model
X = df_pollutants[['Arithmetic Mean', 'Day of Year', 'Latitude', 'Longitude', '1st Max Value']]
y = df_pollutants['AQI']

# Standardize the feature matrix
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
3. Model Training
python
Copy
Edit
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the neural network model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
4. Model Evaluation
python
Copy
Edit
# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae:.2f}")


## Model Visualization
Generate a visual representation of the model architecture:

python
Copy
Edit
from tensorflow.keras.utils import plot_model

# Visualize the model architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

## Neural Network Architecture
Input Layer: 5 features

Hidden Layer 1: 64 neurons, ReLU activation

Hidden Layer 2: 32 neurons, ReLU activation

Hidden Layer 3: 16 neurons, ReLU activation

Output Layer: 1 neuron (AQI)

## Conclusion
ClearSkyAI provides a robust framework for AQI prediction using neural networks. Future improvements may include:

Incorporating meteorological features (e.g., temperature, humidity)

Using advanced architectures (e.g., LSTM for temporal patterns)

Real-time deployment via API or dashboard

This system demonstrates how machine learning can enhance environmental monitoring and support data-informed public health strategies.

