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


