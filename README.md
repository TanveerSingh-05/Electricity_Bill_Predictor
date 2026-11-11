# Household Electricity Consumption & Bill Predictor using ANN

This project is a machine learning web application that predicts a household's monthly electricity consumption (in kWh) and estimates the corresponding bill based on daily appliance usage patterns. It uses a Feedforward Artificial Neural Network built with Scikit-Learn and an interactive interface created using Streamlit.

---

## Project Overview

Estimating energy consumption based on household habits involves non-linear relationships, as appliance usage compounds depending on family size and hours of operation. This project uses a Multi-Layer Perceptron (MLPRegressor) to model these relationships from input parameters and return a continuous consumption prediction.

---

## Neural Network Architecture

- Input Layer: 7 numerical features representing household appliance counts and daily usage hours.
- Hidden Layer 1: 64 Neurons with ReLU activation.
- Hidden Layer 2: 32 Neurons with ReLU activation.
- Output Layer: 1 Neuron predicting total monthly consumption (kWh).
- Optimizer: Adam Optimizer.
- Feature Scaling: StandardScaler integrated into a Scikit-Learn Pipeline to prevent data leakage and normalize input distributions.

---

## Technical Stack

- Python 3
- Scikit-Learn (MLPRegressor, Pipeline, StandardScaler, Metrics)
- Pandas & NumPy (Data processing and synthetic data creation)
- Streamlit (Web interface)
- Joblib (Model persistence)

---

## Project Structure

Electricity_Bill_Predictor/
│
├── dataset/
│   └── electricity.csv        # Synthetic dataset (1,000 records)
├── model/
│   └── ann_model.pkl          # Serialized model pipeline
├── train.py                   # Data generation, scaling, and model training
├── app.py                     # Streamlit application UI
├── requirements.txt           # Environment dependencies
└── README.md                  # Project documentation

---

## Setup and Execution

1. Clone the repository:
   git clone https://github.com/TanveerSingh-05/Electricity_Bill_Predictor.git
   cd Electricity_Bill_Predictor

2. Install the required dependencies:
   pip install -r requirements.txt

3. Train the neural network model:
   python train.py

4. Launch the local application:
   streamlit run app.py

---

## Model Evaluation

The network was evaluated on a 20% holdout test set using standard regression metrics:
- R² Score: ~0.9315
- Mean Absolute Error (MAE): ~68.74 kWh
- Root Mean Squared Error (RMSE): ~86.50 kWh

The high R² score indicates that the two-layer neural network effectively captures the non-linear interaction terms present in appliance consumption patterns.