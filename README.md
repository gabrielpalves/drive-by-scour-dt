# Probabilistic Digital Twin for Drive-By Scour Monitoring of Railway Bridges

This repository contains a **Digital Twin (DT) framework** for monitoring scour in railway bridges using drive-by data. It integrates a **Dynamic Bayesian Network (DBN)** for probabilistic state estimation with a **Deep Learning (CNN) classifier** for interpreting sensor data.

The framework is designed to simulate the entire lifecycle of a bridge, tracking the evolution of scour damage and recommending maintenance actions based on a reliability-based policy.

## 🚀 Key Features
* **Probabilistic Framework:** Uses a Dynamic Bayesian Network (DBN) to maintain a belief distribution over damage states, robust to sensor noise.

* **Deep Learning Classifier:** Implements a 1D CNN (optimized via Bayesian Optimization) to classify damage levels from drive-by vibration signals.

* **Physics-Based Simulation:** Includes a stochastic degradation model based on Kamariotis et al. (2024) to simulate realistic scour evolution.

* **Drive-By Monitoring:** Simulates the collection of acceleration data from passing trains to infer structural health.

* **Automated Workflow:** Includes a full pipeline for data loading, offline classifier training, and online DT simulation.

## 📂 Repository Structure
* `drive_by_DT.py`: **Main entry point.** Runs the online Digital Twin simulation. It handles the physical asset evolution, the DBN inference, and decision-making.

* `classifier.py`: **Offline Training Module.** Trains the CNN classifier on the dataset, performs hyperparameter optimization (Optuna), and saves the model artifacts (.pth, .pkl, .npy).

* `load_data.py`: **Data Loader.** Utilities for loading and processing MATLAB (.mat) data files containing bridge response simulations.

* `data_only_noise/`: Directory containing the simulated bridge response data (files `0001.mat` to `0061.mat`).

## 🛠️ Installation
1. Clone the repository

~~~
git clone https://github.com/gabrielpalves/drive-by-scour-dt
cd drive-by-scour-dt
~~~

2. Install the required Python packages

~~~
pip install numpy pandas matplotlib scipy torch scikit-learn seaborn pgmpy optuna
~~~

## 🏃‍♂️ Usage
To run the full framework, simply execute the main script:

~~~
python drive-by-DT.py
~~~

How it works:

1. **Check:** The script first checks for trained model artifacts (`best_cnn_model.pth`, `scaler.pkl`, `conf_mat_dbn.npy`).

2. **Train (Offline Phase):** If artifacts are missing, it automatically calls `classifier.py` to:
  * Load the data.
  * Optimize CNN hyperparameters using Optuna.
  * Train the final model.
  * Generate the confusion matrix for the DBN.

3. **Simulate (Online Phase):** Once the model is ready, it starts the 60-year simulation loop:
  * Evolves the physical bridge state (scour).
  * Simulates a train passage and gets sensor data.
  * Uses the CNN to predict the damage state.
  * Updates the DBN belief based on the prediction and transition logic.
  * Decides on maintenance (Do Nothing vs. Repair).

## 📊 Methodology
1. Digital Twin Framework
The core of the DT is a Dynamic Bayesian Network (adapted from Torzoni et al., 2024). It fuses:
* **Prior Belief:** The state from the previous time step.
* **Transition Model:** A probabilistic model of how scour evolves over time (gradual deterioration).
* **Observation Model:** The likelihood of the true state given the CNN's prediction (derived from the classifier's confusion matrix).

2. Damage Classification
A 1D Convolutional Neural Network (CNN) is used to map time-series signals (processed via FFT) to discrete damage classes.
* **Input:** Frequency-domain representation of acceleration signals.
* **Output:** One of 13 discrete damage levels (0% to 60%).
* **Optimization:** Hyperparameters (layers, filters, learning rate) are tuned using Optuna.

3. Physical Asset Simulation
The ground truth is simulated using a physics-based deterioration model:
* **Gradual Scour:** modeled as a stochastic process $X(t) = A t^B \exp(\omega(t))$ (Kamariotis et al., 2024).
* **Damage Mapping:** Continuous damage is discretized into classes for the DBN (e.g., Label 0 = 60% Damage, Label 12 = Healthy).

## 📝 References

* **Torzoni et al. (2024)** - "A digital twin framework for civil engineering structures"
* **Kamariotis et al. (2024)** - "A framework for quantifying the value of vibration-based structural health monitoring"
* **Fernandes et al. (2025)** - "Early Multi-damage Classification in Railway Bridges Using Drive-by Numerical Measurements"
