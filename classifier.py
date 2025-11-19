import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import optuna
import time
import warnings
import os
import scipy.io
import random
import pickle
import sys

# --- 1. Data load function ---
# Assuming 'load_data.py' exists in the same directory
try:
    from load_data import load_data
except ImportError:
    print("Error: 'load_data.py' not found.")
    print("Please make sure your 'load_data.py' script is in the same directory.")
    sys.exit()

# --- 2. Configuration ---
DISCRETIZED_DAMAGE = 5
# Calculate N_CLASSES dynamically
DAMAGE_CASES_TO_LOAD = list(range(0, 61, DISCRETIZED_DAMAGE))
N_CLASSES = len(DAMAGE_CASES_TO_LOAD) # Should be 13

N_PAA_SEGMENTS = 512  # Segments for FFT bins
TEST_SET_SIZE = 0.20  # 20% of data held back for final testing
N_KFOLD_SPLITS = 5    # 5-fold CV for hyperparameter tuning
N_OPTUNA_TRIALS = 20  # Number of different architectures to test
EPOCHS_FINAL = 35     # Max epochs for final training
EPOCHS_OPTUNA = 25    # Epochs for each CV fold (keep low for speed)
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Suppress optuna's trial pruning warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 3. Preprocessing & Helper Functions ---

def normalize_cpd(matrix):
    """Normalizes a matrix by its columns."""
    col_sums = matrix.sum(axis=0, keepdims=True)
    # Avoid division by zero if a column is all zeros
    col_sums[col_sums == 0] = 1
    return matrix / col_sums

def paa(x, n_segments):
    """
    Applies Piecewise Aggregate Approximation (PAA) to a 1D time series.
    Inspired by Fernandes et al. (2025), Sec 4.2.
    """
    n = len(x)
    # Ensure segments are at least 1 point long
    segment_size = max(1.0, n / n_segments)
    paa_result = np.zeros(n_segments)
    
    for i in range(n_segments):
        start_idx = int(i * segment_size)
        end_idx = int((i + 1) * segment_size)
        # Handle the last segment to include all remaining points
        if i == n_segments - 1:
            end_idx = n
        
        # Ensure we don't have an empty slice
        if start_idx >= end_idx:
            if start_idx > 0:
                paa_result[i] = x[start_idx-1] # Use previous value
            else:
                paa_result[i] = 0.0 # Should not happen
            continue

        segment = x[start_idx:end_idx]
        paa_result[i] = np.mean(segment)
        
    return paa_result

def fft_preprocess(x, n_fft_bins):
    """Takes a 1D signal and returns its frequency magnitude."""
    # Get the real part of the FFT
    fft_coeffs = np.fft.rfft(x)
    # Get the magnitude
    fft_mag = np.abs(fft_coeffs)
    
    # Resize to a fixed number of bins (simple interpolation)
    # This is a basic way to get a fixed size, like PAA
    if len(fft_mag) != n_fft_bins:
        fft_mag_resized = np.interp(
            np.linspace(0, len(fft_mag), n_fft_bins),
            np.arange(len(fft_mag)),
            fft_mag
        )
    else:
        fft_mag_resized = fft_mag
        
    return fft_mag_resized.astype(np.float32)

# --- 4. PyTorch Model Definition ---
class Simple1DCNN(nn.Module):
    """
    Dynamic 1D CNN for time series classification.
    The architecture is defined by the hyperparameters passed from a params dict.
    """
    def __init__(self, n_segments, n_classes, params):
        super(Simple1DCNN, self).__init__()
        self.params = params
        self.layers = nn.ModuleList()
        in_channels = 1
        current_seq_len = n_segments
        
        # --- FIX: Read from params dict, not trial ---
        n_conv_layers = self.params['n_conv_layers']
        
        for i in range(n_conv_layers):
            # --- FIX: Read from params dict, not trial ---
            out_channels = self.params[f'n_filters_l{i}']
            kernel_size = self.params[f'kernel_size_l{i}']
            
            self.layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding='same'))
            self.layers.append(nn.ReLU())
            
            # --- FIX: Read from params dict, not trial ---
            if self.params[f'pooling_l{i}']:
                self.layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
                current_seq_len = current_seq_len // 2 # Update sequence length
            
            in_channels = out_channels
            
        self.layers.append(nn.Flatten())
        
        # Calculate flattened size
        flattened_size = current_seq_len * in_channels
        
        # --- FIX: Read from params dict, not trial ---
        n_dense_layers = self.params['n_dense_layers']
        in_features = flattened_size
        
        for i in range(n_dense_layers):
            # --- FIX: Read from params dict, not trial ---
            out_features = self.params[f'n_dense_units_l{i}']
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(nn.ReLU())
            # --- FIX: Read from params dict, not trial ---
            self.layers.append(nn.Dropout(self.params[f'dropout_l{i}']))
            in_features = out_features
            
        # Final output layer
        self.layers.append(nn.Linear(in_features, n_classes))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# --- 5. Training & Evaluation Functions ---
def train_model(model, loader, optimizer, criterion, device):
    model.train()
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# --- 6. Optuna Hyperparameter Optimization Objective ---
def objective(trial, X_train_val, y_train_val, n_segments, n_classes, device):
    """
    The objective function for Optuna to optimize.
    Uses 5-fold cross-validation.
    """
    
    # --- Build params dict from trial ---
    params = {}
    params['lr'] = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    
    params['n_conv_layers'] = trial.suggest_int('n_conv_layers', 1, 3)
    for i in range(params['n_conv_layers']):
        params[f'n_filters_l{i}'] = trial.suggest_int(f'n_filters_l{i}', 32, 128, log=True)
        params[f'kernel_size_l{i}'] = trial.suggest_int(f'kernel_size_l{i}', 3, 7, step=2)
        params[f'pooling_l{i}'] = trial.suggest_categorical(f'pooling_l{i}', [True, False])

    params['n_dense_layers'] = trial.suggest_int('n_dense_layers', 1, 2)
    for i in range(params['n_dense_layers']):
        params[f'n_dense_units_l{i}'] = trial.suggest_int(f'n_dense_units_l{i}', 32, 128, log=True)
        params[f'dropout_l{i}'] = trial.suggest_float(f'dropout_l{i}', 0.1, 0.5)
    
    kf = KFold(n_splits=N_KFOLD_SPLITS, shuffle=True, random_state=42)
    cv_accuracies = []
    
    print(f"\nOptuna Trial {trial.number}:", end=" ")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
        # Pass params dict to constructor
        model = Simple1DCNN(n_segments, n_classes, params).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        criterion = nn.CrossEntropyLoss()
        
        X_train_fold, X_val_fold = X_train_val[train_idx], X_train_val[val_idx]
        y_train_fold, y_val_fold = y_train_val[train_idx], y_train_val[val_idx]
        
        train_dataset = TensorDataset(X_train_fold, y_train_fold)
        val_dataset = TensorDataset(X_val_fold, y_val_fold)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        for epoch in range(EPOCHS_OPTUNA): # Quick training for CV
            train_model(model, train_loader, optimizer, criterion, device)
            
        val_acc = evaluate_model(model, val_loader, device)
        cv_accuracies.append(val_acc)
        print(f"F{fold+1}..", end="")

    avg_accuracy = np.mean(cv_accuracies)
    print(f" Avg. Acc: {avg_accuracy:.4f}")
    
    # Add pruning
    trial.report(avg_accuracy, fold)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
        
    return avg_accuracy # Optuna will maximize this

# --- 7. Main Execution Function ---
def run_classifier_pipeline(X_flat, y_flat_original, n_classes, best_params=None):
    """
    Runs the full training pipeline.
    
    Args:
        X_flat (np.ndarray): 2D array of all time series data.
        y_flat_original (np.ndarray): 1D array of original damage labels (60, 58, ... 0).
        n_classes (int): The number of classes (e.g., 13).
        best_params (dict, optional): If provided, skips Optuna and uses these params.

    Returns:
        final_model (torch.nn.Module): The trained PyTorch model.
        scaler (MinMaxScaler): The fitted scaler object (for online use).
    """
    print(f"Using device: {DEVICE}")
    
    # --- Correct Label Mapping ---
    # y_flat_original contains [60, 55, 50, ..., 0]
    # We map this to [0, 1, 2, ..., 12]
    # (60 - 60) // 5 = 0
    # (60 - 55) // 5 = 1
    # ...
    # (60 - 0) // 5 = 12
    y_flat_mapped = (60 - y_flat_original) // DISCRETIZED_DAMAGE
    
    # --- Split Data ---
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_flat, y_flat_mapped, test_size=TEST_SET_SIZE, random_state=42, stratify=y_flat_mapped
    )
    print(f"Train/Val set size: {X_train_val.shape[0]}, Test set size: {X_test.shape[0]}")

    # --- ### CORRECTED PREPROCESSING ORDER ### ---
    print("\nStarting preprocessing...")
    
    # 1. Apply FFT to raw data
    print(f"Applying FFT to reduce {X_flat.shape[1]} features to {N_PAA_SEGMENTS} bins...")
    X_train_val_fft = np.array([fft_preprocess(x, N_PAA_SEGMENTS) for x in X_train_val])
    X_test_fft = np.array([fft_preprocess(x, N_PAA_SEGMENTS) for x in X_test])

    # 2. Fit Scaler on FFT data
    print(f"Fitting MinMaxScaler on {N_PAA_SEGMENTS}-feature FFT data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_val_scaled = scaler.fit_transform(X_train_val_fft)
    X_test_scaled = scaler.transform(X_test_fft)
    
    # --- ### END OF FIX ### ---
    
    # --- Convert to PyTorch Tensors ---
    # Add a channel dimension for Conv1D: (N, C, L)
    X_train_val_tensor = torch.tensor(X_train_val_scaled).float().unsqueeze(1)
    y_train_val_tensor = torch.tensor(y_train_val).long()
    
    X_test_tensor = torch.tensor(X_test_scaled).float().unsqueeze(1)
    y_test_tensor = torch.tensor(y_test).long()
    
    print(f"Final tensor shape for CNN input: {X_train_val_tensor.shape}")

    # --- Hyperparameter Optimization (Conditional) ---
    if best_params is None:
        print(f"\n--- Starting Optuna Hyperparameter Search ({N_OPTUNA_TRIALS} trials) ---")
        # Add a pruner to speed up the search
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(direction='maximize', pruner=pruner)
        study.optimize(
            lambda trial: objective(
                trial, X_train_val_tensor, y_train_val_tensor, N_PAA_SEGMENTS, n_classes, DEVICE
            ),
            n_trials=N_OPTUNA_TRIALS
        )
        
        best_params = study.best_params
        print(f"\nBest trial finished with accuracy: {study.best_value:.4f}")
        print("Best hyperparameters found:")
        print(best_params)
    else:
        print("\n--- Skipping Optuna: Using provided hyperparameters ---")
        print(best_params)

    # --- Final Model Training ---
    print("\n--- Training Final Model on All Train/Val Data ---")
    
    # Create the model using the best_params dictionary
    final_model = Simple1DCNN(N_PAA_SEGMENTS, n_classes, best_params).to(DEVICE)
    
    train_val_dataset = TensorDataset(X_train_val_tensor, y_train_val_tensor)
    train_val_loader = DataLoader(train_val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(EPOCHS_FINAL):
        train_model(final_model, train_val_loader, optimizer, criterion, DEVICE)
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS_FINAL - 1:
            # Quick check on training accuracy
            train_acc = evaluate_model(final_model, train_val_loader, DEVICE)
            print(f"Epoch [{epoch+1}/{EPOCHS_FINAL}], Train Accuracy: {train_acc:.4f}")
            
    print(f"Final training complete in {time.time() - start_time:.2f}s")
    
    # Save the model and the scaler
    model_path = "best_cnn_model.pth"
    scaler_path = "scaler.pkl"
    
    torch.save(final_model.state_dict(), model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"Final model saved to '{model_path}'")
    print(f"Data scaler saved to '{scaler_path}'")

    # --- Final Evaluation on Test Set ---
    print("\n--- Evaluating Model on Unseen Test Set ---")
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    test_acc = evaluate_model(final_model, test_loader, DEVICE)
    print(f"Final Test Set Accuracy: {test_acc:.4f}")

    # --- Plot and Save Confusion Matrix ---
    print("Generating and saving DBN confusion matrix...")
    final_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = final_model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    # The labels for the matrix are the mapped labels, [0, 1, ..., 12]
    class_labels = list(range(n_classes))
    # cm[i, j] is True=i, Pred=j
    cm = confusion_matrix(all_labels, all_preds, labels=class_labels)
    
    # --- NEW: Process and Save the DBN Confusion Matrix ---
    # The DBN needs P(True | Predicted), which is the
    # column-normalized version of the standard confusion matrix.
    conf_mat_for_dbn = normalize_cpd(cm)
    # Add a small epsilon to avoid P=0, which can cause issues
    conf_mat_for_dbn += 1e-10 
    conf_mat_for_dbn = normalize_cpd(conf_mat_for_dbn)
    
    cm_path = 'conf_mat_dbn.npy'
    np.save(cm_path, conf_mat_for_dbn)
    print(f"DBN-ready confusion matrix P(True|Predicted) saved to '{cm_path}'")
    
    # --- UPDATED: Plot the row-normalized matrix (P(Predicted | True)) ---
    # This is more intuitive for visualization.
    cm_row_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_row_sum[cm_row_sum == 0] = 1 # Avoid division by zero
    cm_normalized = cm.astype('float') / cm_row_sum
    cm_normalized = np.nan_to_num(cm_normalized) 
    
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_labels, 
                yticklabels=class_labels, vmin=0, vmax=1)
    
    # Show ticks only every 5 labels to avoid clutter
    tick_spacing = 5
    if n_classes < 15: # Adjust for 13 classes
        tick_spacing = 1 
        
    plt.xticks(ticks=np.arange(n_classes)[::tick_spacing] + 0.5, 
               labels=class_labels[::tick_spacing])
    plt.yticks(ticks=np.arange(n_classes)[::tick_spacing] + 0.5, 
               labels=class_labels[::tick_spacing], rotation=0)

    plt.title(f'Normalized Confusion Matrix (P(Predicted | True)) - Test Set Accuracy: {test_acc:.4f}')
    plt.ylabel(f'True Mapped Label (0={60}% dmg ... {n_classes-1}={0}% dmg)')
    plt.xlabel(f'Predicted Mapped Label (0={60}% dmg ... {n_classes-1}={0}% dmg)')
    
    plot_path = 'confusion_matrix.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix plot saved to '{plot_path}'")
    
    # Return the trained model, the scaler, and the parameters
    return final_model, scaler, best_params
    

# --- Run the main function ---
if __name__ == "__main__":
    # Ensure you have the required libraries:
    # pip install torch numpy matplotlib seaborn scikit-learn optuna scipy
    
    # --- Configuration for Data Loading ---
    DOF_TO_USE = 1             # Row index to use
    
    # Load damage cases [0, 5, 10, ..., 60]
    DISCRETIZED_DAMAGE = 5
    DAMAGE_CASES_TO_LOAD = list(range(0, 61, DISCRETIZED_DAMAGE))
    
    # Calculate N_CLASSES dynamically
    N_CLASSES = len(DAMAGE_CASES_TO_LOAD)
    print(f"Total number of classes: {N_CLASSES}") # Should be 13
    
    PASSAGES_TO_LOAD = 200     # Number of passages to load
    FILE_DIRECTORY = 'data_only_noise'       # Assumes files are in the same directory
    
    # --- Run Load Data Function ---
    X_data, y_labels, X_flat, y_flat_original = load_data(
        damage_cases=DAMAGE_CASES_TO_LOAD,
        dof=DOF_TO_USE,
        Npass=PASSAGES_TO_LOAD,
        file_path=FILE_DIRECTORY
    )
    
    if X_flat is not None and y_flat_original is not None:
        
        # --- EXAMPLE 1: Run Optuna to find best parameters ---
        # print("\n--- RUNNING WITH OPTUNA SEARCH ---")
        # trained_model, data_scaler, found_params = run_classifier_pipeline(
        #     X_flat, y_flat_original, N_CLASSES, best_params=None
        # )
        # print("\n--- OFFLINE PHASE COMPLETE ---")
        # print("Trained model object:", trained_model)
        # print("Data scaler object:", data_scaler)
        # print("Found parameters:", found_params)
        
        # --- EXAMPLE 2: Skip Optuna and use known parameters ---
        my_saved_params = {
        'lr': 0.0007692906789968344,
        'n_conv_layers': 3,
        'n_filters_l0': 68,
        'kernel_size_l0': 7,
        'pooling_l0': True,
        'n_filters_l1': 116,
        'kernel_size_l1': 5,
        'pooling_l1': True,
        'n_filters_l2': 107,
        'kernel_size_l2': 7,
        'pooling_l2': False,
        'n_dense_layers': 2,
        'n_dense_units_l0': 71,
        'dropout_l0': 0.25977147662915395,
        'n_dense_units_l1': 49,
        'dropout_l1': 0.32846734092972624
        }
        
        print("\n--- RUNNING... SKIPPING OPTUNA ---")
        trained_model_2, _, _ = run_classifier_pipeline(
            X_flat, y_flat_original, N_CLASSES, best_params=my_saved_params
        )
        
    else:
        print("Error: Data loading failed.")
        print(f"Please check 'FILE_DIRECTORY' ('{FILE_DIRECTORY}') and ensure .mat files are present.")