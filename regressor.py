import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import optuna
import time
import warnings
import os

from load_data import load_data

# --- 1. Configuration ---
DISCRETIZED_DAMAGE = 5
N_CLASSES = int(61 / DISCRETIZED_DAMAGE) + 1  # Number of damage cases (not necessary for the regressor)
N_PAA_SEGMENTS = 512  # Segments for PAA (adjust based on your data length)
TEST_SET_SIZE = 0.20  # 20% of data held back for final testing
N_KFOLD_SPLITS = 5    # 5-fold CV for hyperparameter tuning
N_OPTUNA_TRIALS = 10  # Number of different architectures to test
EPOCHS_FINAL = 50     # Max epochs for final training
EPOCHS_OPTUNA = 15    # Epochs for each CV fold (keep low for speed)
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Suppress optuna's trial pruning warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 2. Load Data ---
VEHICLE = 0         # 0: First, 1: Intermediate, 2: Last
SENSOR_POS = 0      # 0: Bogie, 1: Wheel
DOF = 1             # Row index to use

# Load damage cases
DAMAGE_CASES_TO_LOAD = list(range(0, 61, DISCRETIZED_DAMAGE))

# Number of passages to load
PASSAGES_TO_LOAD = 200

# Path to your files
FILE_DIRECTORY = 'data_only_noise' # Assumes files are in the same directory

X_flat, y_flat = load_data(
        vehicle=VEHICLE,
        sensor_position=SENSOR_POS,
        dof=DOF,
        damage_cases=DAMAGE_CASES_TO_LOAD,
        Npass=PASSAGES_TO_LOAD,
        file_path=FILE_DIRECTORY
    )

# Map the classes
y_flat = y_flat // DISCRETIZED_DAMAGE

# --- 3. Preprocessing: Piecewise Aggregate Approximation (PAA) ---
def paa(x, n_segments):
    """
    Applies Piecewise Aggregate Approximation (PAA) to a 1D time series.
    """
    n = len(x)
    segment_size = max(1.0, n / n_segments)
    paa_result = np.zeros(n_segments)
    
    for i in range(n_segments):
        start_idx = int(i * segment_size)
        end_idx = int((i + 1) * segment_size)
        if i == n_segments - 1:
            end_idx = n
        
        if start_idx >= end_idx:
            if start_idx > 0: paa_result[i] = x[start_idx-1]
            else: paa_result[i] = 0.0
            continue

        segment = x[start_idx:end_idx]
        paa_result[i] = np.mean(segment)
        
    return paa_result

# --- 4. PyTorch Model Definition (Changed for Regression) ---
class Simple1DCNN_Regressor(nn.Module):
    """
    Dynamic 1D CNN for time series REGRESSION.
    """
    def __init__(self, n_segments, trial):
        super(Simple1DCNN_Regressor, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = 1
        current_seq_len = n_segments
        
        n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3)
        for i in range(n_conv_layers):
            out_channels = trial.suggest_int(f'n_filters_l{i}', 32, 128, log=True)
            kernel_size = trial.suggest_int(f'kernel_size_l{i}', 3, 7, step=2)
            self.layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding='same'))
            self.layers.append(nn.ReLU())
            if trial.suggest_categorical(f'pooling_l{i}', [True, False]):
                self.layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
                current_seq_len = current_seq_len // 2
            in_channels = out_channels
            
        self.layers.append(nn.Flatten())
        flattened_size = current_seq_len * in_channels
        
        n_dense_layers = trial.suggest_int('n_dense_layers', 1, 2)
        in_features = flattened_size
        for i in range(n_dense_layers):
            out_features = trial.suggest_int(f'n_dense_units_l{i}', 32, 128, log=True)
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(trial.suggest_float(f'dropout_l{i}', 0.1, 0.5)))
            in_features = out_features
            
        # --- KEY CHANGE ---
        # Final output layer: 1 neuron, no activation
        self.layers.append(nn.Linear(in_features, 1))
        # ---
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# --- 5. Training & Evaluation Functions (Changed for Regression) ---
def train_model(model, loader, optimizer, criterion, device):
    model.train()
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch) # y_batch is now float
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
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
    # --- KEY CHANGE ---
    # We now calculate Mean Squared Error, not accuracy
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    mse = mean_squared_error(all_labels, all_preds)
    return mse # Return MSE

# --- 6. Optuna Objective (Changed for Regression) ---
def objective(trial, X_train_val, y_train_val, n_segments, device):
    """
    The objective function for Optuna to optimize (minimizing MSE).
    """
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    
    kf = KFold(n_splits=N_KFOLD_SPLITS, shuffle=True, random_state=42)
    cv_mses = []
    
    print(f"\nOptuna Trial {trial.number}:", end=" ")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
        model = Simple1DCNN_Regressor(n_segments, trial).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # --- KEY CHANGE ---
        criterion = nn.MSELoss() # Mean Squared Error Loss
        
        X_train_fold, X_val_fold = X_train_val[train_idx], X_train_val[val_idx]
        y_train_fold, y_val_fold = y_train_val[train_idx], y_train_val[val_idx]
        
        train_dataset = TensorDataset(X_train_fold, y_train_fold)
        val_dataset = TensorDataset(X_val_fold, y_val_fold)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        for epoch in range(EPOCHS_OPTUNA):
            train_model(model, train_loader, optimizer, criterion, device)
            
        val_mse = evaluate_model(model, val_loader, device)
        cv_mses.append(val_mse)
        print(f"F{fold+1}..", end="")

    avg_mse = np.mean(cv_mses)
    print(f" Avg. MSE: {avg_mse:.4f}")
    
    trial.report(avg_mse, 0)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
        
    return avg_mse # Optuna will MINIMIZE this

# --- 7. Main Execution Function (Changed for Regression) ---
def run_regressor_pipeline(X_flat, y_flat):
    """
    Runs the full regressor pipeline.
    """
    
    # --- Reshape and Map Data for Regressor ---
    try:
        # We use the ORIGINAL damage case numbers (0, 2, 4...) as the target
        # We also make sure it's float and (N, 1) shape for the MSELoss
        y_flat_original = y_flat.reshape((-1, 1)).astype(np.float32)
        # ---
        
        print(f"\nData reshaped to: X_flat {X_flat.shape}")
        print(f"Target labels y_flat: {y_flat_original.shape} (values 0.0, 2.0, ... 60.0)")
        
    except Exception as e:
        print(f"Error reshaping/mapping data: {e}. Make sure X and y are loaded correctly.")
        return

    print(f"Using device: {DEVICE}")
    
    # --- Split Data ---
    # We stratify based on the original integer labels to ensure all classes are represented
    # y_stratify = y_2d_original_cases.reshape(-1)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_flat, y_flat_original, test_size=TEST_SET_SIZE, random_state=42, stratify=y_flat
    )
    print(f"Train/Val set size: {X_train_val.shape[0]}, Test set size: {X_test.shape[0]}")

    # --- Preprocessing (Normalization & PAA) ---
    print("\nStarting preprocessing...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_val_norm = scaler.fit_transform(X_train_val)
    X_test_norm = scaler.transform(X_test)
    
    print(f"Applying PAA to reduce {X_flat.shape[1]} features to {N_PAA_SEGMENTS} segments...")
    X_train_val_paa = np.array([paa(x, N_PAA_SEGMENTS) for x in X_train_val_norm])
    X_test_paa = np.array([paa(x, N_PAA_SEGMENTS) for x in X_test_norm])
    
    # --- Convert to PyTorch Tensors ---
    X_train_val_tensor = torch.tensor(X_train_val_paa).float().unsqueeze(1)
    y_train_val_tensor = torch.tensor(y_train_val).float() # Target is float
    
    X_test_tensor = torch.tensor(X_test_paa).float().unsqueeze(1)
    y_test_tensor = torch.tensor(y_test).float() # Target is float
    
    print(f"Final tensor shape for CNN input: {X_train_val_tensor.shape}")

    # --- Hyperparameter Optimization ---
    print(f"\n--- Starting Optuna Hyperparameter Search ({N_OPTUNA_TRIALS} trials) ---")
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    # --- KEY CHANGE ---
    study = optuna.create_study(direction='minimize', pruner=pruner) # We MINIMIZE MSE
    study.optimize(
        lambda trial: objective(
            trial, X_train_val_tensor, y_train_val_tensor, N_PAA_SEGMENTS, DEVICE
        ),
        n_trials=N_OPTUNA_TRIALS
    )
    
    best_params = study.best_params
    print(f"\nBest trial finished with MSE: {study.best_value:.4f}")
    print("Best hyperparameters found:")
    print(best_params)

    # --- Final Model Training ---
    print("\n--- Training Final Model on All Train/Val Data ---")
    best_trial = study.best_trial
    final_model = Simple1DCNN_Regressor(N_PAA_SEGMENTS, best_trial).to(DEVICE)
    
    train_val_dataset = TensorDataset(X_train_val_tensor, y_train_val_tensor)
    train_val_loader = DataLoader(train_val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])
    # --- KEY CHANGE ---
    criterion = nn.MSELoss()

    start_time = time.time()
    for epoch in range(EPOCHS_FINAL):
        train_model(final_model, train_val_loader, optimizer, criterion, DEVICE)
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS_FINAL - 1:
            # Quick check on training MSE
            train_mse = evaluate_model(final_model, train_val_loader, DEVICE)
            print(f"Epoch [{epoch+1}/{EPOCHS_FINAL}], Train RMSE: {np.sqrt(train_mse):.4f}") # Show RMSE
            
    print(f"Final training complete in {time.time() - start_time:.2f}s")
    
    model_path = "best_cnn_regressor_model.pth"
    torch.save(final_model.state_dict(), model_path)
    print(f"Final model saved to '{model_path}'")

    # --- Final Evaluation on Test Set ---
    print("\n--- Evaluating Model on Unseen Test Set ---")
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    test_mse = evaluate_model(final_model, test_loader, DEVICE)
    test_rmse = np.sqrt(test_mse)
    print(f"Final Test Set MSE: {test_mse:.4f}")
    print(f"Final Test Set RMSE: {test_rmse:.4f}") # Root Mean Squared Error

    # --- Plot Results (Scatter Plot, not Confusion Matrix) ---
    print("Generating results plot...")
    final_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = final_model(X_batch)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    # Calculate R-squared
    r2 = r2_score(all_labels, all_preds)
    print(f"Test Set R-squared: {r2:.4f}")

    plt.figure(figsize=(10, 10))
    # Plot a 1:1 line
    min_val = min(np.min(all_labels), np.min(all_preds)) - 2
    max_val = max(np.max(all_labels), np.max(all_preds)) + 2
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Scatter plot of True vs. Predicted
    sns.scatterplot(x=all_labels, y=all_preds, alpha=0.5)
    
    plt.xlabel('True Damage Case (0, 2, ..., 60)')
    plt.ylabel('Predicted Damage Case')
    plt.title(f'Regression Results - Test Set (RMSE: {test_rmse:.4f}, $R^2$: {r2:.4f})')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    plot_path = 'regression_results.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Results scatter plot saved to '{plot_path}'")

# --- Run the main function ---
if __name__ == "__main__":
    if 'X_flat' in locals() and 'y_flat' in locals():
        run_regressor_pipeline(X_flat, y_flat)
    else:
        print("Error: 'X_flat' and 'y_flat' are not defined.")
        print("Please load your data (X) and original labels (y) before running.")

