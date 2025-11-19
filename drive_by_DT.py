import os
import sys
import numpy as np
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import MinMaxScaler
import scipy.io
import warnings
import time

# Check if pgmpy is installed
try:
    from pgmpy.models import DynamicBayesianNetwork as DBN
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import DBNInference
except ImportError:
    print("ImportError: pgmpy not found.")
    print("Please install pgmpy: pip install pgmpy")
    sys.exit()

# Check if plotters is available, otherwise disable plotting
try:
    from plotters.plotter import Plot # Assuming this is in a 'plotters' folder
    PLOTTING_ENABLED = True
except ImportError:
    # print("Warning: 'plotters.plotter.Plot' not found. Plotting will be disabled.")
    # print("Simulation will still run and save CSV results.")
    PLOTTING_ENABLED = False


# --- 1. Classifier/Data Configuration (for reference) ---
# These parameters defined the "offline" model we are loading
DISCRETIZED_DAMAGE = 5
# Calculate N_CLASSES dynamically
DAMAGE_CASES_TO_LOAD = list(range(0, 61, DISCRETIZED_DAMAGE))
N_CLASSES = len(DAMAGE_CASES_TO_LOAD) # Should be 13

DOF_TO_USE = 1             # Row index to use
PASSAGES_TO_LOAD = 200     # Number of passages to load
FILE_DIRECTORY = 'data_only_noise' 
N_PAA_SEGMENTS = 512  # Must match the model's training! (Using FFT)
TEST_SET_SIZE = 0.20
N_KFOLD_SPLITS = 5
N_OPTUNA_TRIALS = 20
EPOCHS_FINAL = 35
EPOCHS_OPTUNA = 25
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This will be populated by the 'load_data' function
# Based on your error, it's 5831
N_RAW_FEATURES = 5831 

# --- 2. DT Simulation Configuration ---
SIMULATION_YEARS = 60
MONITORING_INTERVAL = 1.0 # (Years) How often drive-by monitoring happens
MODEL_ACCURACY = 0.94   # Your test accuracy
# "30% damage or more" -> 30% damage is case 30.
# Mapped label = (60 - 30) // 5 = 6
# So, labels 0-6 (more damaged) trigger repair.
DAMAGE_LABEL_THRESHOLD = 6  

rnd.seed(42)
np.random.seed(40)
torch.manual_seed(40)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"

# --- 3. Preprocessing Function (must match training script) ---
def fft_preprocess(x, n_fft_bins):
    """Takes a 1D signal and returns its frequency magnitude."""
    fft_coeffs = np.fft.rfft(x)
    fft_mag = np.abs(fft_coeffs)
    if len(fft_mag) != n_fft_bins:
        fft_mag_resized = np.interp(
            np.linspace(0, len(fft_mag), n_fft_bins),
            np.arange(len(fft_mag)),
            fft_mag
        )
    else:
        fft_mag_resized = fft_mag
    return fft_mag_resized.astype(np.float32)

# --- 4. PyTorch Model Definition (must match training script) ---
# We must redefine the class here so torch.load can reconstruct it.
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
        
        n_conv_layers = self.params['n_conv_layers']
        
        for i in range(n_conv_layers):
            out_channels = self.params[f'n_filters_l{i}']
            kernel_size = self.params[f'kernel_size_l{i}']
            self.layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding='same'))
            self.layers.append(nn.ReLU())
            if self.params[f'pooling_l{i}']:
                self.layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
                current_seq_len = current_seq_len // 2
            in_channels = out_channels
            
        self.layers.append(nn.Flatten())
        flattened_size = current_seq_len * in_channels
        
        n_dense_layers = self.params['n_dense_layers']
        in_features = flattened_size
        
        for i in range(n_dense_layers):
            out_features = self.params[f'n_dense_units_l{i}']
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(self.params[f'dropout_l{i}']))
            in_features = out_features
            
        self.layers.append(nn.Linear(in_features, n_classes))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# --- 5. Helper Functions ---
def normalize_cpd(matrix):
    col_sums = matrix.sum(axis=0, keepdims=True)
    # Avoid division by zero if a column is all zeros
    col_sums[col_sums == 0] = 1
    return matrix / col_sums

def build_mock_confusion_matrix(n_classes, accuracy):
    """
    Builds a "blurry" diagonal confusion matrix based on an overall accuracy.
    This represents P(Predicted | True). We will transpose this
    for the DBN's P(True | Predicted).
    """
    if accuracy < 1.0 / n_classes:
        accuracy = 1.0 / n_classes
        
    # Calculate std dev (sigma) for the Gaussian blur
    sigma = (1.0 - accuracy) * (n_classes / 10.0) + 0.5 
    
    conf_mat = np.zeros((n_classes, n_classes))
    for i in range(n_classes): # True Label
        for j in range(n_classes): # Predicted Label
            dist = abs(i - j)
            prob = np.exp(-0.5 * (dist / sigma)**2)
            conf_mat[i, j] = prob
            
    # Normalize each *row* to sum to 1 (P(Predicted | True=i) must sum to 1)
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    conf_mat = conf_mat / row_sums
    
    # We need P(True | Predicted) for the DBN, which is the transpose
    # and column-normalized.
    conf_mat_for_dbn = normalize_cpd(conf_mat.T)
    print("Mock Confusion Matrix (for DBN) built successfully.")
    return conf_mat_for_dbn

# --- 6. DBN Framework Classes (Adapted for Scour) ---

class GetSetup:
    """
    Creates the probabilistic graphical model.
    Adapted for 1D scour damage states [0, 1, ..., 12].
    """
    def __init__(self, states_list, actions_list, conf_mat_dt):
        self.states_list = states_list
        self.actions_list = actions_list
        self.n_states = len(states_list)
        self.conf_mat_dt = conf_mat_dt
    
    def _get_low_diag_transition(self, prob_adv):
        """
        Returns a lower diagonal transition matrix.
        Damage state can only stay the same or increase by one.
        State 0 is max damage (60%). State 12 is healthy (0%).
        So, "damage" means moving from 12 -> 11 -> 10 ...
        """
        n_states = self.n_states
        trans = np.zeros((n_states, n_states))
        
        for i in range(n_states):
            if i == 0: # First state (max damage)
                trans[i, i] = 1.0 # Stays at max damage
            else:
                trans[i, i] = 1 - prob_adv  # Prob of staying in this state
                trans[i-1, i] = prob_adv  # Prob of moving to next damage state (i-1)
        return trans
    
    def _get_restart_trainsition(self):
        """Returns a transition matrix for the perfect maintenance action."""
        trans = np.zeros((self.n_states, self.n_states))
        # After repair, state is "healthy" (last state, e.g., index 12)
        trans[self.n_states - 1, :] = 1.0
        return trans
        
    def _get_combined_transition(self, t1, t2):
        """Combines two cpds: p(D_t | D_t-1, U_t-1) * p(D_t | D_NN_t)"""
        n_states = t1.shape[0]
        card_1 = t1.shape[1]
        card_2 = t2.shape[1]
        comb = np.zeros((n_states, card_1 * card_2))
        for i in range(n_states):
            for j in range(card_1):
                for k in range(card_2):
                    comb[i, j * card_2 + k] = t1[i, j] * t2[i, k]
        return normalize_cpd(comb)

    def get_transitions(self, p_nothing):
        """Returns the transition matrix for each possible action."""
        # p(D_t | D_t-1, U_t-1=do_nothing):
        trans_prob_nothing = self._get_low_diag_transition(prob_adv=p_nothing)
        # p(D_t | D_t-1, U_t-1=perfect_repair):
        trans_prob_perfect = self._get_restart_trainsition()
        return np.stack((trans_prob_nothing, trans_prob_perfect), 0)
    
    def get_graph(self, cpd_d_to_u, transitions):
        """Returns all the structures required to create the main graph."""
        states = self.states_list
        Ns = self.n_states
        actions = self.actions_list
        Na = len(actions)
        
        graph_structure = ([(('U^{A}_{-1}', 0), ('D', 0)),
                            (('D_{NN}', 0), ('D', 0)),
                            (('D', 0), ('U^{P}', 0)),
                            (('D', 0), ('D', 1)),
                            (('U^{A}_{-1}', 1), ('D', 1)),
                            (('D_{NN}', 1), ('D', 1)),
                            (('D', 1), ('U^{P}', 1)),])
        
        digital_subgraph = ([(('D', 0), ('U', 0)),
                             (('D', 0), ('D', 1)), 
                             (('U', 0), ('D', 1)),
                             (('D', 1), ('U', 1))])
        
        # p(D_t | D_t-1, U_t-1=do_nothing) * p(D_t | D_NN_t)
        cond_d_nothing = self._get_combined_transition(t1=self.conf_mat_dt, t2=transitions[0])
        # p(D_t | D_t-1, U_t-1=perfect_repair) * p(D_t | D_NN_t)
        cond_d_perfect = self._get_combined_transition(t1=self.conf_mat_dt, t2=transitions[1])
        cond_d = np.concatenate((cond_d_nothing, cond_d_perfect), 1)
        
        # --- Define CPDs ---
        U_a_0_cpd = TabularCPD(('U^{A}_{-1}', 0), Na, np.ones((Na, 1))/Na,
                               state_names={('U^{A}_{-1}', 0): actions})
                
        D_NN_0_cpd = TabularCPD(('D_{NN}', 0), Ns, np.ones((Ns, 1))/Ns,
                                state_names = {('D_{NN}', 0): states})
    
        # p(D_0 | U_A_-1, D_NN_0)
        # At t=0, D_0 does not depend on D_-1. So p(D_0|...) = p(D_0|D_NN_0)
        init_trans = np.tile(self.conf_mat_dt, (1, Na))
        D_0_cpd = TabularCPD(('D', 0), Ns, init_trans,
                             evidence=[('U^{A}_{-1}', 0),('D_{NN}', 0)],
                             evidence_card=[Na, Ns],
                             state_names={('D', 0): states,
                                          ('U^{A}_{-1}', 0): actions,
                                          ('D_{NN}', 0): states})
        
        U_p_0_cpd = TabularCPD(('U^{P}', 0), Na, cpd_d_to_u,
                               evidence=[('D', 0)], evidence_card=[Ns],
                               state_names={('U^{P}', 0): actions, ('D', 0): states})
    
        D_0_sub_cpd = TabularCPD(('D', 0), Ns, np.ones((Ns, 1))/Ns, 
                                 state_names={('D', 0): states})
    
        U_0_sub_cpd = TabularCPD(('U', 0), Na, cpd_d_to_u,
                                 evidence=[('D', 0)], evidence_card=[Ns],
                                 state_names={('U', 0): actions, ('D', 0): states})
        
        U_a_1_cpd = TabularCPD(('U^{A}_{-1}', 1), Na, np.ones((Na, 1))/Na,
                               state_names={('U^{A}_{-1}', 1): actions})
    
        D_NN_1_cpd = TabularCPD(('D_{NN}', 1), Ns, np.ones((Ns, 1))/Ns,
                                state_names = {('D_{NN}', 1): states})
    
        # This is the main transition CPD: p(D_t | D_t-1, U_t-1, D_NN_t)
        D_1_cpd = TabularCPD(('D', 1), Ns, cond_d,
                             evidence=[('U^{A}_{-1}', 1), ('D_{NN}', 1), ('D', 0)],
                             evidence_card=[Na, Ns, Ns],
                             state_names={('D', 1): states,
                                          ('U^{A}_{-1}', 1): actions,
                                          ('D_{NN}', 1): states,
                                          ('D', 0): states})
        
        U_p_1_cpd = TabularCPD(('U^{P}', 1), Na, cpd_d_to_u,
                               evidence=[('D', 1)], evidence_card=[Ns],
                               state_names={('U^{P}', 1): actions, ('D', 1): states})
    
        D_1_sub_cpd = TabularCPD(('D', 1), Ns, np.concatenate(np.array([transitions[i] for i in range(Na)]),1),
                                 evidence=[('U', 0),('D', 0)], evidence_card=[Na, Ns],
                                 state_names={('D', 1): states,
                                              ('U', 0): actions,
                                              ('D', 0): states})
        
        U_1_sub_cpd = TabularCPD(('U', 1), Na, cpd_d_to_u,
                                 evidence=[('D', 1)], evidence_card=[Ns],
                                 state_names={('U', 1): actions, ('D', 1): states})
    
        list_cpd_graph = [U_a_0_cpd, D_NN_0_cpd, D_0_cpd, U_p_0_cpd, 
                          U_a_1_cpd, D_NN_1_cpd, D_1_cpd, U_p_1_cpd]
        list_cpd_subgraph = [D_0_sub_cpd, U_0_sub_cpd, D_1_sub_cpd, U_1_sub_cpd]
        
        return graph_structure, digital_subgraph, list_cpd_graph, list_cpd_subgraph


class ScourModel:
    """
    Represents the *physical* bridge and its *true* damage state.
    Evolves using the Kamariotis et al. (2024) gradual scour model.
    """
    def __init__(self):
        self.current_X = 0.0  # Internal state X(t) from Kamariotis
        self.sample_parameters()
        self.X_MAX_FOR_MAPPING = 20.0 # Assumed max X(t)
        self.DAMAGE_CASE_MAX = 60.0   # 60% damage
        self.time = 0.0

    def sample_parameters(self):
        """Samples the stochastic parameters for the gradual model."""
        A_mean = 1.94e-4; A_cov = 0.4
        A_sigma = np.log(A_cov**2 + 1)**0.5
        A_mu = np.log(A_mean) - 0.5 * A_sigma**2
        self.A = np.random.lognormal(mean=A_mu, sigma=A_sigma) 
        
        B_mean = 2.0; B_cov = 0.10
        self.B = np.random.normal(loc=B_mean, scale=B_mean * B_cov)

    def evolve(self, delta_t):
        """Evolves the *gradual* damage (no shock) over one time step."""
        t_avg = self.time + delta_t / 2.0 
        gradual_rate = self.A * self.B * (t_avg ** (self.B - 1))
        
        omega_k_mean = -0.005; omega_k_cov = 0.10
        omega_k = np.random.normal(loc=omega_k_mean, scale=np.abs(omega_k_mean * omega_k_cov))
        
        delta_X_gradual = gradual_rate * delta_t * np.exp(omega_k)
        
        self.current_X += delta_X_gradual
        self.time += delta_t
        return self.get_current_damage_case()

    def get_current_damage_case(self):
        """Maps the internal state X(t) to your [0-60] damage case scale."""
        damage_case = (self.current_X / self.X_MAX_FOR_MAPPING) * self.DAMAGE_CASE_MAX
        return min(damage_case, self.DAMAGE_CASE_MAX) # Cap at max damage

    def repair(self):
        """Resets the physical state to "new"."""
        print(f"  -> PHYSICAL ACTION: Bridge repaired. Damage reset to 0.")
        self.current_X = 0.0
        self.time = 0.0
        self.sample_parameters()

def simulate_2d_ttbi_model(continuous_damage_percent):
    """
    (PLACEHOLDER FUNCTION)
    This function simulates a call to your 2D TTBI model.
    It takes the *true* damage, calculates stiffness, and returns a
    mock sensor signal that the classifier can process.
    """
    # 1. Calculate stiffness based on damage
    # 0% damage -> 100% stiffness
    # 60% damage -> 40% stiffness
    stiffness_percent = 100.0 - continuous_damage_percent
    
    # This is where you would call your MATLAB/Python TTBI model
    # print(f"  (Simulating TTBI model with stiffness = {stiffness_percent:.2f}%)")
    
    # 2. Create a mock signal
    # We must return a realistic-looking signal, otherwise the
    # loaded classifier (which was trained on real FFTs) will fail.
    # We create a base signal (e.g., sine waves)
    
    # *** FIX: This must match the length of the data your scaler was fit on ***
    # This value (5831) comes from the error message.
    # If your load_data function finds a different length, update this.
    N_COLS_EXPECTED = N_RAW_FEATURES 
    
    t = np.linspace(0, 1, N_COLS_EXPECTED)
    mock_signal = np.sin(2 * np.pi * 30 * t) + np.sin(2 * np.pi * 75 * t)
    
    # Add a "hint" of the damage.
    # We shift the frequency slightly based on damage.
    damage_shift_freq = 75 * (stiffness_percent / 100.0)
    mock_signal += np.sin(2 * np.pi * damage_shift_freq * t)
    
    # Add random noise (simulating 'data_only_noise')
    mock_signal += np.random.normal(0, 0.5, N_COLS_EXPECTED)
    
    return mock_signal.astype(np.float32)


class PhysicalAsset:
    """
    Physical Asset class. Simulates the ground truth.
    """
    def __init__(self, degradation_law="scour_gradual"):
        self.scour_model = ScourModel()
        self.state_continuous = 0.0 # True continuous damage (e.g., 23.4%)
        self.degradation_law = degradation_law
        self.time = 0.0
        # This global must be set by the load_data function
        if 'N_RAW_FEATURES' not in globals():
            print("Error: N_RAW_FEATURES is not set. Run a data-loading script first.")
            # Set a default to avoid crashing, but this is a problem
            global N_RAW_FEATURES
            N_RAW_FEATURES = 5831 


    def get_observation_signal(self):
        """
        Gets a new, raw sensor signal from the bridge by calling
        the (placeholder) TTBI simulation.
        """
        return simulate_2d_ttbi_model(self.state_continuous)

    def update_physical_state(self, action_str):
        """Update the physical state given the current p_state and an action."""
        if action_str == "perfect_repair":
            self.scour_model.repair()
            
        # Evolve *after* action
        if self.degradation_law == "scour_gradual":
            self.state_continuous = self.scour_model.evolve(delta_t=MONITORING_INTERVAL)
        else:
            # Placeholder for other laws
            self.state_continuous += 0.5 
            
        self.time += MONITORING_INTERVAL
        
    def get_true_mapped_label(self):
        """Gets the true continuous state and maps it to the closest discrete label."""
        # Map: (60 - 23.4) // 5 = 7
        # Label 0 = 60% (max damage), Label 12 = 0% (healthy)
        true_label = round((60 - self.state_continuous) / DISCRETIZED_DAMAGE)
        return int(max(0, min(N_CLASSES - 1, true_label)))


class DigitalAsset:
    """
    Digital Asset class. Loads the trained classifier and performs predictions.
    """
    def __init__(self, model, scaler, n_classes, n_segments):
        self.model = model
        self.scaler = scaler
        self.n_classes = n_classes
        self.n_segments = n_segments # for FFT
        self.model.to(DEVICE)
        self.model.eval() # Set model to evaluation mode

    def estimate_state(self, raw_signal_obs):
        """
        Takes a raw sensor observation, preprocesses it, and predicts a state.
        This is the "online" inference pipeline.
        """
        # 1. Preprocess the raw signal
        signal_fft = fft_preprocess(raw_signal_obs, self.n_segments)
        
        # 2. Scale the signal using the *loaded* scaler
        # scaler.transform expects a 2D array
        signal_scaled = self.scaler.transform(signal_fft.reshape(1, -1))
        
        # 3. Convert to PyTorch Tensor
        # Add channel dimension: (N, C, L) -> (1, 1, n_segments)
        signal_tensor = torch.tensor(signal_scaled).float().unsqueeze(0).to(DEVICE)
        
        # 4. Predict
        with torch.no_grad():
            outputs = self.model(signal_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted_label = torch.max(probabilities.data, 1)
            
        return predicted_label.item()
            
class Planner:
    """
    Planner class. Uses a simple threshold policy.
    """
    def __init__(self, states_list, actions_list, threshold_label):
        self.states_list = states_list
        self.actions_list = actions_list
        self.threshold_label = threshold_label # Mapped label (e.g., 6)
        self.policy = self.compute_threshold_policy()

    def compute_threshold_policy(self):
        """
        Builds a deterministic policy matrix based on the threshold.
        Labels 0-6 (damaged) -> Action 1 (repair)
        Labels 7-12 (healthy) -> Action 0 (do nothing)
        """
        Ns = len(self.states_list)
        Na = len(self.actions_list)
        policy = np.zeros([Ns, Na])
        
        # state_list[0] = '0' (max damage)
        # state_list[12] = '12' (healthy)
        # threshold_label = 6 (maps to 30% damage)
        for s in range(Ns):
            state_label = int(self.states_list[s])
            if state_label <= self.threshold_label: # 0, 1, 2, 3, 4, 5, 6
                policy[s, 1] = 1.0 # Action 1: "perfect_repair"
            else: # 7, 8, 9, 10, 11, 12
                policy[s, 0] = 1.0 # Action 0: "do_nothing"
        
        return policy

class Graph:
    """
    Graph class. Manages the DBN and the simulation loop.
    """
    def __init__(self, physical_asset, digital_asset, states_list, actions_list, planner):
        self.physical_asset = physical_asset
        self.digital_asset = digital_asset
        self.states_list = states_list
        self.actions_list = actions_list
        self.dbn = DBN()
        self.dbn_predict = DBN()
        self.hist_var = None
        self.hist_d_state_prob = None
        self.hist_actions_prob = None
        self.time_index = -1
        self.planner = planner

    def assemble_graph(self, graph_structure, cpd_list, digital_subgraph, list_cpd_subgraph):
        """Adds nodes, edges, and CPDs to the DBN."""
        self.dbn.add_edges_from(graph_structure)
        for cpd_table in cpd_list:
            cpd_table.normalize()
            self.dbn.add_cpds(cpd_table)
        self.dbn.check_model()
        print("\nMain DBN assembled and checked.")

        self.dbn_predict.add_edges_from(digital_subgraph)
        for cpd_table in list_cpd_subgraph:
            cpd_table.normalize()
            self.dbn_predict.add_cpds(cpd_table)
        self.dbn_predict.check_model()
        print("Prediction DBN assembled and checked.")
        self.time_index = -1
        
    def _label_to_damage(self, label):
        """Helper to convert a label (0-12) back to damage % (60-0)."""
        # label 0 = 60, label 12 = 0
        # formula: damage = 60 - label * 5
        damage_pct = 60 - (label * DISCRETIZED_DAMAGE)
        return damage_pct
    
    def simulate(self, n_steps, n_samples):
        """Runs the main simulation loop."""
        
        # Initialization
        if self.time_index == -1:
            self.time_index = 0
            
            # 1. Physical asset starts at t=0 and evolves for first interval
            self.physical_asset.update_physical_state(action_str="do_nothing")
            
            # 2. Get first observation
            raw_signal = self.physical_asset.get_observation_signal()
            
            # 3. DT estimates state
            predicted_label = self.digital_asset.estimate_state(raw_signal)
            
            # 4. Get initial belief (from D_0_cpd)
            # We use the DBN to get the initial belief P(D_0 | D_NN_0)
            dbn_infer_init = DBNInference(self.dbn)
            evid_init = {('D_{NN}', 0): self.states_list[predicted_label],
                         ('U^{A}_{-1}', 0): 'do_nothing'} # Assume "do_nothing" at t=-1
            inf = dbn_infer_init.forward_inference([('D', 0)], evidence=evid_init)
            last_d_state_prob = inf[('D', 0)].values

            # 5. Get first action
            current_action_prob = np.matmul(self.planner.policy.T, last_d_state_prob)
            current_action = self.actions_list[np.argmax(current_action_prob)]
            
            # 6. Initialize History
            true_label = self.physical_asset.get_true_mapped_label()
            self.hist_var = {
                'p_state': self.physical_asset.state_continuous,
                'p_state_discrete': self.states_list[true_label],
                'd_state': str(self.states_list[np.argmax(last_d_state_prob)]),
                'd_state_nn': self.states_list[predicted_label],
                'current_action': current_action,
            }
            self.hist_var = pd.DataFrame(self.hist_var, index=[0,])
            
            self.hist_d_state_prob = pd.DataFrame([last_d_state_prob.tolist()])
            self.hist_d_state_prob.columns = self.states_list

            self.hist_actions_prob = pd.DataFrame([current_action_prob.tolist()])
            self.hist_actions_prob.columns = self.actions_list

        # Main Simulation Loop
        dbn_infer = DBNInference(self.dbn)
        
        for i in range(n_steps):
            self.time_index += 1
            print(f'\nSimulation step = {i + 1} of {n_steps}, Time index = {self.time_index}')

            # 1. Get last action and belief
            last_action = self.hist_var.current_action.iloc[-1]
            last_d_state_prob = self.hist_d_state_prob.iloc[-1].to_numpy()

            # 2. Update physical state based on action
            self.physical_asset.update_physical_state(action_str=last_action)
            
            # 3. Get new observation from physical asset
            raw_signal = self.physical_asset.get_observation_signal()
            
            # 4. DT estimates new state (D_NN)
            predicted_label = self.digital_asset.estimate_state(raw_signal)

            # 5. DBN Inference
            evid = {}
            evid[('U^{A}_{-1}', 1)] = last_action
            evid[('D_{NN}', 1)] = self.states_list[predicted_label]

            # Sample from the *previous* belief distribution for D_0
            sampled = np.random.choice(len(self.states_list), size=n_samples, p=last_d_state_prob)
            
            new_d_state_prob = np.zeros(len(self.states_list))
            new_action_prob = np.zeros(len(self.actions_list))

            for j in range(n_samples):
                evid[('D', 0)] = self.states_list[sampled[j]]
                inf = dbn_infer.forward_inference([('D', 1),('U^{P}', 1)], evidence=evid)
                new_d_state_prob += inf[('D', 1)].values
                new_action_prob += inf[('U^{P}', 1)].values

            # Normalize distributions
            last_d_state_prob = normalize_cpd(new_d_state_prob)
            current_action_prob = normalize_cpd(new_action_prob)
            
            # 6. Get new action
            current_action = self.actions_list[np.argmax(current_action_prob)]
            
            # 7. Update History
            true_label = self.physical_asset.get_true_mapped_label()
            new_var = {
                'p_state': self.physical_asset.state_continuous,
                'p_state_discrete': self.states_list[true_label],
                'd_state': str(self.states_list[np.argmax(last_d_state_prob)]),
                'd_state_nn': self.states_list[predicted_label],
                'current_action': current_action,
            }
            
            self.hist_var = pd.concat([self.hist_var, pd.DataFrame(new_var, index=[0,])], ignore_index=True)
            
            new_prob_df = pd.DataFrame([last_d_state_prob.tolist()], columns=self.states_list)
            self.hist_d_state_prob = pd.concat([self.hist_d_state_prob, new_prob_df], ignore_index=True)
            
            new_action_df = pd.DataFrame([current_action_prob.tolist()], columns=self.actions_list)
            self.hist_actions_prob = pd.concat([self.hist_actions_prob, new_action_df], ignore_index=True)
            
            # --- Enhanced Printing ---
            true_dmg = self.physical_asset.state_continuous
            guess_label = predicted_label
            guess_dmg = self._label_to_damage(guess_label)
            belief_label = np.argmax(last_d_state_prob)
            belief_dmg = self._label_to_damage(belief_label)
            
            print(f"  True State:       {true_dmg:.2f}% damage (Label {true_label})")
            print(f"  Classifier Guess: {guess_dmg}% damage (Label {guess_label})")
            print(f"  DT Belief (MAP):  {belief_dmg}% damage (Label {belief_label})")
            print(f"  DT Action:        {current_action}")

    def predict(self, n_steps, n_samples):
        """Future state prediction for digital asset only."""
        print(f'\nFuture prediction from time index = {self.time_index}')
        
        # Get the DBN's *current belief* as the starting point
        current_belief = self.hist_d_state_prob.iloc[self.time_index].to_list()
        evid = [[prob] for prob in current_belief]
        
        sim = self.dbn_predict.simulate(
            n_time_slices=n_steps+1, 
            n_samples=n_samples, 
            virtual_evidence=[TabularCPD(('D', 0), len(self.states_list), evid)],
            show_progress=False
        )
        sim = sim.to_numpy()

        prob_D = np.zeros((n_steps+1, len(self.states_list)))
        prob_U = np.zeros((n_steps+1, len(self.actions_list)))

        for i in range(n_steps+1):
            for j in range(n_samples):
                prob_D[i, sim[j, 2*i]] += 1
                prob_U[i, sim[j, 2*i+1]] += 1

        prob_D = normalize_cpd(prob_D.T).T
        result_D = pd.DataFrame(prob_D.tolist(), columns=self.states_list)
        prob_U = normalize_cpd(prob_U.T).T
        result_U = pd.DataFrame(prob_U.tolist(), columns=self.actions_list)
        return result_D, result_U
         
if __name__ == '__main__':
    
    print(f"--- Initializing Scour Digital Twin ---")
    print(f"Total damage classes: {N_CLASSES}") # Should be 13
    
    # --- 1. Load Offline-Trained Models ---
    model_path = "best_cnn_model.pth"
    scaler_path = "scaler.pkl"
    cm_path = "conf_mat_dbn.npy"

    # These are the hyperparameters from your successful training run
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

    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(cm_path):
        print(f"--- OFFLINE FILES NOT FOUND ---")
        print(f"  Model: {model_path} {' (Found)' if os.path.exists(model_path) else ' (MISSING)'}")
        print(f"  Scaler: {scaler_path} {' (Found)' if os.path.exists(scaler_path) else ' (MISSING)'}")
        print(f"  Conf Mat: {cm_path} {' (Found)' if os.path.exists(cm_path) else ' (MISSING)'}")
        print("\nRunning the 'offline' classifier training pipeline first...")
        
        # 1. Import the necessary functions from the classifier script
        try:
            # We assume the classifier script is named 'cnn_classifier_fixed.py'
            from cnn_classifier_fixed import load_data, run_classifier_pipeline
        except ImportError:
            print("\n--- CRITICAL ERROR ---")
            print("Could not import 'classifier.py'.")
            print("Please make sure 'classifier.py' is in the same directory as this script.")
            sys.exit()
        except Exception as e:
            print(f"\nAn error occurred during import: {e}")
            sys.exit()

        # 2. Load the data using the settings from this file
        print(f"\nLoading data from: {FILE_DIRECTORY}")
        X_data, y_labels, X_flat, y_flat_original = load_data(
            damage_cases=DAMAGE_CASES_TO_LOAD,
            dof=DOF_TO_USE,
            Npass=PASSAGES_TO_LOAD,
            file_path=FILE_DIRECTORY
        )
        
        if X_flat is None:
            print("Data loading failed during offline phase. Exiting.")
            sys.exit()
            
        # 3. Run the full training pipeline
        # We pass `best_params=None` to force it to run Optuna
        print("\nStarting offline training pipeline (this may take a while)...")
        # This will train and save 'best_cnn_model.pth', 'scaler.pkl', and 'conf_mat_dbn.npy'
        _, _, my_saved_params = run_classifier_pipeline(
            X_flat, y_flat_original, N_CLASSES, best_params=None
        )
        print("--- Offline training complete. Files saved. ---")
        
    else:
        print("--- Offline files found. Loading them. ---")
        # Files exist, so my_saved_params remains the one hard-coded above
        pass
    
    # --- Load "Offline" Assets ---
    print(f"\nLoading classifier from: {model_path}")
    print(f"Loading scaler from: {scaler_path}")
    
    # Load the scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    # **Check the loaded scaler's features**
    if scaler.n_features_in_ != N_PAA_SEGMENTS:
        print(f"Error: Scaler feature mismatch!")
        print(f"  Scaler expects {scaler.n_features_in_} features.")
        print(f"  Script is set to generate {N_PAA_SEGMENTS} features (from N_PAA_SEGMENTS).")
        print(f"  Please ensure N_PAA_SEGMENTS in this script ({N_PAA_SEGMENTS})")
        print(f"  matches the N_PAA_SEGMENTS used to train the model.")
        sys.exit()
        
    # Load the model
    model = Simple1DCNN(N_PAA_SEGMENTS, N_CLASSES, my_saved_params)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("Classifier and scaler loaded successfully.")

    # --- 2. Setup the DBN ---
    
    # array of all the possible states (mapped labels 0-12)
    # State 0 = 60% damage, State 12 = 0% damage
    possible_states = [str(i) for i in range(N_CLASSES)]

    # array of all the possible actions
    possible_actions = ["do_nothing", "perfect_repair"]

    # Confusion matrix based on our accuracy
    print(f"Loading confusion matrix from: {cm_path}")
    conf_mat_dt = np.load(cm_path)

    setup_frame = GetSetup(states_list=possible_states,
                           actions_list=possible_actions,
                           conf_mat_dt=conf_mat_dt)

    # Assume a 5% chance of deteriorating to the next state each year
    transitions = setup_frame.get_transitions(p_nothing=0.05)

    planner_frame = Planner(states_list=possible_states,
                            actions_list=possible_actions,
                            threshold_label=DAMAGE_LABEL_THRESHOLD)

    # cpd_d_to_u is the policy matrix
    cpd_d_to_u = planner_frame.policy.T 

    graph_structure, digital_subgraph, list_cpd_graph, list_cpd_subgraph = \
        setup_frame.get_graph(cpd_d_to_u, transitions)

    # --- 3. Instantiate Assets and Graph ---
    frame = PhysicalAsset(degradation_law="scour_gradual")
    
    digital_frame = DigitalAsset(model=model,
                                 scaler=scaler,
                                 n_classes=N_CLASSES,
                                 n_segments=N_PAA_SEGMENTS)

    graph_frame = Graph(physical_asset=frame,
                        digital_asset=digital_frame,
                        states_list=possible_states, 
                        actions_list=possible_actions,
                        planner=planner_frame)

    graph_frame.assemble_graph(graph_structure=graph_structure,
                                cpd_list=list_cpd_graph,
                                digital_subgraph=digital_subgraph,
                                list_cpd_subgraph=list_cpd_subgraph)

    # --- 4. Run Simulation ---
    graph_frame.simulate(n_steps=SIMULATION_YEARS, n_samples=100)
    
    # --- 5. Plot Results ---
    if PLOTTING_ENABLED:
        try:
            # Create directory if it doesn't exist
            plot_dir = './plotters/scour_dt_results'
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
                
            plotting_frame = Plot(graph_frame, plot_dir)
            plotting_frame.plot_history_all_together()
            plotting_frame.plot_prediction_all_together(n_steps=20, n_samples=1000)
            print(f"\nPlots saved to '{plot_dir}'")
        except Exception as e:
            print(f"\nAn error occurred during plotting: {e}")
            PLOTTING_ENABLED = False

    if not PLOTTING_ENABLED:
        print("Saving history data to CSV.")
        graph_frame.hist_var.to_csv("dt_history.csv")
        graph_frame.hist_d_state_prob.to_csv("dt_belief_history.csv")