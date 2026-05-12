
#%%
import numpy as np
print(np.__version__)
print(np.__file__)
#%%
# =========================
import uproot
import awkward as ak
import pandas as pd
# =========================
# Classical Machine Learning (CML)
# =========================
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

# =========================
# Quantum Machine Learning (QML)
# =========================
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler

from qiskit.circuit.library import (
    ZZFeatureMap,
    RealAmplitudes
)

from qiskit_algorithms.optimizers import (
    COBYLA,
    SPSA
)

from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# =========================
# Plotting & utilities
# =========================
import matplotlib.pyplot as plt
#%%
#-------------------------------------------------------
#Loading variables
#-------------------------------------------------------
import os


BASE_DIR = "D:/cern/ML/lep_had"

FEATURE_NAMES = [

    "HT",
    "dR_tW",
    "m_top",
    "centrality",
    "aplanarity",
    "m_W",
    "jet_pt_k"
]

CUT_VARIABLES = [
    "N_bjets",
    "N_jets"]

def load_dataset(sample_name):
    
    folder = os.path.join(BASE_DIR, f"{sample_name}")

    data = {}

    # --- load physics features ---
    for feat in FEATURE_NAMES:
        path = os.path.join(folder, f"{feat}.npy")
        data[feat] = np.load(path)
    for var in CUT_VARIABLES:
        path = os.path.join(folder, f"{var}.npy")
        data[var] = np.load(path)

    # --- load generator  weights ---
    weight_path = os.path.join(folder, "genWeight.npy")
    data["genweight"] = np.load(weight_path)

    return data

tW_top = load_dataset("output_tW_top")
tW_antitop = load_dataset("output_tW_antitop")
ttbar = load_dataset("output_ttbar")

#%%
#computing event weights

SIGMA = {
    "tW_top": 32.45,
    "tW_antitop": 32.51,
    "ttbar": 687.1
}

GSW = {
       "tW_top": 1719571.8112018944,
       "tW_antitop": 942760.9999996417,
       "ttbar": 5341946.080000233}


LUMI_PB = 36000.0
def compute_event_weights(data, process_name):

    genw = data["genweight"]

    sigma = SIGMA[process_name]

    sumw = GSW[process_name]

    scale = (sigma * LUMI_PB) / sumw

    data["weight"] = genw * scale

    print(f"{process_name}")
    print(f"  Σ(genWeight) = {sumw:.3e}")
    print(f"  scale factor = {scale:.6e}")
    print(f"  Σ(eventWeight) = {np.sum(data['weight']):.3e}\n")

    return data

tW_top = compute_event_weights(tW_top, "tW_top")
tW_antitop = compute_event_weights(tW_antitop, "tW_antitop")
ttbar = compute_event_weights(ttbar, "ttbar")
#%%
# ------------------------------------------------------
# PRESELECTION (Hadronic Channel)
# ------------------------------------------------------
print(f"events for tW_top_hadronic before preselection = {len(tW_top['HT'])}")
print(f"events for tW_antitop_hadronic before preselection = {len(tW_antitop['HT'])}")
print(f"events for ttbar_hadronic before preselction = {len(ttbar['HT'])}")
# ------------------------------------------------------
# REGION SPLIT (BOOSTED REGIME)
# ------------------------------------------------------

def split_regions(data):

    regions = {}

    # Region 0 → Nb = 1
    mask_1b = (data["N_bjets"] == 1) 
    
    # Region 1 → Nb ≥ 2
    mask_2b = (data["N_bjets"] >= 2) 

    def build(mask):
        out = {}
        for key in FEATURE_NAMES:
            out[key] = data[key][mask]
        out["weight"] = data["weight"][mask]
        return out

    regions["1b"] = build(mask_1b)
    regions["2b"] = build(mask_2b)

    return regions

# Apply to all datasets

tW_top_sel          = split_regions(tW_top)
tW_antitop_sel   = split_regions(tW_antitop)
ttbar_sel       = split_regions(ttbar)

#==================================================
# stacked background + combined signal + S/B ratio
#==================================================


def print_weight_sign_stats(name, weights):
    n_pos = np.sum(weights > 0)
    n_neg = np.sum(weights < 0)
    print(f"{name:8s} | +ve: {n_pos:6d} | -ve: {n_neg:6d} | total: {len(weights):6d}")

print("Weight sign statistics (after preselection for 1b region):")
print_weight_sign_stats("tW_top",     tW_top_sel["1b"]["weight"])
print_weight_sign_stats("tW_antitop", tW_antitop_sel["1b"]["weight"])
print_weight_sign_stats("ttbar",  ttbar_sel["1b"]["weight"])

print("Weight sign statistics (after preselection for 2b region):")
print_weight_sign_stats("tW_top",     tW_top_sel["2b"]["weight"])
print_weight_sign_stats("tW_antitop", tW_antitop_sel["2b"]["weight"])
print_weight_sign_stats("ttbar",  ttbar_sel["2b"]["weight"])

#%% 
# ------------------------------------------------------
# Combine tW_top and tW_antitop → single tW signal sample
# ------------------------------------------------------

tW = {}

for key in FEATURE_NAMES:
    tW[key] = np.concatenate([
        tW_top[key],
        tW_antitop[key]
    ])
for key in CUT_VARIABLES:
    tW[key] = np.concatenate([
        tW_top[key],
        tW_antitop[key]
    ])
# combine weights
tW["weight"] = np.concatenate([
    tW_top["weight"],
    tW_antitop["weight"]
])

print("\nCombined signal sample (tW):")
print(f"  Events = {len(tW['HT'])}")
print(f"  Σw(tW) = {np.sum(tW['weight']):.3e}")
#%% ======================================================
# REGION DEFINITIONS
# ======================================================

def build_region_mask(data, region):

    Nj = data["N_jets"]
    Nb = data["N_bjets"]

    if region == "1b":
        return (Nb == 1) 

    elif region == "2b":
        return (Nb >= 2) 
    else:
        raise ValueError("Unknown region")
REGIONS = ["1b","2b"]
def apply_region(data, mask):

    out = {}
    for key in data:
        out[key] = data[key][mask]
    return out
#%% ======================================================
# REGION LOOP
# ======================================================

for region in REGIONS:

    print(f"\n==============================")
    print(f"Processing Region: {region}")
    print(f"==============================")

    # masks
    mask_sig = build_region_mask(tW, region)
    mask_bkg = build_region_mask(ttbar, region)

    tW_reg = apply_region(tW, mask_sig)
    tt_reg = apply_region(ttbar, mask_bkg)

    print("Signal events:", len(tW_reg["HT"]))
    print("Weighted signal yield:",np.sum(tW_reg["weight"]))
    print("Background events:", len(tt_reg["HT"]))
    print("Weighted background yield:", np.sum(tt_reg["weight"]))
#%%
#stacked background plotting fucntions
def stacked_plot(sig_vals, sig_wts,
                 bkg_vals_list, bkg_wts_list,
                 bins, xlabel):

    # ---------- clean helper ----------
    def clean(x, w):
        x = np.asarray(x).reshape(-1)
        w = np.asarray(w).reshape(-1)
        good = np.isfinite(x) & np.isfinite(w)
        return x[good], w[good]

    sig_vals, sig_wts = clean(sig_vals, sig_wts)

    bkg_vals_clean = []
    bkg_wts_clean  = []
    for v, w in zip(bkg_vals_list, bkg_wts_list):
        vc, wc = clean(v, w)
        bkg_vals_clean.append(vc)
        bkg_wts_clean.append(wc)

    # ---------- figure ----------
    fig, (ax, rax) = plt.subplots(
        2, 1,
        figsize=(8,7),
        gridspec_kw={"height_ratios":[3,1]},
        sharex=True
    )

    # ---------- stacked backgrounds ----------
    hist_bkg, edges, _ = ax.hist(
        bkg_vals_clean,
        bins=bins,
        weights=bkg_wts_clean,
        stacked=True,
        histtype="stepfilled",
        alpha=0.65,
        label=["ttbar"]
    )

    total_bkg = np.sum(hist_bkg, axis=0)
    centers = 0.5*(edges[:-1] + edges[1:])

    # ---------- signal ----------
    sig_hist, _ = np.histogram(sig_vals, bins=bins, weights=sig_wts)
    ax.hist(
        sig_vals, bins=bins,
        weights=sig_wts,
        histtype="step",
        linewidth=3,
        color="red",
        label="tW signal"
    )

    # ---------- S/B ----------
    ratio = np.divide(
        sig_hist, total_bkg,
        out=np.zeros_like(sig_hist, dtype=float),
        where=(total_bkg > 0)
    )

    rax.step(centers, ratio, where="mid",
             color="red", linewidth=2)

    # ---------- cosmetics ----------
    ax.set_yscale("log")
    ax.set_ylabel("Events")
    ax.legend(frameon=False)
    ax.grid(alpha=0.4)

    rax.set_xlabel(xlabel)
    rax.set_ylabel("S / B")
    rax.set_ylim(0, np.max(ratio)*1.4 if np.max(ratio)>0 else 1)
    rax.grid(alpha=0.4)

    plt.tight_layout()
    plt.show()
    
#======================================================
# REGION-WISE STACKED PLOTS
# ======================================================

for region in REGIONS:

    print(f"\n--- Stacked plots: {region} ---")

    mask_sig = build_region_mask(tW, region)
    mask_bkg = build_region_mask(ttbar, region)

    tW_reg = apply_region(tW, mask_sig)
    tt_reg = apply_region(ttbar, mask_bkg)

    def get_bkg_region(key):
        return [tt_reg[key]], [tt_reg["weight"]]

    for feat in FEATURE_NAMES:

        vals, wts = get_bkg_region(feat)

        stacked_plot(
            tW_reg[feat], tW_reg["weight"],
            vals, wts,
            bins=50,
            xlabel=f"{feat} ({region})"
        )
#%%
# ======================================================
# SEPARATION POWER CALCULATION (Feature-wise)
# ======================================================

def compute_separation(x_sig, w_sig, x_bkg, w_bkg, bins=50):
    
    # --- define common range ---
    xmin = min(np.min(x_sig), np.min(x_bkg))
    xmax = max(np.max(x_sig), np.max(x_bkg))
    
    # --- histograms ---
    hist_sig, bin_edges = np.histogram(
        x_sig, bins=bins, range=(xmin, xmax), weights=w_sig
    )
    
    hist_bkg, _ = np.histogram(
        x_bkg, bins=bin_edges, weights=w_bkg
    )
    
    # --- normalize using ABS weights (important for NLO) ---
    sig_norm = np.sum(np.abs(hist_sig))
    bkg_norm = np.sum(np.abs(hist_bkg))
    
    if sig_norm == 0 or bkg_norm == 0:
        return 0.0
    
    hist_sig = hist_sig / sig_norm
    hist_bkg = hist_bkg / bkg_norm
    
    # --- compute separation ---
    mask = (hist_sig + hist_bkg) > 0
    
    sep = 0.5 * np.sum(
        (hist_sig[mask] - hist_bkg[mask])**2 /
        (hist_sig[mask] + hist_bkg[mask])
    )
    
    return sep

# ======================================================
# COMPUTE FOR ALL FEATURES
# ======================================================
for region in REGIONS:

    print(f"\n==============================")
    print(f"Processing Region: {region}")
    print(f"==============================")

    # masks
    mask_sig = build_region_mask(tW, region)
    mask_bkg = build_region_mask(ttbar, region)

    tW_reg = apply_region(tW, mask_sig)
    tt_reg = apply_region(ttbar, mask_bkg)
    print("\n=== Separation Power ===")

    separation_results = {}

    for feat in FEATURE_NAMES:

        sep = compute_separation(
            tW_reg[feat], tW_reg["weight"],
            tt_reg[feat], tt_reg["weight"],
            bins=50)

        separation_results[feat] = sep
        print(f"{feat:15s} : {sep:.4f}")
#%%
# build feature matrices

X_sig = np.column_stack([tW[f] for f in FEATURE_NAMES])
X_bkg = np.column_stack([ttbar[f] for f in FEATURE_NAMES])
# ======================================================
# BUILD REGION FEATURE (GLOBAL)
# ======================================================

def get_region_label(data):
    Nj = data["N_jets"]
    Nb = data["N_bjets"]

    region = np.full(len(Nj), -1)

    region[(Nb==1)] = 0   # 1b
    region[(Nb>=2)] = 1   # 2b
    

    return region

region_sig = get_region_label(tW)
region_bkg = get_region_label(ttbar)

region_all = np.concatenate([region_sig, region_bkg])

# append region as last feature
region_all = region_all.reshape(-1,1)

X = np.hstack([
    np.vstack([X_sig, X_bkg]),
    region_all
])

Y = np.concatenate([
    np.ones(len(X_sig)),
    np.zeros(len(X_bkg))
])

W = np.concatenate([
    tW["weight"],
    ttbar["weight"]
])

print("Dense H-feature matrix shape:", X.shape)
#%%
#------------------------------------------------------
#Building exact train-test and weight normalized arrays
#------------------------------------------------------

W = W / np.mean(W)

from sklearn.model_selection import train_test_split

Xtr, Xte, Ytr, Yte, Wtr, Wte, Rtr, Rte = train_test_split(
    X, Y, W, region_all,
    test_size=0.30,
    random_state=42,
    stratify=Y
)

Wtr = np.abs(Wtr)
Wtr /= np.mean(Wtr)
Wte = np.abs(Wte)
Wte /= np.mean(Wte)
#%%
#----------------------------
#Scaling for Quantum Training
#----------------------------

# ==========================================
# ROBUST SCALING FOR VQC (DROP-IN BLOCK)
# ==========================================

from sklearn.preprocessing import RobustScaler
import numpy as np

# ------------------------------------------
# Step 1: separate physics features (exclude region)
# ------------------------------------------
Xtr_phys = Xtr[:, :-1]
Xte_phys = Xte[:, :-1]

Rtr_feat = Xtr[:, -1].reshape(-1,1)
Rte_feat = Xte[:, -1].reshape(-1,1)

# ------------------------------------------
# Step 2: robust scaling (median + IQR)
# ------------------------------------------
robust_scaler = RobustScaler(
    with_centering=True,
    with_scaling=True,
    quantile_range=(10, 90)   # more stable than (25,75) for HEP
)

Xtr_robust = robust_scaler.fit_transform(Xtr_phys)
Xte_robust = robust_scaler.transform(Xte_phys)

# ------------------------------------------
# Step 3: compress extreme tails (important)
# ------------------------------------------
# prevents huge angles in quantum circuit

Xtr_robust = np.tanh(Xtr_robust)
Xte_robust = np.tanh(Xte_robust)

# ------------------------------------------
# Step 4: map to quantum range [0, π]
# ------------------------------------------
Xtr_scaled = (Xtr_robust + 1.0) * (np.pi / 2.0)
Xte_scaled = (Xte_robust + 1.0) * (np.pi / 2.0)

# ------------------------------------------
# Step 5: reattach region feature
# ------------------------------------------
Xtr_q = np.hstack([Xtr_scaled, Rtr_feat])
Xte_q = np.hstack([Xte_scaled, Rte_feat])

#%%
Xtr_sub = Xtr
Ytr_sub = Ytr
Wtr_sub = Wtr
#######################################
#######################################
#BDT FOR CLASSICAL MACHINE LEARNING ###
#MLP DENSE CLASSICAL NEURAL NETWORK ###
#######################################
import xgboost as xgb

bdt = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=2,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    eval_metric="logloss"
)

bdt.fit(Xtr_sub, Ytr_sub, sample_weight=Wtr_sub)

scores_bdt = bdt.predict_proba(Xte)[:,1]
scores_bdt_train = bdt.predict_proba(Xtr_sub)[:,1]

# ==========================================
# PYTORCH MLP (IMPROVED TRAINING)
# ==========================================
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# ------------------------------------------
# Feature scaling (VERY IMPORTANT)
# ------------------------------------------
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler(
    with_centering=True,
    with_scaling=True,
    quantile_range=(10, 90)
)
Xtr_sub = scaler.fit_transform(Xtr_sub)
Xte = scaler.transform(Xte)

# ------------------------------------------
# Convert to torch tensors
# ------------------------------------------
Xtr_t = torch.tensor(Xtr_sub, dtype=torch.float32)
Ytr_t = torch.tensor(Ytr_sub.reshape(-1,1), dtype=torch.float32)
Wtr_t = torch.tensor(Wtr_sub.reshape(-1,1), dtype=torch.float32)

Xte_t = torch.tensor(Xte, dtype=torch.float32)

# ------------------------------------------
# Normalize weights (stability)
# ------------------------------------------
Wtr_t = Wtr_t / torch.mean(torch.abs(Wtr_t))

# ------------------------------------------
# Model definition (improved architecture)
# ------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)

model = MLP(Xtr_sub.shape[1])

# ------------------------------------------
# Loss + optimizer
# ------------------------------------------
def weighted_bce_logits(logits, targets, weights):

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction='none'
    )

    return torch.sum(weights * loss) / torch.sum(torch.abs(weights))

optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay = 1e-3)

# ------------------------------------------
# Training loop (mini-batch)
# ------------------------------------------
epochs = 200
batch_size = 128

for epoch in range(epochs):

    model.train()

    perm = torch.randperm(len(Xtr_t))

    for i in range(0, len(Xtr_t), batch_size):

        idx = perm[i:i+batch_size]

        xb = Xtr_t[idx]
        yb = Ytr_t[idx]
        wb = Wtr_t[idx]

        optimizer.zero_grad()

        outputs = model(xb)

        loss = weighted_bce_logits(outputs, yb, wb)

        loss.backward()
        optimizer.step()

    # optional monitoring
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ------------------------------------------
# Prediction (same as sklearn output)
# ------------------------------------------
model.eval()
with torch.no_grad():
    logits = model(Xte_t)
    scores_mlp = torch.sigmoid(logits).numpy().flatten()
with torch.no_grad():
    logits_train = model(Xtr_t)
    scores_mlp_train = torch.sigmoid(logits_train).numpy().flatten()
#%%
# ==========================================
# PENNYLANE VQC WITH EVENT WEIGHTS
# ==========================================

# ======================================
# PENNYLANE VARIATIONAL QUANTUM CLASSIFIER
# ======================================

import pennylane as qml
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np


# ---------------------------
# Convert data to torch
# ---------------------------

Xtr_torch = torch.tensor(Xtr_q, dtype=torch.float32)
Xte_torch = torch.tensor(Xte_q, dtype=torch.float32)

Ytr_torch = torch.tensor(Ytr, dtype=torch.float32)
Wtr_torch = torch.tensor(Wtr, dtype=torch.float32)


# ---------------------------
# Quantum device
# ---------------------------

n_qubits = Xtr_q.shape[1]

dev = qml.device("default.qubit", wires=n_qubits)


# ---------------------------
# Quantum circuit (vectorized)
# ---------------------------

@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit(x, weights):

    # x shape: (batch_size, features)

    for i in range(n_qubits):
        qml.RY(x[:, i], wires=i)
        qml.RZ(x[:, i], wires=i)

    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# ---------------------------
# VQC model
# ---------------------------

class VQC(nn.Module):

    def __init__(self):

        super().__init__()

        self.weights = nn.Parameter(
            0.01 * torch.randn((7, n_qubits, 3))
        )

        self.readout = nn.Sequential(
            nn.Linear(n_qubits, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
            )

    def forward(self, x):

        q_out = circuit(x, self.weights)

        q_out = torch.stack(q_out, dim=1)

        # FIX: ensure same dtype as torch model
        q_out = q_out.float()

        logits = self.readout(q_out)

        return logits.squeeze()

model = VQC()


# ---------------------------
# Optimizer
# ---------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay = 5*1e-4)


# ---------------------------
# Weighted BCE loss
# ---------------------------

def weighted_bce(logits, targets, weights):

    probs = torch.sigmoid(logits)

    loss = -(targets * torch.log(probs + 1e-8) +
             (1 - targets) * torch.log(1 - probs + 1e-8))

    return torch.sum(weights * loss)/torch.sum(torch.abs(weights))


# ---------------------------
# DataLoader (mini-batch)
# ---------------------------

dataset = TensorDataset(Xtr_torch, Ytr_torch, Wtr_torch)

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True
)

#%%
# ---------------------------
# Training
# ---------------------------

epochs = 200

for epoch in tqdm(range(epochs), desc="VQC training"):

    epoch_loss = 0

    for xb, yb, wb in loader:

        optimizer.zero_grad()

        logits = model(xb)

        loss = weighted_bce(logits, yb, wb)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch} loss = {epoch_loss:.4f}")


# ---------------------------
# Inference
# ---------------------------

with torch.no_grad():

    logits = model(Xte_torch)

    scores_vqc = torch.sigmoid(logits).numpy()
with torch.no_grad():
    logits_train = model(Xtr_torch)
    scores_vqc_train = torch.sigmoid(logits_train).numpy()

# --------------------------------
# save trained weights
# --------------------------------

weights_np = model.weights.detach().numpy()
SAVE_PATH = "D:/cern/ML/complex variable/"
np.save(
    SAVE_PATH + "pennylane_vqc_weights_hadronic_resolved.npy",
    weights_np
)

print("Saved → pennylane_vqc_weights.npy")


# --------------------------------
# save scores
# --------------------------------

np.save(
    SAVE_PATH + "scores_vqc_pennylane_hadronic_resolved.npy",
    scores_vqc
)
np.save( SAVE_PATH + "scores_vqc_pennylane_hadronic_resolved_train.npy", scores_vqc_train)
print("Saved → scores_vqc_pennylane.npy")


#%%
#Bootstrap function
from sklearn.metrics import roc_auc_score

def bootstrap_auc(scores, Y, W, n_boot=500):

    aucs = []
    N = len(scores)

    for _ in range(n_boot):

        idx = np.random.randint(0, N, N)

        auc_val = roc_auc_score(
            Y[idx],
            scores[idx],
            sample_weight=W[idx]
        )

        aucs.append(auc_val)

    aucs = np.array(aucs)

    return np.mean(aucs), np.std(aucs)

def region_bootstrap_auc(scores, Y, W, region):

    REGION_LABELS = {0:"1b", 1:"2b"}

    for r in [0,1]:

        mask = (region == r)

        if np.sum(mask) < 20:
            continue

        mean, err = bootstrap_auc(
            scores[mask],
            Y[mask],
            W[mask]
        )

        print(f"{REGION_LABELS[r]} : {mean:.4f} ± {err:.4f}")
#%%

# ======================================================
# REGION-WISE EVALUATION (CONDITIONAL MODEL)
# ======================================================
SAVE_PATH = "D:/cern/ML/complex variable/"
scores_vqc = np.load( SAVE_PATH +"scores_vqc_pennylane_hadronic_resolved.npy")
from sklearn.metrics import roc_auc_score

REGION_LABELS = {
    0: "1b",
    1: "2b"
}

def evaluate_per_region(scores, Y, W, region):

    for r in [0,1]:

        mask = (region == r)

        if np.sum(mask) < 20:
            continue

        auc_val = roc_auc_score(
            Y[mask],
            scores[mask],
            sample_weight=W[mask]
        )

        print(f"{REGION_LABELS[r]} AUC = {auc_val:.4f}")

print("\n=== BDT REGION PERFORMANCE ===")
evaluate_per_region(scores_bdt, Yte, Wte, Rte.flatten())

print("\n=== MLP REGION PERFORMANCE ===")
evaluate_per_region(scores_mlp, Yte, Wte, Rte.flatten())

print("\n=== VQC REGION PERFORMANCE ===")
evaluate_per_region(scores_vqc, Yte, Wte, Rte.flatten())

print("\n=== BDT REGION AUC (bootstrap) ===")
region_bootstrap_auc(scores_bdt, Yte, Wte, Rte.flatten())

print("\n=== MLP REGION AUC (bootstrap) ===")
region_bootstrap_auc(scores_mlp, Yte, Wte, Rte.flatten())

print("\n=== VQC REGION AUC (bootstrap) ===")
region_bootstrap_auc(scores_vqc, Yte, Wte, Rte.flatten())

#%%
#Global AUC
auc_bdt = roc_auc_score(Yte, scores_bdt, sample_weight=Wte)
auc_mlp = roc_auc_score(Yte, scores_mlp, sample_weight=Wte)
auc_vqc = roc_auc_score(Yte, scores_vqc, sample_weight=Wte)

mean_bdt, err_bdt = bootstrap_auc(scores_bdt, Yte, Wte)
mean_mlp, err_mlp = bootstrap_auc(scores_mlp, Yte, Wte)
mean_vqc, err_vqc = bootstrap_auc(scores_vqc, Yte, Wte)

print("\n=== GLOBAL AUC (with uncertainty) ===")

print(f"BDT : {mean_bdt:.4f} ± {err_bdt:.4f}")
print(f"MLP : {mean_mlp:.4f} ± {err_mlp:.4f}")
print(f"VQC : {mean_vqc:.4f} ± {err_vqc:.4f}")
#%%
def count_events_per_region(Y, W, R):

    REGION_LABELS = {0:"1b", 1:"2b"}

    print("\n=== TEST SET EVENT COUNTS ===\n")

    for r in [0,1]:

        mask = (R == r)

        n_total = np.sum(mask)
        n_sig   = np.sum((Y == 1) & mask)
        n_bkg   = np.sum((Y == 0) & mask)

        w_sig = np.sum(W[(Y == 1) & mask])
        w_bkg = np.sum(W[(Y == 0) & mask])

        print(f"{REGION_LABELS[r]}:")
        print(f"  Events (total) = {n_total}")
        print(f"  Signal events  = {n_sig}")
        print(f"  Background     = {n_bkg}")
        print(f"  Σw(signal)     = {w_sig:.3f}")
        print(f"  Σw(background) = {w_bkg:.3f}")
        print()
count_events_per_region(Yte, Wte, Rte.flatten())
#%%
def bootstrap_metric(metric_func, scores, Y, W, n_boot=300):

    vals = []
    N = len(scores)

    for _ in range(n_boot):

        idx = np.random.randint(0, N, N)

        val = metric_func(
            scores[idx],
            Y[idx],
            W[idx]
        )

        vals.append(val)

    vals = np.array(vals)

    return np.mean(vals), np.std(vals)

def region_bootstrap(metric_func, scores, Y, W, region):

    REGION_LABELS = {0:"1b", 1:"2b"}

    for r in [0,1]:

        mask = (region == r)

        if np.sum(mask) < 20:
            continue

        mean, err = bootstrap_metric(
            metric_func,
            scores[mask],
            Y[mask],
            W[mask]
        )

        print(f"{REGION_LABELS[r]} : {mean:.4f} ± {err:.4f}")
#%%
# ======================================
# GLOBAL SEPARATION (BDT / MLP / VQC)
# ======================================

def separation(scores, Y, W, bins=50):

    sig = (Y == 1)
    bkg = (Y == 0)

    s_hist, edges = np.histogram(scores[sig], bins=bins,
                                 weights=W[sig])
    b_hist, _     = np.histogram(scores[bkg], bins=edges,
                                 weights=W[bkg])

    # normalize with ABS weights (shape only)
    s_norm = np.sum(np.abs(s_hist))
    b_norm = np.sum(np.abs(b_hist))

    if s_norm == 0 or b_norm == 0:
        return 0

    s_hist = s_hist / s_norm
    b_hist = b_hist / b_norm

    mask = (s_hist + b_hist) > 0

    sep = 0.5 * np.sum(
        (s_hist[mask] - b_hist[mask])**2 /
        (s_hist[mask] + b_hist[mask])
    )

    return sep


print("\n=== SEPARATION POWER ===")

sep_bdt, err_sep_bdt = bootstrap_metric(separation, scores_bdt, Yte, Wte)
sep_mlp, err_sep_mlp = bootstrap_metric(separation, scores_mlp, Yte, Wte)
sep_vqc, err_sep_vqc = bootstrap_metric(separation, scores_vqc, Yte, Wte)

print(f"BDT : {sep_bdt:.4f} ± {err_sep_bdt:.4f}")
print(f"MLP : {sep_mlp:.4f} ± {err_sep_mlp:.4f}")
print(f"VQC : {sep_vqc:.4f} ± {err_sep_vqc:.4f}")

# ======================================
# MONOTONICITY TEST (Purity vs Score)
# ======================================

def monotonicity_curve(scores, Y, W, bins=20):

    edges = np.linspace(scores.min(), scores.max(), bins+1)

    centers = []
    purity  = []

    for i in range(bins):

        mask = (scores >= edges[i]) & (scores < edges[i+1])

        if np.sum(mask) < 20:
            continue

        S = np.sum(W[(Y==1) & mask])
        B = np.sum(W[(Y==0) & mask])

        if (S + B) <= 0:
            continue

        centers.append(0.5*(edges[i] + edges[i+1]))
        purity.append(S / (S + B))

    return np.array(centers), np.array(purity)


# compute curves
c_bdt, p_bdt = monotonicity_curve(scores_bdt, Yte, Wte)
c_mlp, p_mlp = monotonicity_curve(scores_mlp, Yte, Wte)
c_vqc, p_vqc = monotonicity_curve(scores_vqc, Yte, Wte)

# plot
plt.figure(figsize=(7,6))

plt.plot(c_bdt, p_bdt, marker='o', label="BDT")
plt.plot(c_mlp, p_mlp, marker='o', label="MLP")
plt.plot(c_vqc, p_vqc, marker='o', label="VQC")

plt.xlabel("Classifier score")
plt.ylabel("Signal purity S / (S + B)")
plt.title("Monotonicity check")
plt.legend()
plt.grid(alpha=0.3)

plt.show()

# ======================================
# SIGNIFICANCE CURVES (S / sqrt(B))
# ======================================

def significance_curve(scores, Y, W):

    thresholds = np.linspace(scores.min(), scores.max(), 200)

    Z = []

    for t in thresholds:

        mask = scores > t

        S = np.sum(W[(Y==1) & mask])
        B = np.sum(W[(Y==0) & mask])

        if B > 0:
            Z.append(S / np.sqrt(B))
        else:
            Z.append(0)

    return thresholds, np.array(Z)

def significance_metric(scores, Y, W):
    _, Z = significance_curve(scores, Y, W)
    return np.max(Z)

# compute curves
thr_bdt, Z_bdt = significance_curve(scores_bdt, Yte, Wte)
thr_mlp, Z_mlp = significance_curve(scores_mlp, Yte, Wte)
thr_vqc, Z_vqc = significance_curve(scores_vqc, Yte, Wte)

Z_m_bdt, err_z_bdt = bootstrap_metric(significance_metric, scores_bdt, Yte, Wte)
Z_m_mlp, err_z_mlp = bootstrap_metric(significance_metric, scores_mlp, Yte, Wte)
Z_m_vqc, err_z_vqc = bootstrap_metric(significance_metric, scores_vqc, Yte, Wte)
# plot
plt.figure(figsize=(7,6))

plt.plot(thr_bdt, Z_bdt, label="BDT")
plt.plot(thr_mlp, Z_mlp, label="MLP")
plt.plot(thr_vqc, Z_vqc, label="VQC")

plt.xlabel("Score threshold")
plt.ylabel("S / sqrt(B)")
plt.title("Significance comparison")

plt.legend()
plt.grid(alpha=0.3)

plt.show()


# print best significance
print("\n=== MAX SIGNIFICANCE ===")
print(f"BDT : {Z_m_bdt:.4f} ± {err_z_bdt:.4f}")
print(f"MLP : {Z_m_mlp:.4f} ± {err_z_mlp:.4f}")
print(f"VQC : {Z_m_vqc:.4f} ± {err_z_vqc:.4f}")
#%%
#Likelihood 
def to_likelihood_ratio(scores, eps=1e-6):
    scores = np.clip(scores, eps, 1 - eps)
    return (scores / (1 - scores))

lr_bdt = to_likelihood_ratio(scores_bdt)
lr_mlp = to_likelihood_ratio(scores_mlp)
lr_vqc = to_likelihood_ratio(scores_vqc)

def likelihood_plot_region(lr, Y, W, region, model_name):

    REGION_LABELS = {0:"1b", 1:"2b"}

    for r in [0,1]:

        mask = (region == r)

        sig = (Y==1) & mask
        bkg = (Y==0) & mask

        plt.figure(figsize=(6,5))

        plt.hist(lr[sig], bins=50, weights=W[sig],
                 histtype="step", linewidth=2, label="tW")

        plt.hist(lr[bkg], bins=50, weights=W[bkg],
                 histtype="step", linestyle="--", linewidth=2, label="ttbar")

        plt.xlabel("Likelihood Ratio (S/B)")
        plt.ylabel("Weighted events")
        plt.title(f"{model_name} ({REGION_LABELS[r]})")
        plt.yscale('log')

        plt.legend()
        plt.grid(alpha=0.3)

        plt.show()

likelihood_plot_region(lr_bdt, Yte, Wte, Rte.flatten(), "BDT")
likelihood_plot_region(lr_mlp, Yte, Wte, Rte.flatten(), "MLP")
likelihood_plot_region(lr_vqc, Yte, Wte, Rte.flatten(), "VQC")

def compare_models_region(lr_dict, Y, W, region):

    REGION_LABELS = {0:"1b", 1:"2b"}

    for r in [0,1]:

        mask = (region == r)

        plt.figure(figsize=(6,5))

        for name, lr in lr_dict.items():

            sig = (Y==1) & mask

            plt.hist(lr[sig], bins=50, weights=W[sig],
                     histtype="step", linewidth=2, label=f"{name} (signal)")

        plt.xlabel("Likelihood Ratio")
        plt.ylabel("Events")
        plt.title(f"Signal comparison ({REGION_LABELS[r]})")
        plt.yscale('log')
        

        plt.legend()
        plt.grid(alpha=0.3)

        plt.show()

lr_dict = {
    "BDT": lr_bdt,
    "MLP": lr_mlp,
    "VQC": lr_vqc
}

compare_models_region(
    lr_dict,
    Yte,
    Wte,
    Rte.flatten()
)
#%%
from sklearn.metrics import roc_auc_score
scores_vqc_train = np.load(SAVE_PATH + "scores_vqc_pennylane_hadronic_resolved_train.npy")

mean_bdt_train, err_bdt_train = bootstrap_auc(scores_bdt_train, Ytr, Wtr)
mean_mlp_train, err_mlp_train = bootstrap_auc(scores_mlp_train, Ytr, Wtr)
mean_vqc_train, err_vqc_train = bootstrap_auc(scores_vqc_train, Ytr, Wtr)

print("\n=== TRAIN vs TEST AUC ===")

print(f"BDT : Train = {mean_bdt_train:.4f} ± {err_bdt_train:.4f} | Test = {mean_bdt:.4f} ± {err_bdt:.4f}")
print(f"MLP : Train = {mean_mlp_train:.4f} ± {err_mlp_train:.4f}| Test = {mean_mlp:.4f} ± {err_mlp:.4f}")
print(f"VQC : Train = {mean_vqc_train:.4f} ± {err_vqc_train:.4f} | Test = {mean_vqc:.4f} ± {err_vqc:.4f}")

#%%
# ======================================
# REGION-WISE SEPARATION (BDT / MLP / VQC)
# ======================================

def separation(scores, Y, W, bins=50):

    sig = (Y == 1)
    bkg = (Y == 0)

    s_hist, edges = np.histogram(scores[sig], bins=bins,
                                 weights=W[sig])
    b_hist, _     = np.histogram(scores[bkg], bins=edges,
                                 weights=W[bkg])

    # normalize with ABS weights
    s_norm = np.sum(np.abs(s_hist))
    b_norm = np.sum(np.abs(b_hist))

    if s_norm == 0 or b_norm == 0:
        return 0

    s_hist = s_hist / s_norm
    b_hist = b_hist / b_norm

    mask = (s_hist + b_hist) > 0

    sep = 0.5 * np.sum(
        (s_hist[mask] - b_hist[mask])**2 /
        (s_hist[mask] + b_hist[mask])
    )

    return sep


def separation_per_region(scores, Y, W, region):

    REGION_LABELS = {0:"1b", 1:"2b"}

    print("\n=== REGION-WISE SEPARATION ===")

    for r in [0,1]:

        mask = (region == r)

        if np.sum(mask) < 20:
            continue

        sep_val = separation(
            scores[mask],
            Y[mask],
            W[mask]
        )

        return sep_val

# ======================================
# REGION-WISE SIGNIFICANCE
# ======================================

def significance_curve(scores, Y, W):

    thresholds = np.linspace(scores.min(), scores.max(), 200)

    Z = []

    for t in thresholds:

        mask = scores > t

        S = np.sum(W[(Y==1) & mask])
        B = np.sum(W[(Y==0) & mask])

        if B > 0:
            Z.append(S / np.sqrt(B))
        else:
            Z.append(0)

    return thresholds, np.array(Z)


def significance_per_region(scores, Y, W, region):

    REGION_LABELS = {0:"1b", 1:"2b"}

    print("\n=== REGION-WISE MAX SIGNIFICANCE ===")

    for r in [0,1]:

        mask = (region == r)

        if np.sum(mask) < 20:
            continue

        _, Z = significance_curve(
            scores[mask],
            Y[mask],
            W[mask]
        )

        return np.max(Z)
def auc_metric(scores, Y, W):
    return roc_auc_score(Y, scores, sample_weight=W)

def separation_metric(scores, Y, W):
    return separation(scores, Y, W)

def significance_metric(scores, Y, W):
    _, Z = significance_curve(scores, Y, W)
    return np.max(Z)
def summary_table(scores_dict, Y, W, region):

    REGION_LABELS = {0:"1b", 1:"2b"}

    print("\n=== SUMMARY TABLE ===\n")

    for r in [0,1]:

        mask = (region == r)

        if np.sum(mask) < 20:
            continue

        print(f"\n--- {REGION_LABELS[r]} ---")

        for name, scores in scores_dict.items():

            # --- AUC ---
            auc_mean, auc_err = bootstrap_metric(
                auc_metric,
                scores[mask], Y[mask], W[mask]
            )

            # --- Separation ---
            sep_mean, sep_err = bootstrap_metric(
                separation_metric,
                scores[mask], Y[mask], W[mask]
            )

            # --- Significance ---
            Z_mean, Z_err = bootstrap_metric(
                significance_metric,
                scores[mask], Y[mask], W[mask]
            )

            print(f"{name:5s} | "
                  f"AUC = {auc_mean:.4f} ± {auc_err:.4f} | "
                  f"Sep = {sep_mean:.4f} ± {sep_err:.4f} | "
                  f"Z = {Z_mean:.4f} ± {Z_err:.4f}")
scores_dict = {
    "BDT": scores_bdt,
    "MLP": scores_mlp,
    "VQC": scores_vqc
}

summary_table(scores_dict, Yte, Wte, Rte.flatten())
#%%
#Model Evaluation
# ======================================
# ROC curves (BDT / MLP / VQC)
# ======================================

from sklearn.metrics import roc_curve, auc

fpr_bdt, tpr_bdt, _ = roc_curve(Yte, scores_bdt, sample_weight=Wte)
fpr_mlp, tpr_mlp, _ = roc_curve(Yte, scores_mlp, sample_weight=Wte)
fpr_vqc, tpr_vqc, _ = roc_curve(Yte, scores_vqc, sample_weight=Wte)

auc_bdt = auc(fpr_bdt, tpr_bdt)
auc_mlp = auc(fpr_mlp, tpr_mlp)
auc_vqc = auc(fpr_vqc, tpr_vqc)

plt.figure(figsize=(12,5))

# --- ROC ---
plt.subplot(1,2,1)
plt.plot(fpr_bdt, tpr_bdt, label=f"BDT ({auc_bdt:.3f})")
plt.plot(fpr_mlp, tpr_mlp, label=f"MLP ({auc_mlp:.3f})")
plt.plot(fpr_vqc, tpr_vqc, label=f"VQC ({auc_vqc:.3f})")
plt.plot([0,1], [0,1], 'k--', alpha=0.5)
plt.xlabel("Background efficiency")
plt.ylabel("Signal efficiency")
plt.title("ROC Curve")
plt.legend()
plt.grid(alpha=0.3)

# --- Efficiency vs Rejection ---
plt.subplot(1,2,2)
plt.plot(tpr_bdt, 1-fpr_bdt, label="BDT")
plt.plot(tpr_mlp, 1-fpr_mlp, label="MLP")
plt.plot(tpr_vqc, 1-fpr_vqc, label="VQC")
plt.xlabel("Signal efficiency")
plt.ylabel("Background rejection")
plt.title("Efficiency vs Rejection")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

def plot_eff_rej_region(scores_dict, Y, W, region):

    REGION_LABELS = {0:"1b", 1:"2b"}

    for r in [0,1]:

        mask = (region == r)

        if np.sum(mask) < 20:
            continue

        plt.figure(figsize=(7,6))

        for name, scores in scores_dict.items():

            s = scores[mask]
            y = Y[mask]
            w = W[mask]

            fpr, tpr, _ = roc_curve(y, s, sample_weight=w)

            plt.plot(tpr, 1 - fpr, label=name)

        plt.xlabel("Signal efficiency")
        plt.ylabel("Background rejection")
        plt.title(f"Efficiency vs Rejection ({REGION_LABELS[r]})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

plot_eff_rej_region(scores_dict, Yte, Wte, Rte.flatten())

def plot_significance_region(scores_dict, Y, W, region):

    REGION_LABELS = {0:"1b", 1:"2b"}

    for r in [0,1]:

        mask = (region == r)

        if np.sum(mask) < 20:
            continue

        plt.figure(figsize=(7,6))

        for name, scores in scores_dict.items():

            s = scores[mask]
            y = Y[mask]
            w = W[mask]

            thresholds = np.linspace(s.min(), s.max(), 200)
            Z_vals = []

            for t in thresholds:

                sel = s > t

                S = np.sum(w[(y==1) & sel])
                B = np.sum(w[(y==0) & sel])

                if B > 0:
                    Z_vals.append(S / np.sqrt(B))
                else:
                    Z_vals.append(0)
            
            Z_vals = np.array(Z_vals)
            imax = np.argmax(Z_vals)
            plt.plot(thresholds, Z_vals, label=name)
            plt.scatter(thresholds[imax], Z_vals[imax], s=40)
            

        plt.xlabel("Score threshold")
        plt.ylabel("Significance (S / √B)")
        plt.title(f"Significance vs Threshold ({REGION_LABELS[r]})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

plot_significance_region(scores_dict, Yte, Wte, Rte.flatten())
#%%