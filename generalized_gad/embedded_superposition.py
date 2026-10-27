import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import concurrent.futures
import matplotlib.pyplot as plt
from scipy.linalg import pinv
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def generate_embedded_data(n_samples, d, D_in, prob_active=0.05):
    """
    Returns both the observable high-dimensional data and the projection matrix
    used to embed the latent features, allowing us to track structural interference.
    """
    mask = (torch.rand(n_samples, d) < prob_active).float()
    vals = torch.rand(n_samples, d)
    X_latent = mask * vals
    
    projection_matrix = torch.randn(d, D_in) / np.sqrt(d)
    X_embedded = X_latent @ projection_matrix
    X_embedded += torch.randn(n_samples, D_in) * 0.01
    
    return X_embedded, projection_matrix

def train_tms_autoencoder(X, N_capacity, epochs=2500, lr=0.01):
    torch.set_num_threads(1)
    D_in = X.shape[1]
    
    W = nn.Parameter(torch.randn(N_capacity, D_in) / np.sqrt(N_capacity))
    b1 = nn.Parameter(torch.zeros(N_capacity) - 0.1) 
    b2 = nn.Parameter(torch.zeros(D_in))
    
    optimizer = optim.Adam([W, b1, b2], lr=lr)
    
    for _ in range(max(1, min(epochs, int(os.getenv("GAD_EPOCHS", str(epochs)))))):
        optimizer.zero_grad()
        h = torch.relu(X @ W.T + b1)
        X_hat = h @ W + b2
        loss = torch.nn.functional.mse_loss(X_hat, X)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            W.data = W.data / torch.norm(W.data, dim=0, keepdim=True)
            
    return W.detach(), b1.detach()

def off_diagonal_frobenius(matrix):
    """Calculates the Frobenius norm of only the off-diagonal elements."""
    off_diag = matrix.clone()
    off_diag.fill_diagonal_(0)
    return torch.norm(off_diag, p='fro').item()

def experiment_superposition_mechanics(d, N=100, D_in=500, prob_active=0.05, seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    X_embedded, P = generate_embedded_data(n_samples=N, d=d, D_in=D_in, prob_active=prob_active)
    
    W_learned, b1_learned = train_tms_autoencoder(X_embedded, N_capacity=N)
    H_learned = torch.relu(X_embedded @ W_learned.T + b1_learned)
    
    # 1. Structural Interference: Project weights back to the d latent features
    # W is (N, D_in) and P is (d, D_in). Effective latent weights: W_latent = W @ P.T
    W_latent = W_learned @ P.T
    
    # Normalize latent weights to isolate cosine similarity (interference)
    W_latent = W_latent / torch.norm(W_latent, dim=0, keepdim=True)
    Gram_struct = W_latent.T @ W_latent
    struct_interference = off_diagonal_frobenius(Gram_struct)
    
    # 2. Active Representation Interference: Conjugate Kernel geometry
    H_normalized = H_learned / (torch.norm(H_learned, dim=0, keepdim=True) + 1e-8)
    Gram_active = H_normalized.T @ H_normalized
    active_interference = off_diagonal_frobenius(Gram_active)
    
    # 3. Aliasing Operator Norm
    try:
        aliasing_norm = np.linalg.norm(pinv(H_learned.numpy(), atol=1e-6), ord=2)
    except np.linalg.LinAlgError:
        aliasing_norm = np.nan

    return {
        'd': d,
        'struct_int': struct_interference,
        'active_int': active_interference,
        'aliasing_norm': aliasing_norm
    }

def run_mechanics_sweep(d_values, N=100, D_in=500, trials_per_d=max(1, min(10, int(os.getenv("GAD_TRIALS", "10"))))):
    results = {d: {'struct': [], 'active': [], 'alias': []} for d in d_values}
    tasks = [(d, N, D_in, 0.05, d * 1000 + t) for d in d_values for t in range(trials_per_d)]
            
    print(f"Running {len(tasks)} parallel experiments...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, min(28, int(os.getenv("GAD_MAX_WORKERS", "28"))))) as executor:
        futures = [executor.submit(experiment_superposition_mechanics, *task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            d = res['d']
            results[d]['struct'].append(res['struct_int'])
            results[d]['active'].append(res['active_int'])
            results[d]['alias'].append(res['aliasing_norm'])
            
    d_plot = sorted(d_values)
    return (
        d_plot, 
        [np.nanmean(results[d]['struct']) for d in d_plot],
        [np.nanmean(results[d]['active']) for d in d_plot],
        [np.nanmean(results[d]['alias']) for d in d_plot]
    )

if __name__ == '__main__':
    N_datapoints = 100
    D_in_ambient = 500
    d_sweep = list(range(10, 210, 10))[::max(1, int(os.getenv("GAD_GRID_STRIDE", "1")))]
    
    d_plot, struct_int, active_int, alias_norm = run_mechanics_sweep(
        d_sweep, N=N_datapoints, D_in=D_in_ambient, trials_per_d=max(1, min(15, int(os.getenv("GAD_TRIALS", "15"))))
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(d_plot, struct_int, label='Structural Interference ($W_{latent}^T W_{latent}$)', color='red', linestyle='--')
    ax1.plot(d_plot, active_int, label='Active Representation Interference ($H^T H$)', color='blue', linewidth=2.5)
    ax1.axvline(x=N_datapoints, color='black', linestyle=':', label=f'Capacity Limit ($N={N_datapoints}$)')
    ax1.set_title("Mechanism: Weight Geometry vs Active Geometry")
    ax1.set_xlabel("True Generating Features ($d$)")
    ax1.set_ylabel("Off-Diagonal Frobenius Norm")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(d_plot, alias_norm, label='Aliasing Operator Norm', color='purple', linewidth=2.5)
    ax2.axvline(x=N_datapoints, color='black', linestyle=':', label=f'Capacity Limit ($N={N_datapoints}$)')
    ax2.set_yscale('log')
    ax2.set_title("Result: Aliasing Mitigation")
    ax2.set_xlabel("True Generating Features ($d$)")
    ax2.set_ylabel("$||H^+||_2$ [Log Scale]")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("superposition_mechanics.png", dpi=300)
    print("Experiment complete.")
