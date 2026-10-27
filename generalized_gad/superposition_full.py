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
    """Generates d sparse latent features embedded into a D_in ambient space."""
    mask = (torch.rand(n_samples, d) < prob_active).float()
    vals = torch.rand(n_samples, d)
    X_latent = mask * vals
    
    projection_matrix = torch.randn(d, D_in) / np.sqrt(d)
    X_embedded = X_latent @ projection_matrix
    X_embedded += torch.randn(n_samples, D_in) * 0.01
    
    return X_embedded, projection_matrix

def train_tms_autoencoder(X, N_capacity, epochs=2500, lr=0.01, reg_type='none'):
    """
    Trains the autoencoder with selectable regularization: 'none', 'l2', or 'l1'.
    """
    torch.set_num_threads(1)
    D_in = X.shape[1]
    
    W = nn.Parameter(torch.randn(N_capacity, D_in) / np.sqrt(N_capacity))
    b1 = nn.Parameter(torch.zeros(N_capacity) - 0.1) 
    b2 = nn.Parameter(torch.zeros(D_in))
    
    # Configure optimizer based on regularization type
    wd = 1e-2 if reg_type == 'l2' else 0.0
    l1_coeff = 1e-3 if reg_type == 'l1' else 0.0
    
    optimizer = optim.AdamW([W, b1, b2], lr=lr, weight_decay=wd)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        h = torch.relu(X @ W.T + b1)
        X_hat = h @ W + b2
        
        loss = torch.nn.functional.mse_loss(X_hat, X)
        
        # Add L1 penalty to activations if requested
        if reg_type == 'l1':
            loss += l1_coeff * torch.mean(torch.abs(h))
            
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            W.data = W.data / torch.norm(W.data, dim=0, keepdim=True)
            
    return W.detach(), b1.detach()

def run_regularization_experiment(d, N=100, D_in=500, prob_active=0.05, seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    X_embedded, _ = generate_embedded_data(n_samples=N, d=d, D_in=D_in, prob_active=prob_active)
    
    # 1. Random ReLU Baseline
    W_rand = torch.randn(N, D_in) / np.sqrt(N)
    W_rand = W_rand / torch.norm(W_rand, dim=0, keepdim=True)
    b1_rand = torch.zeros(N) - 0.1
    H_rand = torch.relu(X_embedded @ W_rand.T + b1_rand).numpy()
    
    # 2. Learned: No Regularization
    W_unreg, b1_unreg = train_tms_autoencoder(X_embedded, N_capacity=N, reg_type='none')
    H_unreg = torch.relu(X_embedded @ W_unreg.T + b1_unreg).numpy()
    
    # 3. Learned: L2 Regularization (AdamW)
    W_l2, b1_l2 = train_tms_autoencoder(X_embedded, N_capacity=N, reg_type='l2')
    H_l2 = torch.relu(X_embedded @ W_l2.T + b1_l2).numpy()
    
    # 4. Learned: L1 Regularization (Activation Sparsity)
    W_l1, b1_l1 = train_tms_autoencoder(X_embedded, N_capacity=N, reg_type='l1')
    H_l1 = torch.relu(X_embedded @ W_l1.T + b1_l1).numpy()
    
    def get_pinv_norm(H_matrix):
        try:
            return np.linalg.norm(pinv(H_matrix, atol=1e-5), ord=2)
        except np.linalg.LinAlgError:
            return np.nan

    return {
        'd': d,
        'alias_rand': get_pinv_norm(H_rand),
        'alias_unreg': get_pinv_norm(H_unreg),
        'alias_l2': get_pinv_norm(H_l2),
        'alias_l1': get_pinv_norm(H_l1)
    }

def run_smooth_sweep(d_values, N=100, D_in=500, trials_per_d=20):
    results = {d: {'rand': [], 'unreg': [], 'l2': [], 'l1': []} for d in d_values}
    tasks = [(d, N, D_in, 0.05, d * 1000 + t) for d in d_values for t in range(trials_per_d)]
            
    print(f"Running {len(tasks)} parallel experiments (Averaging {trials_per_d} seeds per d)...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=28) as executor:
        futures = [executor.submit(run_regularization_experiment, *task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            d = res['d']
            results[d]['rand'].append(res['alias_rand'])
            results[d]['unreg'].append(res['alias_unreg'])
            results[d]['l2'].append(res['alias_l2'])
            results[d]['l1'].append(res['alias_l1'])
            
    d_plot = sorted(d_values)
    return (
        d_plot, 
        [np.nanmean(results[d]['rand']) for d in d_plot],
        [np.nanmean(results[d]['unreg']) for d in d_plot],
        [np.nanmean(results[d]['l2']) for d in d_plot],
        [np.nanmean(results[d]['l1']) for d in d_plot]
    )

if __name__ == '__main__':
    N_capacity = 100
    D_in_ambient = 500
    d_sweep = list(range(10, 210, 10))
    
    d_plot, alias_r, alias_unreg, alias_l2, alias_l1 = run_smooth_sweep(
        d_sweep, N=N_capacity, D_in=D_in_ambient, trials_per_d=20
    )
    
    plt.figure(figsize=(12, 7))
    
    plt.plot(d_plot, alias_r, label='Random ReLU (Baseline)', color='gray', linestyle='-.')
    plt.plot(d_plot, alias_unreg, label='Learned ReLU (No Reg)', color='red', linewidth=2)
    plt.plot(d_plot, alias_l2, label='Learned ReLU (L2 Weight Decay)', color='orange', linewidth=2)
    plt.plot(d_plot, alias_l1, label='Learned ReLU (L1 Activation Sparsity)', color='blue', linewidth=2.5)
    
    plt.axvline(x=N_capacity, color='black', linestyle=':', label=f'Capacity Limit ($N={N_capacity}$)')
    
    plt.yscale('log')
    plt.title("Aliasing Mitigation: Impact of Regularization Strategies")
    plt.xlabel("True Generating Features ($d$)")
    plt.ylabel("Aliasing Operator Norm ($||H^+||_2$) [Log Scale]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("regularization_comparison_aliasing.png", dpi=300)
    print("Experiment complete. Results saved to regularization_comparison_aliasing.png")
