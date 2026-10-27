import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import concurrent.futures
import matplotlib.pyplot as plt
import itertools
from scipy.linalg import pinv
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def generate_embedded_data(n_samples, d, D_in, prob_active=0.05):
    mask = (torch.rand(n_samples, d) < prob_active).float()
    vals = torch.rand(n_samples, d)
    X_latent = mask * vals
    
    projection_matrix = torch.randn(d, D_in) / np.sqrt(d)
    X_embedded = X_latent @ projection_matrix
    X_embedded += torch.randn(n_samples, D_in) * 0.01
    
    return X_embedded

def train_tms_autoencoder(X, N_capacity, epochs=2000, lr=0.01, wd=1e-2):
    torch.set_num_threads(1)
    D_in = X.shape[1]
    
    W = nn.Parameter(torch.randn(N_capacity, D_in) / np.sqrt(N_capacity))
    b1 = nn.Parameter(torch.zeros(N_capacity) - 0.1) 
    b2 = nn.Parameter(torch.zeros(D_in))
    
    optimizer = optim.AdamW([W, b1, b2], lr=lr, weight_decay=wd)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        h = torch.relu(X @ W.T + b1)
        X_hat = h @ W + b2
        loss = torch.nn.functional.mse_loss(X_hat, X)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            W.data = W.data / torch.norm(W.data, dim=0, keepdim=True)
            
    return W.detach(), b1.detach()

def compute_grid_point(N_cap, d, n_samples=100, D_in=500, prob_active=0.05, seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    X_embedded = generate_embedded_data(n_samples, d, D_in, prob_active)
    
    # Random ReLU
    W_rand = torch.randn(N_cap, D_in) / np.sqrt(N_cap)
    W_rand = W_rand / torch.norm(W_rand, dim=0, keepdim=True)
    b1_rand = torch.zeros(N_cap) - 0.1
    H_rand = torch.relu(X_embedded @ W_rand.T + b1_rand).numpy()
    
    # Learned ReLU (AdamW)
    W_learned, b1_learned = train_tms_autoencoder(X_embedded, N_capacity=N_cap)
    H_learned = torch.relu(X_embedded @ W_learned.T + b1_learned).numpy()
    
    def get_pinv_norm(H):
        try:
            return np.linalg.norm(pinv(H, atol=1e-5), ord=2)
        except np.linalg.LinAlgError:
            return np.nan

    return {
        'N': N_cap,
        'd': d,
        'alias_rand': get_pinv_norm(H_rand),
        'alias_learned': get_pinv_norm(H_learned)
    }

def run_2d_sweep(N_values, d_values, n_samples=100, D_in=500, trials=5):
    tasks = [(N_cap, d, n_samples, D_in, 0.05, hash((N_cap, d, t)) % (2**32)) 
             for N_cap, d in itertools.product(N_values, d_values) for t in range(trials)]
    
    print(f"Running {len(tasks)} grid computations across 100 workers...")
    
    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(compute_grid_point, *task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            N_cap, d = res['N'], res['d']
            if (N_cap, d) not in results:
                results[(N_cap, d)] = {'rand': [], 'learned': []}
            results[(N_cap, d)]['rand'].append(res['alias_rand'])
            results[(N_cap, d)]['learned'].append(res['alias_learned'])
            
    # Reconstruct 2D matrices
    Z_rand = np.zeros((len(d_values), len(N_values)))
    Z_learned = np.zeros((len(d_values), len(N_values)))
    
    for i, d in enumerate(d_values):
        for j, N_cap in enumerate(N_values):
            Z_rand[i, j] = np.nanmean(results[(N_cap, d)]['rand'])
            Z_learned[i, j] = np.nanmean(results[(N_cap, d)]['learned'])
            
    return Z_rand, Z_learned

if __name__ == '__main__':
    n_samples = 100
    # Create an evenly spaced grid
    N_sweep = np.arange(20, 220, 20)
    d_sweep = np.arange(20, 220, 20)
    
    Z_rand, Z_learned = run_2d_sweep(N_sweep, d_sweep, n_samples=n_samples, trials=5)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Use LogNorm for clear heat intensity scaling
    from matplotlib.colors import LogNorm
    vmin = min(np.nanmin(Z_rand), np.nanmin(Z_learned))
    vmax = max(np.nanmax(Z_rand), np.nanmax(Z_learned))
    
    c1 = ax1.pcolormesh(N_sweep, d_sweep, Z_rand, shading='auto', norm=LogNorm(vmin=vmin, vmax=vmax), cmap='inferno')
    ax1.set_title(f"Random ReLU Aliasing (Fixed $n={n_samples}$)")
    ax1.set_xlabel("Model Capacity ($N$)")
    ax1.set_ylabel("Data Complexity ($d$)")
    
    c2 = ax2.pcolormesh(N_sweep, d_sweep, Z_learned, shading='auto', norm=LogNorm(vmin=vmin, vmax=vmax), cmap='inferno')
    ax2.set_title(f"Learned ReLU Aliasing (Fixed $n={n_samples}$)")
    ax2.set_xlabel("Model Capacity ($N$)")
    ax2.set_ylabel("Data Complexity ($d$)")
    
    # Add diagonal N=d line reference
    for ax in [ax1, ax2]:
        ax.plot([min(N_sweep), max(N_sweep)], [min(d_sweep), max(d_sweep)], color='cyan', linestyle='--', label='$N = d$')
        ax.axvline(x=n_samples, color='white', linestyle=':', label='$N = n_{samples}$')
        ax.legend()
    
    fig.colorbar(c2, ax=[ax1, ax2], label="Aliasing Operator Norm ($||H^+||_2$)")
    plt.savefig("aliasing_2d_phase_diagram.png", dpi=300)
    print("Phase diagrams generated and saved.")
