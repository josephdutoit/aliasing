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

def learn_global_superposition(d, N=100, sparsity=0.05, global_samples=20000, epochs=1500):
    """
    Trains the superposition geometry on a clean, massive global population.
    Data Generation: Continuous sparse features (no dense background noise).
    """
    torch.set_num_threads(1)
    global_samples = max(1, min(global_samples, int(os.getenv("GAD_GLOBAL_SAMPLES", str(global_samples)))))
    
    # Generate strictly sparse continuous features [0, 1]
    mask = (np.random.rand(global_samples, N) < sparsity).astype(float)
    X_np = mask * np.random.uniform(0.5, 1.0, size=(global_samples, N))
    X = torch.tensor(X_np, dtype=torch.float32)
    
    W = nn.Parameter(torch.randn(d, N) / np.sqrt(d))
    optimizer = optim.Adam([W], lr=0.01)
    
    for _ in range(max(1, min(epochs, int(os.getenv("GAD_EPOCHS", str(epochs)))))):
        optimizer.zero_grad()
        h = torch.relu(X @ W.T)
        X_hat = h @ W
        loss = torch.nn.functional.mse_loss(X_hat, X)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            W.data = W.data / torch.norm(W.data, dim=0, keepdim=True)
            
    return W.detach().numpy()

def evaluate_aliasing_geometry(d, n=40, N=100, sparsity=0.05, seed=None):
    """
    Evaluates the GAD aliasing operator norm on the constrained n=40 sample set.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    W_learned = learn_global_superposition(d, N, sparsity)
    
    W_rand = np.random.normal(0, 1 / np.sqrt(d), size=(d, N))
    W_rand = W_rand / np.linalg.norm(W_rand, axis=0, keepdims=True)
    
    # 3. Clean Evaluation Data Generation (CRITICAL FIX)
    # Strictly sparse, no Gaussian noise leakage.
    mask = (np.random.rand(n, N) < sparsity).astype(float)
    X_eval = mask * np.random.uniform(0.5, 1.0, size=(n, N))
    
    H_linear_learned = X_eval @ W_learned.T
    H_relu_rand      = np.maximum(0, X_eval @ W_rand.T)
    H_relu_learned   = np.maximum(0, X_eval @ W_learned.T)
    
    def get_pinv_norm(H):
        try:
            # atol filters out absolute zeros mathematically created by the ReLU
            return np.linalg.norm(pinv(H, atol=1e-6), ord=2)
        except np.linalg.LinAlgError:
            return np.nan

    return {
        'd': d,
        'pinv_linear_learned': get_pinv_norm(H_linear_learned),
        'pinv_relu_rand': get_pinv_norm(H_relu_rand),
        'pinv_relu_learned': get_pinv_norm(H_relu_learned)
    }

def run_parallel_sweep(d_values, n=40, N=100, trials_per_d=max(1, min(15, int(os.getenv("GAD_TRIALS", "15"))))):
    results_agg = {d: {'lin_learn': [], 'relu_rand': [], 'relu_learn': []} for d in d_values}
    tasks = [(d, n, N, 0.05, d * 1000 + trial) for d in d_values for trial in range(trials_per_d)]
            
    print(f"Running {len(tasks)} parallel experiments...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, min(28, int(os.getenv("GAD_MAX_WORKERS", "28"))))) as executor:
        futures = [executor.submit(evaluate_aliasing_geometry, *task) for task in tasks]
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            d = res['d']
            results_agg[d]['lin_learn'].append(res['pinv_linear_learned'])
            results_agg[d]['relu_rand'].append(res['pinv_relu_rand'])
            results_agg[d]['relu_learn'].append(res['pinv_relu_learned'])
            
    d_plot = sorted(d_values)
    pinv_lin_learn = [np.nanmean(results_agg[d]['lin_learn']) for d in d_plot]
    pinv_relu_rand = [np.nanmean(results_agg[d]['relu_rand']) for d in d_plot]
    pinv_relu_learn = [np.nanmean(results_agg[d]['relu_learn']) for d in d_plot]
        
    return d_plot, pinv_lin_learn, pinv_relu_rand, pinv_relu_learn

if __name__ == '__main__':
    N_features = 100
    n_samples = 40
    d_sweep = list(range(2, 96, 2))[::max(1, int(os.getenv("GAD_GRID_STRIDE", "1")))]
    
    d_plot, pinv_lin_learn, pinv_relu_rand, pinv_relu_learn = run_parallel_sweep(
        d_sweep, n=n_samples, N=N_features, trials_per_d=max(1, min(15, int(os.getenv("GAD_TRIALS", "15"))))
    )
    
    plt.figure(figsize=(12, 7))
    plt.plot(d_plot, pinv_lin_learn, label='Linear Bottleneck (Learned $W$)', color='red', linestyle='--')
    plt.plot(d_plot, pinv_relu_rand, label='ReLU Bottleneck (Random $W$)', color='gray', linestyle='-.')
    plt.plot(d_plot, pinv_relu_learn, label='ReLU Bottleneck (Learned $W$)', color='blue', linewidth=2.5)
    plt.axvline(x=n_samples, color='black', linestyle=':', label=f'Interpolation Singularity ($n={n_samples}$)')
    
    plt.yscale('log')
    plt.title("Aliasing Mitigation: Pure Sparse Superposition vs Random Features")
    plt.xlabel("Bottleneck Capacity ($d$)")
    plt.ylabel("Aliasing Operator Norm ($||M_{TM}^+||_2$) [Log Scale]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("clean_superposition_aliasing.png", dpi=300)
    print("Experiment complete.")
