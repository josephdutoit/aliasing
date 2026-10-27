import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import concurrent.futures
import matplotlib.pyplot as plt
from scipy.linalg import pinv
import warnings

# Suppress expected numerical warnings near the singularity
warnings.filterwarnings("ignore", category=RuntimeWarning)

def learn_global_superposition(d, N=100, sparsity=0.05, global_samples=20000, epochs=1500):
    """
    Trains the superposition geometry on a clean, massive global population
    incorporating a negative bias threshold to accurately drop interference 
    into the negative half-space. L1 regularization is excluded.
    """
    # CRITICAL: Prevent PyTorch from multithreading inside the multiprocessing worker
    torch.set_num_threads(1)
    
    # Generate strictly sparse continuous features [0.5, 1.0] (no dense background noise)
    mask = (np.random.rand(global_samples, N) < sparsity).astype(float)
    X_np = mask * np.random.uniform(0.5, 1.0, size=(global_samples, N))
    X = torch.tensor(X_np, dtype=torch.float32)
    
    # Initialize weights and a negative bias vector
    W = nn.Parameter(torch.randn(d, N) / np.sqrt(d))
    b = nn.Parameter(torch.zeros(d) - 0.5)
    
    optimizer = optim.Adam([W, b], lr=0.01)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass: bottleneck with learned bias
        h = torch.relu(X @ W.T + b)
        X_hat = h @ W
        
        # Mean Squared Error reconstruction loss ONLY
        loss = torch.nn.functional.mse_loss(X_hat, X)
        
        loss.backward()
        optimizer.step()
        
        # Normalize columns of W to unit norm to stabilize the Gram matrix
        with torch.no_grad():
            W.data = W.data / torch.norm(W.data, dim=0, keepdim=True)
            
    return W.detach().numpy(), b.detach().numpy()

def evaluate_aliasing_geometry(d, n=40, N=100, sparsity=0.05, seed=None):
    """
    Evaluates the GAD aliasing operator norm on the constrained n=40 sample set
    using the globally learned superposition weights and bias.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    # 1. Learn the global superposition geometry (Weights and Bias)
    W_learned, b_learned = learn_global_superposition(d, N, sparsity)
    
    # 2. Initialize a random baseline for comparison
    W_rand = np.random.normal(0, 1 / np.sqrt(d), size=(d, N))
    W_rand = W_rand / np.linalg.norm(W_rand, axis=0, keepdims=True)
    
    # 3. Clean Evaluation Data Generation
    # Strictly sparse, no Gaussian noise leakage to preserve exact orthogonalization
    mask = (np.random.rand(n, N) < sparsity).astype(float)
    X_eval = mask * np.random.uniform(0.5, 1.0, size=(n, N))
    
    # 4. Construct the Conjugate Kernels (Fixed Representations)
    H_linear_learned = X_eval @ W_learned.T + b_learned
    H_relu_rand      = np.maximum(0, X_eval @ W_rand.T)
    H_relu_learned   = np.maximum(0, X_eval @ W_learned.T + b_learned)
    
    # 5. Calculate Pseudoinverse Norms (Aliasing Operator)
    def get_pinv_norm(H):
        try:
            # atol filters out absolute zeros mathematically created by the ReLU masking
            return np.linalg.norm(pinv(H, atol=1e-6), ord=2)
        except np.linalg.LinAlgError:
            return np.nan

    return {
        'd': d,
        'pinv_linear_learned': get_pinv_norm(H_linear_learned),
        'pinv_relu_rand': get_pinv_norm(H_relu_rand),
        'pinv_relu_learned': get_pinv_norm(H_relu_learned)
    }

def run_parallel_sweep(d_values, n=40, N=100, trials_per_d=15):
    """
    Executes the capacity sweep in parallel using exactly 28 workers.
    """
    results_agg = {d: {'lin_learn': [], 'relu_rand': [], 'relu_learn': []} for d in d_values}
    tasks = [(d, n, N, 0.05, d * 1000 + trial) for d in d_values for trial in range(trials_per_d)]
            
    print(f"Running {len(tasks)} parallel experiments across 28 workers...")
    
    # Explicitly enforce 28 workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=28) as executor:
        futures = [executor.submit(evaluate_aliasing_geometry, *task) for task in tasks]
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            d = res['d']
            results_agg[d]['lin_learn'].append(res['pinv_linear_learned'])
            results_agg[d]['relu_rand'].append(res['pinv_relu_rand'])
            results_agg[d]['relu_learn'].append(res['pinv_relu_learned'])
            
    # Aggregate results
    d_plot = sorted(d_values)
    pinv_lin_learn = [np.nanmean(results_agg[d]['lin_learn']) for d in d_plot]
    pinv_relu_rand = [np.nanmean(results_agg[d]['relu_rand']) for d in d_plot]
    pinv_relu_learn = [np.nanmean(results_agg[d]['relu_learn']) for d in d_plot]
        
    return d_plot, pinv_lin_learn, pinv_relu_rand, pinv_relu_learn

if __name__ == '__main__':
    N_features = 100
    n_samples = 40
    d_sweep = list(range(2, 96, 2))
    
    d_plot, pinv_lin_learn, pinv_relu_rand, pinv_relu_learn = run_parallel_sweep(
        d_sweep, n=n_samples, N=N_features, trials_per_d=15
    )
    
    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(d_plot, pinv_lin_learn, label='Linear Bottleneck (Learned $W$, $b$)', color='red', linestyle='--')
    plt.plot(d_plot, pinv_relu_rand, label='ReLU Bottleneck (Random $W$)', color='gray', linestyle='-.')
    plt.plot(d_plot, pinv_relu_learn, label='ReLU Bottleneck (Learned $W$, $b$)', color='blue', linewidth=2.5)
    
    plt.axvline(x=n_samples, color='black', linestyle=':', label=f'Interpolation Singularity ($n={n_samples}$)')
    
    plt.yscale('log')
    plt.title("Aliasing Mitigation: Superposition with Learned Bias vs Random Features")
    plt.xlabel("Bottleneck Capacity ($d$)")
    plt.ylabel("Aliasing Operator Norm ($||M_{TM}^+||_2$) [Log Scale]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("superposition_aliasing_bias_only.png", dpi=300)
    print("Experiment complete. Results saved to superposition_aliasing_bias_only.png")
