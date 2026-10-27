import os
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
from scipy.linalg import pinv

def experiment_for_capacity(d, n=40, N=100, sparsity=0.1, seed=None):
    """
    Runs a single fixed-representation Conjugate Kernel experiment for a given bottleneck capacity d.
    n: number of samples (fixed at 40 to match the featurewise sweep[cite: 10])
    N: total number of true orthogonal features
    d: modeled bottleneck capacity
    """
    if seed is not None:
        np.random.seed(seed)
        
    # 1. Generate sparse, true orthogonal features (X)
    # Each sample has a low probability of a feature being active
    X = (np.random.rand(n, N) < sparsity).astype(float)
    # Add small Gaussian noise to simulate realistic inputs
    X += np.random.normal(0, 0.01, size=(n, N))
    
    # 2. Force Superposition (Bottleneck Weights W)
    # Generate a random Gaussian matrix to act as an overcomplete frame
    # This packs N features into d dimensions[cite: 9]
    W = np.random.normal(0, 1 / np.sqrt(d), size=(d, N))
    
    # Normalize the columns of W to have unit norm for stable Gram matrix calculation
    W = W / np.linalg.norm(W, axis=0, keepdims=True)
    
    # 3. Quantify Superposition Interference
    # Calculate the Gram matrix of the weights: G = W^T W
    G = W.T @ W
    # The off-diagonal elements represent the non-orthogonal interference between features
    np.fill_diagonal(G, 0)
    max_interference = np.max(np.abs(G))
    mean_interference = np.mean(np.abs(G))
    
    # 4. Construct Fixed Representations (Conjugate Kernels)
    # Linear bottleneck (unmitigated)
    H_linear = X @ W.T 
    # Non-linear bottleneck (ReLU acts as dynamic mask)
    H_relu = np.maximum(0, X @ W.T)
    
    # 5. Measure Aliasing Vulnerability
    # Aliasing is governed by the norm of the pseudoinverse of the feature matrix[cite: 8]
    # We use the spectral norm (ord=2) of the pseudoinverse
    try:
        # Compute pseudoinverse norms. We floor small singular values to prevent infinite float limits.
        pinv_linear = pinv(H_linear, atol=1e-7)
        pinv_relu = pinv(H_relu, atol=1e-7)
        
        norm_pinv_linear = np.linalg.norm(pinv_linear, ord=2)
        norm_pinv_relu = np.linalg.norm(pinv_relu, ord=2)
    except np.linalg.LinAlgError:
        norm_pinv_linear = np.nan
        norm_pinv_relu = np.nan
        
    return {
        'd': d,
        'max_interference': max_interference,
        'mean_interference': mean_interference,
        'norm_pinv_linear': norm_pinv_linear,
        'norm_pinv_relu': norm_pinv_relu
    }

def run_parallel_sweep(d_values, n=40, N=100, trials_per_d=max(1, min(20, int(os.getenv("GAD_TRIALS", "20"))))):
    """
    Executes the capacity sweep in parallel over available CPU cores.
    """
    results_agg = {d: {'max_int': [], 'mean_int': [], 'pinv_lin': [], 'pinv_relu': []} for d in d_values}
    
    # Create a list of arguments for the pool
    tasks = []
    for d in d_values:
        for trial in range(trials_per_d):
            seed = d * 1000 + trial  # Ensure reproducible but varied seeds
            tasks.append((d, n, N, 0.1, seed))
            
    print(f"Running {len(tasks)} parallel experiments...")
    
    # Execute in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, min(28, int(os.getenv("GAD_MAX_WORKERS", "28"))))) as executor:
        futures = [executor.submit(experiment_for_capacity, *task) for task in tasks]
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            d = res['d']
            results_agg[d]['max_int'].append(res['max_interference'])
            results_agg[d]['mean_int'].append(res['mean_interference'])
            results_agg[d]['pinv_lin'].append(res['norm_pinv_linear'])
            results_agg[d]['pinv_relu'].append(res['norm_pinv_relu'])
            
    # Average the results over trials
    d_plot = []
    pinv_lin_plot = []
    pinv_relu_plot = []
    max_int_plot = []
    
    for d in sorted(d_values):
        d_plot.append(d)
        max_int_plot.append(np.nanmean(results_agg[d]['max_int']))
        pinv_lin_plot.append(np.nanmean(results_agg[d]['pinv_lin']))
        pinv_relu_plot.append(np.nanmean(results_agg[d]['pinv_relu']))
        
    return d_plot, max_int_plot, pinv_lin_plot, pinv_relu_plot

if __name__ == '__main__':
    # Experiment parameters aligned with the featurewise sweep[cite: 10]
    N_features = 100
    n_samples = 40
    d_sweep = list(range(2, 96, 2))[::max(1, int(os.getenv("GAD_GRID_STRIDE", "1")))]
    
    # Run the parallel sweep
    d_plot, max_int_plot, pinv_lin_plot, pinv_relu_plot = run_parallel_sweep(
        d_sweep, n=n_samples, N=N_features, trials_per_d=max(1, min(50, int(os.getenv("GAD_TRIALS", "50"))))
    )
    
    # Plotting the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Proof of Superposition Interference
    ax1.plot(d_plot, max_int_plot, color='purple', marker='o', markersize=4)
    ax1.set_title("Structural Superposition Interference ($W^T W$)")
    ax1.set_xlabel("Bottleneck Capacity ($d$)")
    ax1.set_ylabel("Max Off-Diagonal Magnitude")
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, "Non-zero values prove non-orthogonal feature packing", 
             transform=ax1.transAxes, verticalalignment='top', bbox=dict(alpha=0.8, facecolor='white'))

    # Plot 2: Aliasing Mitigation via Pseudoinverse Norm
    ax2.plot(d_plot, pinv_lin_plot, label='Linear Bottleneck', color='red', linestyle='--')
    ax2.plot(d_plot, pinv_relu_plot, label='ReLU Bottleneck', color='blue')
    ax2.axvline(x=n_samples, color='gray', linestyle=':', label=f'Interpolation Singularity ($n={n_samples}$)')
    
    ax2.set_yscale('log')
    ax2.set_title("Aliasing Operator Norm ($||M_{TM}^+||_2$)")
    ax2.set_xlabel("Bottleneck Capacity ($d$)")
    ax2.set_ylabel("Spectral Norm (Log Scale)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("superposition_aliasing_mitigation.png", dpi=300)
    print("Experiment complete. Results saved to superposition_aliasing_mitigation.png")
