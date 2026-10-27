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
    Generates d true sparse features and embeds them into a D_in dimensional ambient space.
    """
    # 1. Generate intrinsic sparse features (Toy Models of Superposition style)
    mask = (torch.rand(n_samples, d) < prob_active).float()
    vals = torch.rand(n_samples, d)
    X_latent = mask * vals
    
    # 2. Embed into high-dimensional ambient space (D_in)
    # Use a random Gaussian projection matrix
    projection_matrix = torch.randn(d, D_in) / np.sqrt(d)
    X_embedded = X_latent @ projection_matrix
    
    # 3. Add ambient noise to ensure full rank in the observable space
    X_embedded += torch.randn(n_samples, D_in) * 0.01
    
    return X_embedded

def train_tms_autoencoder(X, N_capacity, epochs=2500, lr=0.01):
    """
    Trains a Toy Models of Superposition autoencoder on the high-dimensional input.
    """
    torch.set_num_threads(1)
    D_in = X.shape[1]
    
    W = nn.Parameter(torch.randn(N_capacity, D_in) / np.sqrt(N_capacity))
    b1 = nn.Parameter(torch.zeros(N_capacity) - 0.1) # Negative interference threshold
    b2 = nn.Parameter(torch.zeros(D_in))
    
    optimizer = optim.Adam([W, b1, b2], lr=lr)
    
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

def experiment_feature_complexity(d, N=100, D_in=500, prob_active=0.05, seed=None):
    """
    Evaluates a fixed capacity N across varying true feature dimensions d.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    # Generate exactly N samples embedded in D_in dimensions
    X = generate_embedded_data(n_samples=N, d=d, D_in=D_in, prob_active=prob_active)
    
    # 1. Random ReLU Model (Blind to alignment)
    W_rand = torch.randn(N, D_in) / np.sqrt(N)
    W_rand = W_rand / torch.norm(W_rand, dim=0, keepdim=True)
    b1_rand = torch.zeros(N) - 0.1
    
    H_rand = torch.relu(X @ W_rand.T + b1_rand).numpy()
    
    # 2. Learned ReLU Network (Capable of feature alignment)
    W_learned, b1_learned = train_tms_autoencoder(X, N_capacity=N)
    H_learned = torch.relu(X @ W_learned.T + b1_learned).numpy()
    
    def get_pinv_norm(H):
        try:
            return np.linalg.norm(pinv(H, atol=1e-6), ord=2)
        except np.linalg.LinAlgError:
            return np.nan

    return {
        'd': d,
        'pinv_rand': get_pinv_norm(H_rand),
        'pinv_learned': get_pinv_norm(H_learned)
    }

def run_complexity_sweep(d_values, N=100, D_in=500, trials_per_d=15):
    results = {d: {'rand': [], 'learned': []} for d in d_values}
    tasks = [(d, N, D_in, 0.05, d * 1000 + trial) for d in d_values for trial in range(trials_per_d)]
            
    print(f"Running {len(tasks)} parallel experiments using 28 workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=28) as executor:
        futures = [executor.submit(experiment_feature_complexity, *task) for task in tasks]
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            d = res['d']
            results[d]['rand'].append(res['pinv_rand'])
            results[d]['learned'].append(res['pinv_learned'])
            
    d_plot = sorted(d_values)
    pinv_rand = [np.nanmean(results[d]['rand']) for d in d_plot]
    pinv_learned = [np.nanmean(results[d]['learned']) for d in d_plot]
        
    return d_plot, pinv_rand, pinv_learned

if __name__ == '__main__':
    N_datapoints = 100
    D_in_ambient = 500
    
    # Sweep data complexity d across the fixed model capacity N
    d_sweep = list(range(10, 210, 10))
    
    d_plot, pinv_rand, pinv_learned = run_complexity_sweep(
        d_sweep, N=N_datapoints, D_in=D_in_ambient, trials_per_d=15
    )
    
    plt.figure(figsize=(12, 7))
    plt.plot(d_plot, pinv_rand, label='Random ReLU Features (Unaligned)', color='gray', linestyle='-.')
    plt.plot(d_plot, pinv_learned, label='Learned ReLU Network (Aligned)', color='blue', linewidth=2.5)
    
    plt.axvline(x=N_datapoints, color='black', linestyle=':', label=f'Data Complexity == Model Capacity ($d=N={N_datapoints}$)')
    
    plt.yscale('log')
    plt.title("Double Descent Mitigated by Feature Alignment")
    plt.xlabel("True Generating Features ($d$)")
    plt.ylabel("Aliasing Operator Norm ($||M_{TM}^+||_2$) [Log Scale]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("embedded_complexity_double_descent.png", dpi=300)
    print("Experiment complete. Results saved to embedded_complexity_double_descent.png")
