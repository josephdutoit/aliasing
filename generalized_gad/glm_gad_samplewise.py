import torch
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
import warnings
from tqdm import tqdm

# Enforce float64. The interpolation singularity at M=N causes massive condition
# numbers. Float32 will suffer catastrophic precision loss at the spike.
torch.set_default_dtype(torch.float64)

# Each seed runs in its own worker process, so keep BLAS/torch single-threaded per
# worker. Otherwise N workers x all cores oversubscribes the CPU and can freeze the machine.
torch.set_num_threads(1)

def fisher_weights(M_TM_modeled, W_hat, distribution):
    """
    Per-sample Fisher weights w_i evaluated at the fitted model (observed information).

    For the canonical link, w_i = V(mu_i), the variance function at the fitted mean:
    - Gaussian: w_i = 1
    - Exponential: w_i = 1
    - Poisson: w_i = mu_i = exp(eta_i)
    - Logistic: w_i = p_i(1 - p_i), p_i = sigmoid(eta_i)

    These must be evaluated at the model's prediction, not at the sampled label:
    for Bernoulli labels y(1 - y) is identically zero.
    """
    eta = (M_TM_modeled @ W_hat).squeeze(-1).clamp(-30.0, 30.0)  # clamp avoids exp overflow

    if distribution in ('gaussian', 'exponential'):
        return torch.ones_like(eta)
    elif distribution == 'poisson':
        return torch.exp(eta)
    elif distribution == 'logistic':
        p = torch.sigmoid(eta)
        return p * (1 - p)
    raise ValueError(f"Unknown distribution: {distribution}")


def compute_n_eff(w):
    """
    Soft effective sample size (Kish): n_eff = (sum w)^2 / sum w^2.
    Equals n for constant weights and shrinks as the weight concentrates on few samples,
    so it needs no arbitrary cutoff for "dead" samples.
    """
    return ((w.sum() ** 2) / (w ** 2).sum()).item()


def safe_pinv(A):
    """Moore-Penrose pseudoinverse that returns NaNs (instead of raising) if the SVD fails."""
    if not torch.isfinite(A).all():
        return torch.full((A.shape[1], A.shape[0]), float('nan'), dtype=A.dtype)
    try:
        return torch.linalg.pinv(A)
    except torch.linalg.LinAlgError:
        return torch.full((A.shape[1], A.shape[0]), float('nan'), dtype=A.dtype)


def safe_norm(A, ord):
    """Matrix norm that returns NaN (instead of raising) for non-finite input or SVD failure."""
    if not torch.isfinite(A).all():
        return torch.tensor(float('nan'), dtype=A.dtype)
    try:
        return torch.linalg.norm(A, ord=ord)
    except torch.linalg.LinAlgError:
        return torch.tensor(float('nan'), dtype=A.dtype)


def fit_glm(M_TM, Y, distribution):
    """
    Finds the empirical model weights using L-BFGS.
    Initializing at zero guarantees gradient descent finds the minimum L2 norm solution.
    M_TM: Feature matrix (M samples × N features)
    """
    N = M_TM.shape[1]
    W = torch.zeros((N, 1), requires_grad=True)

    # Strong Wolfe line search is critical for navigating the extreme gradients near M=N
    optimizer = torch.optim.LBFGS([W], lr=1.0, max_iter=500, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        logits = M_TM @ W
        
        # The Bregman Divergence / Negative Log-Likelihood for each Exponential Family
        if distribution == 'gaussian':
            loss = 0.5 * torch.mean((logits - Y)**2)
        elif distribution == 'poisson':
            loss = torch.mean(torch.exp(logits) - Y * logits)
        elif distribution == 'logistic':
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, Y)
        elif distribution == 'exponential':
            loss = torch.mean(Y * torch.exp(-logits) + logits)
        else:
            raise ValueError("Unknown distribution")
            
        loss.backward()
        return loss

    optimizer.step(closure)
    return W.detach()


def run_single_experiment(distribution='gaussian', seed=42):
    """Run a single experiment with given seed, returns error lists."""
    torch.manual_seed(seed)

    F = 100  # True features in the universe
    N = 40   # Modeled features (our model's capacity)
    M_vals = torch.arange(5, 95, 2)

    # Scale true weights down slightly to keep Poisson/Exponential computationally stable
    W_true = torch.randn(F, 1) * 0.5

    # 1. The Best Possible Model (Bregman Projection onto modeled features)
    W_bregman = W_true.clone()
    W_bregman[N:] = 0

    E_mi_list, E_ga_list, E_di_list, E_total_list = [], [], [], []
    Mplus_2norm_list, Mplus_frob_list = [], []  # weighted pseudoinverse norms M_TM^+
    Iplus_2norm_list, Iplus_frob_list = [], []  # Fisher information pseudoinverse norms I^dagger
    XtX_pinv_2norm_list, XtX_pinv_frob_list = [], []  # (X^T X)^+ norms
    D_inv_2norm_list, D_inv_frob_list = [], []  # D^{-1} norms
    M_vals_valid = []  # Track which M values have valid (finite) errors
    singularity_M = None  # Will be set when M_eff >= N

    for M in M_vals:
        M_TM = torch.randn(M, F)  # Feature matrix: M samples × F features
        logits_true = M_TM @ W_true

        # 2. The True Universe (Sample from true distributions to get realistic data)
        if distribution == 'gaussian':
            Y_train = logits_true + torch.randn_like(logits_true) * 0.1
        elif distribution == 'poisson':
            # Sample true integer counts from Poisson distribution
            rates = torch.exp(logits_true)
            Y_train = torch.poisson(rates).float()
        elif distribution == 'logistic':
            # Sample binary outcomes from logistic probabilities
            probs = torch.sigmoid(logits_true)
            Y_train = torch.bernoulli(probs).float()
        elif distribution == 'exponential':
            # Sample exponential variates (using inverse transform sampling)
            rates = torch.exp(logits_true)
            Y_train = -torch.log(torch.rand_like(logits_true)) / rates

        # 3. The Empirical Model (Optimized on the visible N features)
        M_TM_modeled = M_TM[:, :N]  # Use only first N features
        W_hat_modeled = fit_glm(M_TM_modeled, Y_train, distribution)

        # Fisher weights evaluated at the fitted model (observed information),
        # not at the sampled labels (y(1-y) is identically 0 for Bernoulli labels).
        D_ii_star = fisher_weights(M_TM_modeled, W_hat_modeled, distribution)

        # A diverged fit (e.g. Poisson/logistic near separation) gives non-finite weights
        # that would crash pinv below; drop the point like other singular-regime points.
        if not (torch.isfinite(W_hat_modeled).all() and torch.isfinite(D_ii_star).all()):
            warnings.warn(f"Non-finite fit/weights at M={M.item()}, dropping from average", stacklevel=2)
            continue

        # Effective sample size and singularity threshold (n_eff >= d)
        M_eff = compute_n_eff(D_ii_star)
        if singularity_M is None and M_eff >= N:
            singularity_M = M.item()

        # Floor D_ii_star to avoid division by zero in D^{-1}
        D_ii_star = torch.clamp(D_ii_star, min=1e-10)

        # Fisher-weighted pseudoinverse M_TM^+ = (M_TM^T D M_TM)^+ M_TM^T D.
        # Exact identity: M_TM^+ = (D^{1/2} M_TM)^+ D^{1/2}, so we only need the
        # pseudoinverse of the whitened feature matrix. Dead samples (D_ii = 0)
        # become zero rows and drop out automatically.
        D_sqrt = torch.sqrt(D_ii_star)
        M_TM_whitened = D_sqrt[:, None] * M_TM_modeled
        M_TM_plus = safe_pinv(M_TM_whitened) * D_sqrt[None, :]

        # Compute norms of the weighted pseudoinverse
        Mplus_2norm = safe_norm(M_TM_plus, 2).item()  # Spectral norm
        Mplus_frob = torch.linalg.norm(M_TM_plus, ord='fro').item()  # Frobenius norm

        # Fisher information pseudoinverse I^dagger = (M~^T M~)^dagger with M~ = W_T^{1/2} M_TM.
        # Exact identity: ||I^dagger||_2 = ||M~^dagger||_2^2 = 1/sigma_min(M~)^2. Unlike M_TM^+,
        # the weights do not cancel here, so this blows up whenever fewer than d samples are live.
        FIM_pinv = safe_pinv(M_TM_whitened.T @ M_TM_whitened)
        Iplus_2norm = safe_norm(FIM_pinv, 2).item()  # Spectral norm
        Iplus_frob = torch.linalg.norm(FIM_pinv, ord='fro').item()  # Frobenius norm

        # Compute (M_TM^T M_TM)^+ norms
        M_TM_T_M_TM = M_TM_modeled.T @ M_TM_modeled
        M_TM_T_M_TM_pinv = safe_pinv(M_TM_T_M_TM)
        XtX_pinv_2norm = safe_norm(M_TM_T_M_TM_pinv, 2).item()
        XtX_pinv_frob = torch.linalg.norm(M_TM_T_M_TM_pinv, ord='fro').item()

        # Compute D^{-1} norms (inverse of diagonal matrix)
        D_inv_diag = 1.0 / D_ii_star
        D_inv_2norm = torch.max(torch.abs(D_inv_diag)).item()  # Max absolute value
        D_inv_frob = torch.norm(D_inv_diag).item()  # Frobenius norm of diagonal
        
        W_hat = torch.zeros(F, 1)
        W_hat[:N] = W_hat_modeled
        
        # 4. The Aliased Model
        # Dynamically compute the orthogonal null-space projection matrix
        P_data = safe_pinv(M_TM_modeled) @ M_TM_modeled
        P_null = torch.eye(N) - P_data
        
        W_alias_modeled = W_hat_modeled + P_null @ W_bregman[:N]
        W_alias = torch.zeros(F, 1)
        W_alias[:N] = W_alias_modeled
        
        # 5. Geometric Distances (Bregman L2 Divergences in Dual Space)
        E_mi = torch.norm(W_true - W_bregman)**2
        E_ga = torch.norm(W_bregman - W_alias)**2
        E_di = torch.norm(W_alias - W_hat)**2
        E_total = torch.norm(W_true - W_hat)**2
        
        # Check for NaN/Inf before computing errors
        # Also drop the point if any of the tracked norms failed (SVD did not converge)
        norms_finite = bool(torch.isfinite(torch.tensor([Mplus_2norm, Mplus_frob, Iplus_2norm, Iplus_frob, XtX_pinv_2norm, XtX_pinv_frob])).all())
        if not (norms_finite and torch.isfinite(E_mi) and torch.isfinite(E_ga) and torch.isfinite(E_di) and torch.isfinite(E_total)):
            warnings.warn(f"Singular regime at M={M.item()}: NaN/Inf detected, dropping from average", stacklevel=2)
            continue

        # Floor to 1e-7 for valid values
        E_mi_val = max(E_mi.item(), 1e-7)
        E_ga_val = max(E_ga.item(), 1e-7)
        E_di_val = max(E_di.item(), 1e-7)
        E_total_val = max(E_total.item(), 1e-7)

        M_vals_valid.append(M.item())
        E_mi_list.append(E_mi_val)
        E_ga_list.append(E_ga_val)
        E_di_list.append(E_di_val)
        E_total_list.append(E_total_val)
        Mplus_2norm_list.append(Mplus_2norm)
        Mplus_frob_list.append(Mplus_frob)
        Iplus_2norm_list.append(Iplus_2norm)
        Iplus_frob_list.append(Iplus_frob)
        XtX_pinv_2norm_list.append(XtX_pinv_2norm)
        XtX_pinv_frob_list.append(XtX_pinv_frob)
        D_inv_2norm_list.append(D_inv_2norm)
        D_inv_frob_list.append(D_inv_frob)

    return M_vals_valid, E_mi_list, E_ga_list, E_di_list, E_total_list, singularity_M, Mplus_2norm_list, Mplus_frob_list, XtX_pinv_2norm_list, XtX_pinv_frob_list, D_inv_2norm_list, D_inv_frob_list, Iplus_2norm_list, Iplus_frob_list


def _run_seed_wrapper(args):
    """Wrapper for multiprocessing Pool.map"""
    distribution, seed = args
    return run_single_experiment(distribution, seed)


def run_experiment(distribution='gaussian', num_seeds=100, num_workers=None):
    """Run experiment multiple times with different seeds, plot averaged results."""
    if num_workers is None:
        num_workers = os.cpu_count() or 4

    print(f"--- Running Bregman-GAD for: {distribution.upper()} ({num_seeds} seeds, {num_workers} workers) ---")

    # Prepare seed arguments
    seed_args = [(distribution, seed) for seed in range(num_seeds)]

    # Run in parallel
    print(f"  Using {num_workers} workers")

    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(_run_seed_wrapper, seed_args), total=num_seeds, desc="Seeds"))

    # Extract results (now includes filtered M_vals per seed)
    all_M_vals = [r[0] for r in results]
    all_mi = [r[1] for r in results]
    all_ga = [r[2] for r in results]
    all_di = [r[3] for r in results]
    all_total = [r[4] for r in results]
    singularity_Ms = [r[5] for r in results if r[5] is not None]
    all_Mplus_2norm = [r[6] for r in results]
    all_Mplus_frob = [r[7] for r in results]
    all_Iplus_2norm = [r[12] for r in results]
    all_Iplus_frob = [r[13] for r in results]
    all_XtX_pinv_2norm = [r[8] for r in results]
    all_XtX_pinv_frob = [r[9] for r in results]
    all_D_inv_2norm = [r[10] for r in results]
    all_D_inv_frob = [r[11] for r in results]

    # Find common M values across all seeds (intersection)
    common_M_vals = set(all_M_vals[0])
    for M_vals_seed in all_M_vals[1:]:
        common_M_vals &= set(M_vals_seed)
    common_M_vals = sorted(list(common_M_vals))

    # Build index mapping for each seed
    M_vals_indices = []
    for M_vals_seed in all_M_vals:
        indices = [i for i, M in enumerate(M_vals_seed) if M in common_M_vals]
        M_vals_indices.append(indices)

    # Compute mean across all seeds using only common M values
    E_mi_list = []
    E_ga_list = []
    E_di_list = []
    E_total_list = []
    Mplus_2norm_list = []
    Mplus_frob_list = []
    Iplus_2norm_list = []
    Iplus_frob_list = []
    XtX_pinv_2norm_list = []
    XtX_pinv_frob_list = []
    D_inv_2norm_list = []
    D_inv_frob_list = []

    for seed_idx in range(num_seeds):
        indices = M_vals_indices[seed_idx]
        E_mi_list.append([all_mi[seed_idx][i] for i in indices])
        E_ga_list.append([all_ga[seed_idx][i] for i in indices])
        E_di_list.append([all_di[seed_idx][i] for i in indices])
        E_total_list.append([all_total[seed_idx][i] for i in indices])
        Mplus_2norm_list.append([all_Mplus_2norm[seed_idx][i] for i in indices])
        Mplus_frob_list.append([all_Mplus_frob[seed_idx][i] for i in indices])
        Iplus_2norm_list.append([all_Iplus_2norm[seed_idx][i] for i in indices])
        Iplus_frob_list.append([all_Iplus_frob[seed_idx][i] for i in indices])
        XtX_pinv_2norm_list.append([all_XtX_pinv_2norm[seed_idx][i] for i in indices])
        XtX_pinv_frob_list.append([all_XtX_pinv_frob[seed_idx][i] for i in indices])
        D_inv_2norm_list.append([all_D_inv_2norm[seed_idx][i] for i in indices])
        D_inv_frob_list.append([all_D_inv_frob[seed_idx][i] for i in indices])

    # Average across seeds
    E_mi_list = [sum(E_mi_list[s][i] for s in range(num_seeds)) / num_seeds for i in range(len(common_M_vals))]
    E_ga_list = [sum(E_ga_list[s][i] for s in range(num_seeds)) / num_seeds for i in range(len(common_M_vals))]
    E_di_list = [sum(E_di_list[s][i] for s in range(num_seeds)) / num_seeds for i in range(len(common_M_vals))]
    E_total_list = [sum(E_total_list[s][i] for s in range(num_seeds)) / num_seeds for i in range(len(common_M_vals))]
    Mplus_2norm_list = [sum(Mplus_2norm_list[s][i] for s in range(num_seeds)) / num_seeds for i in range(len(common_M_vals))]
    Mplus_frob_list = [sum(Mplus_frob_list[s][i] for s in range(num_seeds)) / num_seeds for i in range(len(common_M_vals))]
    Iplus_frob_list = [sum(Iplus_frob_list[s][i] for s in range(num_seeds)) / num_seeds for i in range(len(common_M_vals))]
    Iplus_2norm_list = [sum(Iplus_2norm_list[s][i] for s in range(num_seeds)) / num_seeds for i in range(len(common_M_vals))]
    XtX_pinv_2norm_list = [sum(XtX_pinv_2norm_list[s][i] for s in range(num_seeds)) / num_seeds for i in range(len(common_M_vals))]
    XtX_pinv_frob_list = [sum(XtX_pinv_frob_list[s][i] for s in range(num_seeds)) / num_seeds for i in range(len(common_M_vals))]
    D_inv_2norm_list = [sum(D_inv_2norm_list[s][i] for s in range(num_seeds)) / num_seeds for i in range(len(common_M_vals))]
    D_inv_frob_list = [sum(D_inv_frob_list[s][i] for s in range(num_seeds)) / num_seeds for i in range(len(common_M_vals))]

    M_vals = common_M_vals

    # Compute average singularity threshold M across seeds
    avg_singularity_M = sum(singularity_Ms) / len(singularity_Ms) if singularity_Ms else None

    # Plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Top subplot: Errors
    ax1.semilogy(M_vals, E_total_list, label='Total Error', color='black', linewidth=2.5, marker='o', markersize=3)
    ax1.semilogy(M_vals, E_mi_list, label='Model Insufficiency ($\mathcal{E}_{mi}$)', color='green', linewidth=1.5)
    ax1.semilogy(M_vals, E_ga_list, label='Generalized Aliasing ($\mathcal{E}_{ga}$)', color='red', linewidth=2)
    ax1.semilogy(M_vals, E_di_list, label='Data Insufficiency ($\mathcal{E}_{di}$)', color='blue', linewidth=2)

    if avg_singularity_M is not None:
        ax1.axvline(x=avg_singularity_M, color='k', linestyle=':', linewidth=1.5, alpha=0.7, label=f'True Singularity ($d = n_{{eff}}$, avg={avg_singularity_M:.1f})')
    else:
        ax1.axvline(x=40, color='k', linestyle=':', linewidth=1.5, alpha=0.7, label='Interpolation Singularity ($d=40$)')
    ax1.grid(True, which='major', alpha=0.4, linewidth=0.8)
    ax1.grid(True, which='minor', alpha=0.15, linewidth=0.4, linestyle=':')

    # Set sensible log-scale limits to focus on interesting region
    all_errors = E_total_list + E_mi_list + E_ga_list + E_di_list
    min_err = min([e for e in all_errors if e > 0]) if any(e > 0 for e in all_errors) else 1e-4
    max_err = max(all_errors)
    ax1.set_ylim(min_err / 3, max_err * 3)

    ax1.set_xlabel('Number of Samples ($n$)', fontsize=11)
    ax1.set_ylabel('Geometric Error (Dual Space)', fontsize=11)
    ax1.set_title(f'Bregman-GAD with Unmodeled Features: {distribution.capitalize()}', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9, fontsize=10)

    # Second subplot: weighted pseudoinverse norms M_TM^+
    ax2.semilogy(M_vals, Iplus_2norm_list, label='$||\\mathcal{I}^{\\dagger}||_2$ (Spectral)', color='purple', linewidth=2, marker='s', markersize=3)
    ax2.semilogy(M_vals, Iplus_frob_list, label='$||\\mathcal{I}^{\\dagger}||_F$ (Frobenius)', color='orange', linewidth=2, marker='^', markersize=3)

    if avg_singularity_M is not None:
        ax2.axvline(x=avg_singularity_M, color='k', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Singularity (avg={avg_singularity_M:.1f})')
    else:
        ax2.axvline(x=40, color='k', linestyle=':', linewidth=1.5, alpha=0.7, label='Singularity ($d=40$)')
    ax2.grid(True, which='major', alpha=0.4, linewidth=0.8)
    ax2.grid(True, which='minor', alpha=0.15, linewidth=0.4, linestyle=':')

    ax2.set_xlabel('Number of Samples ($n$)', fontsize=11)
    ax2.set_ylabel('Norm', fontsize=11)
    ax2.set_title('$||\\mathcal{I}^{\\dagger}||$ (Fisher Information Pseudoinverse)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9, fontsize=10)

    # Third subplot: (X^T X)^+ norms
    ax3.semilogy(M_vals, Mplus_2norm_list, label='$||M_{TM}^{+}||_2$ (Weighted, Spectral)', color='purple', linewidth=2, marker='s', markersize=3)
    ax3.semilogy(M_vals, Mplus_frob_list, label='$||M_{TM}^{+}||_F$ (Weighted, Frobenius)', color='orange', linewidth=2, marker='^', markersize=3)
    ax3.semilogy(M_vals, XtX_pinv_2norm_list, label='$||(M_{TM}^T M_{TM})^{\\dagger}||_2$ (Unweighted, Spectral)', color='brown', linewidth=2, linestyle='--', marker='d', markersize=3)
    ax3.semilogy(M_vals, XtX_pinv_frob_list, label='$||(M_{TM}^T M_{TM})^{\\dagger}||_F$ (Unweighted, Frobenius)', color='pink', linewidth=2, linestyle='--', marker='v', markersize=3)

    if avg_singularity_M is not None:
        ax3.axvline(x=avg_singularity_M, color='k', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Singularity (avg={avg_singularity_M:.1f})')
    else:
        ax3.axvline(x=40, color='k', linestyle=':', linewidth=1.5, alpha=0.7, label='Singularity ($d=40$)')
    ax3.grid(True, which='major', alpha=0.4, linewidth=0.8)
    ax3.grid(True, which='minor', alpha=0.15, linewidth=0.4, linestyle=':')

    ax3.set_xlabel('Number of Samples ($n$)', fontsize=11)
    ax3.set_ylabel('Norm', fontsize=11)
    ax3.set_title('$||M_{TM}^{+}||$ vs. Unweighted $||M_{TM}^{\\dagger}||$', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', framealpha=0.9, fontsize=10)

    # Fourth subplot: D^{-1} norms
    ax4.semilogy(M_vals, D_inv_2norm_list, label='$||W_T^{-1}||_2$ (Max element)', color='teal', linewidth=2, marker='o', markersize=3)
    ax4.semilogy(M_vals, D_inv_frob_list, label='$||W_T^{-1}||_F$ (Frobenius)', color='cyan', linewidth=2, marker='*', markersize=5)

    if avg_singularity_M is not None:
        ax4.axvline(x=avg_singularity_M, color='k', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Singularity (avg={avg_singularity_M:.1f})')
    else:
        ax4.axvline(x=40, color='k', linestyle=':', linewidth=1.5, alpha=0.7, label='Singularity ($d=40$)')
    ax4.grid(True, which='major', alpha=0.4, linewidth=0.8)
    ax4.grid(True, which='minor', alpha=0.15, linewidth=0.4, linestyle=':')

    ax4.set_xlabel('Number of Samples ($n$)', fontsize=11)
    ax4.set_ylabel('Norm', fontsize=11)
    ax4.set_title('$||W_T^{-1}||$ (Inverse Fisher Weights)', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', framealpha=0.9, fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{distribution}_samplewise.png", dpi=150)
    plt.close(fig)

# Run the experiments
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run Bregman-GAD samplewise experiments')
    parser.add_argument('--num-workers', type=int, default=None, help='Number of worker processes (default: auto-detect)')
    parser.add_argument('--num-seeds', type=int, default=100, help='Number of random seeds to run (default: 100)')
    args = parser.parse_args()

    run_experiment('logistic', num_seeds=args.num_seeds, num_workers=args.num_workers)
    run_experiment('gaussian', num_seeds=args.num_seeds, num_workers=args.num_workers)
    run_experiment('poisson', num_seeds=args.num_seeds, num_workers=args.num_workers)
    run_experiment('exponential', num_seeds=args.num_seeds, num_workers=args.num_workers)