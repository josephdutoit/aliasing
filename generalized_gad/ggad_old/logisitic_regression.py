import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

# Set seeds for reproducibility
torch.manual_seed(42)

# ==========================================
# 1. Setup the Ground Truth "Universe"
# ==========================================
N = 100           # Number of parameters (capacity)
M_test = 20000    # Massive test set for population expectations

# CRITICAL FIX 1: The Noise Injection
# Scale the initialization so pre-activations are O(1). 
# This forces p_star into the ~0.3 to ~0.7 range, maximizing Bernoulli 
# variance p(1-p) and injecting the high-frequency label noise needed for aliasing.
W_star = torch.randn(N, 1) / math.sqrt(N)

# Generate population test data and true probabilities (h_*)
X_test = torch.randn(M_test, N)
p_star = torch.sigmoid(X_test @ W_star)

# Sweep sample sizes across the interpolation threshold (N=100)
M_values = list(range(20, 500, 20))

E_ga_list, E_di_list, Total_KL_list = [], [], []

print(f"{'M (Samples)':<12} | {'E_ga (Aliasing)':<15} | {'E_di (Data Insuff.)':<20} | {'Total KL':<10}")
print("-" * 65)

# Binary KL Divergence Helper
def binary_kl(p, q):
    eps = 1e-7
    p = p.clamp(eps, 1-eps)
    q = q.clamp(eps, 1-eps)
    return (p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))).mean()

for M in M_values:
    # Generate finite training data with stochastic label noise
    X_train = torch.randn(M, N)
    p_train = torch.sigmoid(X_train @ W_star)
    y_train = torch.bernoulli(p_train) # The high-frequency aliasing fodder
    
    # ==========================================
    # STAGE 1: The Algorithmic Reality (W_hat)
    # ==========================================
    W_hat = torch.zeros(N, 1, requires_grad=True)
    opt_hat = torch.optim.LBFGS([W_hat], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")
    
    def closure_hat():
        opt_hat.zero_grad()
        loss = F.binary_cross_entropy_with_logits(X_train @ W_hat, y_train)
        # Tiny L2 penalty selects the min-norm separating hyperplane in the overparameterized regime
        total_loss = loss + 1e-4 * torch.sum(W_hat ** 2) 
        total_loss.backward()
        return total_loss
        
    opt_hat.step(closure_hat)
    W_hat_det = W_hat.detach()
    
    # ==========================================
    # STAGE 2: The Non-Linear Bregman Projection (W_alias)
    # ==========================================
    # Calculate null space projection matrix
    X_pinv = torch.linalg.pinv(X_train)
    P_data = X_pinv @ X_train
    P_null = torch.eye(N) - P_data
    
    # CRITICAL FIX 2: Null Space Optimization
    # Initialize a perturbation vector v. We start it at W_star to give the 
    # optimizer a fast convergence to the true geometry in the unseen dimensions.
    v = W_star.clone().requires_grad_(True)
    opt_alias = torch.optim.LBFGS([v], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")
    
    def closure_alias():
        opt_alias.zero_grad()
        # Parameterize W_alias so it is geometrically locked to W_hat on the training data
        W_alias_curr = W_hat_det + (P_null @ v)
        p_alias_curr = torch.sigmoid(X_test @ W_alias_curr)
        
        # Minimize KL Divergence over the population
        loss = binary_kl(p_star, p_alias_curr)
        loss.backward()
        return loss
        
    opt_alias.step(closure_alias)
    
    # Finalize the true Bregman Aliased matrix
    W_alias_det = W_hat_det + (P_null @ v.detach())
    
    # ==========================================
    # 3. Measure Bregman-GAD Errors
    # ==========================================
    p_hat = torch.sigmoid(X_test @ W_hat_det)
    p_alias = torch.sigmoid(X_test @ W_alias_det)
    
    E_ga = binary_kl(p_star, p_alias).item()  # D_KL(h_* || h_alias)
    E_di = binary_kl(p_alias, p_hat).item()   # D_KL(h_alias || h_hat)
    Total_KL = binary_kl(p_star, p_hat).item()
    
    E_ga_list.append(E_ga)
    E_di_list.append(E_di)
    Total_KL_list.append(Total_KL)
    
    print(f"{M:<12} | {E_ga:<15.4f} | {E_di:<20.4f} | {Total_KL:<10.4f}")

# ==========================================
# 4. Plot the Results
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(M_values, Total_KL_list, label="Total Error (W_hat)", color="black", linewidth=2.5)
plt.plot(M_values, E_ga_list, label="Generalized Aliasing (E_ga)", color="red", linestyle="--")
plt.plot(M_values, E_di_list, label="Data Insufficiency (E_di)", color="blue", linestyle="-.")
plt.axvline(x=N, color="gray", linestyle=":", label="Interpolation Threshold (M=N)")

plt.title("Bregman-GAD: True Double Descent in KL Space", fontsize=14)
plt.xlabel("Sample Size (M)", fontsize=12)
plt.ylabel("Expected KL Divergence", fontsize=12)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("bregman_gad.png")