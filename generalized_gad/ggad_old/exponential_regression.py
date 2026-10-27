import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

# Strict 64-bit precision to handle the Itakura-Saito ratios
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

# ==========================================
# 1. Setup the Ground Truth "Universe"
# ==========================================
N = 100           
M_test = 20000    

W_star = torch.randn(N, 1) / math.sqrt(N)
X_test = torch.randn(M_test, N)

# Log-link activation (clamped to prevent precision cascades)
def safe_exp(X, W):
    logits = torch.clamp(X @ W, min=-20.0, max=20.0)
    return torch.exp(logits)

# True population means (mu)
mu_star = safe_exp(X_test, W_star)

M_values = list(range(20, 320, 20))

E_ga_list, E_di_list, Total_Dev_list = [], [], []

print(f"{'M (Samples)':<12} | {'E_ga (Aliasing)':<15} | {'E_di (Data Insuff.)':<20} | {'Total IS Dist':<15}")
print("-" * 68)

# The Bregman Divergence for the Exponential Distribution
def itakura_saito_distance(mu_true, mu_pred):
    eps = 1e-9 
    mu_pred = mu_pred.clamp(min=eps)
    mu_true = mu_true.clamp(min=eps)
    ratio = mu_true / mu_pred
    return (ratio - torch.log(ratio) - 1.0).mean()

for M in M_values:
    # ==========================================
    # 2. The Finite Sample & Algorithmic Reality
    # ==========================================
    X_train = torch.randn(M, N)
    mu_train = safe_exp(X_train, W_star)
    
    # Generate labels using Exponential stochastic sampling.
    # Variance equals mu^2, creating massive, heavy-tailed outliers.
    dist = torch.distributions.Exponential(rate=1.0 / mu_train.clamp(min=1e-9))
    y_train = dist.sample() 
    
    W_hat = torch.zeros(N, 1, requires_grad=True)
    opt_hat = torch.optim.LBFGS([W_hat], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")
    
    def closure_hat():
        opt_hat.zero_grad()
        logits = torch.clamp(X_train @ W_hat, min=-20.0, max=20.0)
        
        # Negative Log-Likelihood of the Exponential distribution
        # NLL = log(mu) + (y / mu) = logits + y * exp(-logits)
        loss = (logits + y_train * torch.exp(-logits)).mean()
        
        total_loss = loss + 1e-3 * torch.sum(W_hat ** 2) 
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([W_hat], max_norm=10.0)
        return total_loss
        
    opt_hat.step(closure_hat)
    W_hat_det = W_hat.detach()
    
    # ==========================================
    # STAGE 2: The Non-Linear Bregman Projection (W_alias)
    # ==========================================
    X_pinv = torch.linalg.pinv(X_train)
    P_data = X_pinv @ X_train
    P_null = torch.eye(N) - P_data
    
    v = W_star.clone().requires_grad_(True)
    opt_alias = torch.optim.LBFGS([v], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")
    
    def closure_alias():
        opt_alias.zero_grad()
        W_alias_curr = W_hat_det + (P_null @ v)
        mu_alias_curr = safe_exp(X_test, W_alias_curr)
        
        # Minimize Itakura-Saito distance in the unobserved dimensions
        loss = itakura_saito_distance(mu_star, mu_alias_curr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([v], max_norm=10.0)
        return loss
        
    opt_alias.step(closure_alias)
    W_alias_det = W_hat_det + (P_null @ v.detach())
    
    # ==========================================
    # 3. Measure Bregman-GAD Errors
    # ==========================================
    mu_hat = safe_exp(X_test, W_hat_det)
    mu_alias = safe_exp(X_test, W_alias_det)
    
    E_ga = itakura_saito_distance(mu_star, mu_alias).item()
    E_di = itakura_saito_distance(mu_alias, mu_hat).item()
    Total_Dev = itakura_saito_distance(mu_star, mu_hat).item()
    
    E_ga_list.append(E_ga)
    E_di_list.append(E_di)
    Total_Dev_list.append(Total_Dev)
    
    print(f"{M:<12} | {E_ga:<15.4f} | {E_di:<20.4f} | {Total_Dev:<15.4f}")

# ==========================================
# 4. Plot the Results
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(M_values, Total_Dev_list, label="Total Error (W_hat)", color="black", linewidth=2.5)
plt.plot(M_values, E_ga_list, label="Generalized Aliasing (E_ga)", color="red", linestyle="--")
plt.plot(M_values, E_di_list, label="Data Insufficiency (E_di)", color="blue", linestyle="-.")
plt.axvline(x=N, color="gray", linestyle=":", label="Interpolation Threshold (M=N)")

plt.title("Bregman-GAD: True Double Descent with Itakura-Saito Distance", fontsize=14)
plt.xlabel("Sample Size (M)", fontsize=12)
plt.ylabel("Expected Itakura-Saito Distance", fontsize=12)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("exponential_regression.png")