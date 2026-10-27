import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

# CRITICAL FIX 1: Enforce strict 64-bit precision globally
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

# ==========================================
# 1. Setup the Ground Truth "Universe"
# ==========================================
N = 100           # Number of parameters (capacity)
M_test = 20000    # Massive test set for population expectations

# Initialize true weights
W_star = torch.randn(N, 1) / math.sqrt(N)
X_test = torch.randn(M_test, N)

# CRITICAL FIX 2: Bounded Exponential Activation
def safe_exp(X, W):
    # Clamp logits between -20 and 20. 
    # exp(20) is ~4.8e8, well within float64 bounds, preventing overflow cascades.
    logits = torch.clamp(X @ W, min=-20.0, max=20.0)
    return torch.exp(logits)

# True population rates
lambda_star = safe_exp(X_test, W_star)

# Sweep sample sizes spanning the N=100 threshold
M_values = list(range(20, 320, 20))

E_ga_list, E_di_list, Total_Dev_list = [], [], []

print(f"{'M (Samples)':<12} | {'E_ga (Aliasing)':<15} | {'E_di (Data Insuff.)':<20} | {'Total Deviance':<15}")
print("-" * 68)

# Poisson Deviance (Bregman Divergence)
def poisson_deviance(y_true, y_pred):
    eps = 1e-9 # Stricter epsilon for float64
    y_pred = y_pred.clamp(min=eps)
    y_true = y_true.clamp(min=eps)
    return (y_true * torch.log(y_true / y_pred) - (y_true - y_pred)).mean()

for M in M_values:
    # ==========================================
    # 2. The Finite Sample & Algorithmic Reality
    # ==========================================
    X_train = torch.randn(M, N)
    
    # Generate labels using heteroscedastic Poisson stochastic sampling
    lambda_train = safe_exp(X_train, W_star)
    y_train = torch.poisson(lambda_train) 
    
    # STAGE 1: The Algorithmic Reality (W_hat)
    W_hat = torch.zeros(N, 1, requires_grad=True)
    # Reduced learning rate and increased max_iter for stable convergence
    opt_hat = torch.optim.LBFGS([W_hat], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")
    
    def closure_hat():
        opt_hat.zero_grad()
        logits = torch.clamp(X_train @ W_hat, min=-20.0, max=20.0)
        loss = F.poisson_nll_loss(logits, y_train, log_input=True)
        # 1e-3 L2 penalty bounds the explosive singularity at M=100
        total_loss = loss + 1e-3 * torch.sum(W_hat ** 2) 
        total_loss.backward()
        
        # CRITICAL FIX 3: Gradient Clipping
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
        lambda_alias_curr = safe_exp(X_test, W_alias_curr)
        loss = poisson_deviance(lambda_star, lambda_alias_curr)
        loss.backward()
        
        # Gradient Clipping for the projection optimization
        torch.nn.utils.clip_grad_norm_([v], max_norm=10.0)
        return loss
        
    opt_alias.step(closure_alias)
    W_alias_det = W_hat_det + (P_null @ v.detach())
    
    # ==========================================
    # 3. Measure Bregman-GAD Errors
    # ==========================================
    lambda_hat = safe_exp(X_test, W_hat_det)
    lambda_alias = safe_exp(X_test, W_alias_det)
    
    E_ga = poisson_deviance(lambda_star, lambda_alias).item()
    E_di = poisson_deviance(lambda_alias, lambda_hat).item()
    Total_Dev = poisson_deviance(lambda_star, lambda_hat).item()
    
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

plt.title("Bregman-GAD: True Double Descent in Poisson Space", fontsize=14)
plt.xlabel("Sample Size (M)", fontsize=12)
plt.ylabel("Expected Poisson Deviance", fontsize=12)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("poisson_regression.png")