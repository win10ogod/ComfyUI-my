import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# Helper to expand sigma to batch size, correcting the dimension error
def expand_sigma(sigma, x):
    if isinstance(sigma, (float, int)):
        sigma = torch.tensor(sigma, device=x.device, dtype=x.dtype)
    return sigma.expand(x.shape[0])

def get_d(model, x, sigma, extra_args):
    """
    Calculates the derivative dx/d(sigma), which is (x - denoised) / sigma.
    This version ensures sigma is correctly shaped for the model.
    """
    sigma_batched = expand_sigma(sigma, x)
    denoised = model(x, sigma_batched, **extra_args)
    return (x - denoised) / sigma

@torch.enable_grad() # Use context manager for cleaner gradient handling
def sample_fshores(model, x, sigmas, extra_args=None, callback=None, disable=None, lambda_=0.05, eta=0.01):
    """
    Fixed-Step High-Order Resonance-Enhanced Sampler (FSHORES)
    Corrected implementation based on user's design and feedback.
    """
    extra_args = extra_args or {}
    x_current = x
    
    for i in tqdm(range(len(sigmas) - 1), disable=disable, desc="FSHORES"):
        sigma_t = sigmas[i]
        sigma_next = sigmas[i+1]

        if sigma_next == 0:
            continue

        # The step h is negative, following k-diffusion convention for correct integration direction
        h = (sigma_next - sigma_t).item()

        # Wrapper for the derivative function
        def f(x_in, sigma_in):
            return get_d(model, x_in, sigma_in, extra_args)

        # Dormand-Prince 5(4) coefficients
        c2, c3, c4, c5 = 1/5, 3/10, 4/5, 8/9
        a31, a32 = 3/40, 9/40
        a41, a42, a43 = 44/45, -56/15, 32/9
        a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
        a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
        b1, b3, b4, b5, b6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84

        # RK5 stages with proper sigma interpolation
        d1 = f(x_current, sigma_t)
        k1 = h * d1
        
        d2 = f(x_current + 1/5 * k1, sigma_t + c2 * h)
        k2 = h * d2

        d3 = f(x_current + a31 * k1 + a32 * k2, sigma_t + c3 * h)
        k3 = h * d3

        d4 = f(x_current + a41 * k1 + a42 * k2 + a43 * k3, sigma_t + c4 * h)
        k4 = h * d4

        d5 = f(x_current + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4, sigma_t + c5 * h)
        k5 = h * d5
        
        d6 = f(x_current + a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5, sigma_next) # c6=1
        k6 = h * d6

        # Standard 5th order update. With h < 0, this correctly moves x towards the denoised state.
        x_hat = x_current + (b1 * d1 + b3 * d3 + b4 * d4 + b5 * d5 + b6 * d6) * h

        # --- Resonance Enhancement Term (R) ---
        
        x_grad = x_current.detach().requires_grad_(True)
        
        # Re-calculate d_current with graph enabled on a clean tensor
        d_current_grad = f(x_grad, sigma_t)
        
        # Calculate gradient of the norm of the score: grad_s = nabla_x ||s_theta(x, t)||_2
        # grad_outputs must be a tuple or list.
        grad_s = torch.autograd.grad(d_current_grad.sum(), x_grad, grad_outputs=(torch.ones_like(d_current_grad.sum()),))[0]
        
        # Approximate second derivative using finite differences
        delta_x = eta * k1.detach()
        d_perturbed = f(x_current.detach() + delta_x, sigma_t)
        # Use abs(h) for the time delta approximation, as h is negative
        df_dt_approx = (d_perturbed - d1.detach()) / (0.01 * abs(h) + 1e-8)

        # Calculate Resonance Term R
        R = lambda_ * (sigma_t**2) * (df_dt_approx * grad_s)
        
        # Final update with Resonance Enhancement. h**2 is positive as required.
        x_next = x_hat + (h**2) * R
        
        if callback is not None:
            callback({'x': x_current, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': x_next})

        x_current = x_next

    return x_current
