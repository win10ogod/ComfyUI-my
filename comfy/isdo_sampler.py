import torch
import torch.nn.functional as F
from tqdm.auto import trange

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / sigma.view(-1, 1, 1, 1)

@torch.no_grad()
def sample_isdo(model, x, sigmas, extra_args=None, callback=None, disable=None, 
                 max_correction_strength=0.5, perturbation_angle=2.0):
    """ 
    Infinite Spectral Diffusion Odyssey (ISDO) Sampler - Final, Stabilized Implementation.

    This version introduces a sigma-dependent correction strength, ensuring that the topological 
    corrections are strongest at the beginning of sampling and gently fade out, preventing 
    noise from being introduced in the final steps. This provides stability and a clean convergence.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    max_sigma = sigmas[0] # The starting sigma, which is the largest.

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_cur = sigmas[i]
        
        # 1. Get the baseline denoised output
        denoised_base = model(x, sigma_cur * s_in, **extra_args)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma_cur, 'sigma_hat': sigma_cur, 'denoised': denoised_base})

        if sigmas[i + 1] == 0:
            x = denoised_base
            continue

        # --- ISDO Derivative Calculation ---
        
        # Calculate sigma-dependent correction strength
        # The correction is strongest at the beginning and fades to zero.
        current_strength = max_correction_strength * (sigma_cur / max_sigma)

        # 2. Symmetry Probe to find instability
        angle = torch.deg2rad(torch.ones(x.shape[0], device=x.device) * perturbation_angle)
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        theta = torch.zeros(x.shape[0], 2, 3, device=x.device)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_perturbed = F.grid_sample(x, grid, align_corners=False)

        denoised_perturbed = model(x_perturbed, sigma_cur * s_in, **extra_args)

        theta_inv = torch.zeros(x.shape[0], 2, 3, device=x.device)
        theta_inv[:, 0, 0] = cos_a
        theta_inv[:, 0, 1] = sin_a
        theta_inv[:, 1, 0] = -sin_a
        theta_inv[:, 1, 1] = cos_a
        grid_inv = F.affine_grid(theta_inv, denoised_perturbed.size(), align_corners=False)
        denoised_aligned = F.grid_sample(denoised_perturbed, grid_inv, align_corners=False)

        # 3. Construct the ISDO derivative
        instability = denoised_aligned - denoised_base
        target = denoised_base - instability * current_strength
        d_isdo = to_d(x, sigma_cur, target)

        # 4. Solve with a standard Euler step
        dt = sigmas[i + 1] - sigma_cur
        x = x + d_isdo * dt

    return x
