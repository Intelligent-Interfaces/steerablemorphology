import torch
from typing import Dict, Any, Tuple
from steerablemorphology.simulation.engine import SpudLeniaSimulator

def compute_finite_time_lyapunov(
    sim: SpudLeniaSimulator,
    state: torch.Tensor,
    horizon_steps: int = 120,
    perturbation_eps: float = 1e-4,
    params: Dict[str, Any] = None
) -> Tuple[float, torch.Tensor]:
    """
    Estimates the finite-time local Lyapunov exponent (lambda) by rolling out
    the baseline trajectory and a perturbed trajectory over the predictive horizon.
    
    Args:
        sim: SpudLeniaSimulator instance.
        state: Current state tensor (1, 3, H, W).
        horizon_steps: Number of forward lookahead steps (T_pred).
        perturbation_eps: Magnitude of isotropic perturbation.
        params: Current parameter dictionary.
        
    Returns:
        lambda_val: Scalar Lyapunov exponent (s^-1).
        final_predicted_state: Final state tensor after horizon_steps.
    """
    # 1. Generate isotropic perturbation
    noise = torch.randn_like(state)
    noise_norm = torch.norm(noise)
    delta_0 = perturbation_eps * (noise / (noise_norm + 1e-8))
    state_pert = torch.clamp(state + delta_0, 0.0, 1.0)
    
    # 2. Forward rollouts over horizon
    base_curr = state
    pert_curr = state_pert
    
    with torch.no_grad():
        for _ in range(horizon_steps):
            base_curr = sim.forward_step(base_curr, params)
            pert_curr = sim.forward_step(pert_curr, params)
            
    # 3. Compute Euclidean divergence
    diff_final = pert_curr - base_curr
    norm_final = torch.norm(diff_final).item()
    norm_initial = torch.norm(delta_0).item()
    
    # 4. Compute Lyapunov exponent (s^-1)
    total_physical_time = horizon_steps * sim.cfg.dt  # e.g., 120 * 0.12 = 14.4 s
    if norm_final <= 1e-12:
        lambda_val = -1.0
    else:
        lambda_val = (1.0 / total_physical_time) * torch.log(torch.tensor(norm_final / (norm_initial + 1e-12))).item()
        
    return float(lambda_val), base_curr
