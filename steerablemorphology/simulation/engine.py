import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Dict, Any

from steerablemorphology.simulation.kernels import generate_annular_kernel

@dataclass
class SimulationConfig:
    grid_size: int = 256
    dt: float = 0.12            # Outer timestep (seconds)
    n_substeps: int = 10        # Inner sub-steps for CFL stability (dt_sub = 0.012 s)
    dx: float = 0.5             # Spatial discretization (microns)
    diff_m: float = 0.015       # Lipid passive diffusion
    diff_c: float = 0.010       # Crowding protein diffusion
    diff_s: float = 0.040       # Nutrient diffusion
    r_short: float = 3.5        # Bilayer cohesion kernel radius
    sigma_short: float = 1.8
    r_crowd: float = 7.0        # Steric crowding kernel radius
    sigma_crowd: float = 2.5
    r_feed: float = 13.5        # Feeder assimilation kernel radius
    sigma_feed: float = 3.5
    mu_lip: float = 0.150       # Lipid potential center
    sigma_lip: float = 0.038
    alpha_neck: float = 1.25    # Equatorial furrow brake
    gamma_assim: float = 0.85   # Feeder assimilation gain
    eta_expr: float = 0.14      # Protein expression rate
    delta_decay: float = 0.035  # Protein turnover decay
    beta_curve: float = 0.22    # Curvature segregation affinity
    lambda_waste: float = 0.020 # Waste clearance rate
    device: str = "cpu"

class SpudLeniaSimulator(nn.Module):
    """
    Differentiable continuous cellular automaton simulator for protocell cytokinesis.
    Solves coupled non-local reaction-diffusion-convolution equations.
    """
    def __init__(self, config: SimulationConfig = None):
        super().__init__()
        self.cfg = config or SimulationConfig()
        self.device = torch.device(self.cfg.device)
        self.kernel_size = 41
        
        # Initialize multi-scale annular kernels
        self.k_short = generate_annular_kernel(self.cfg.r_short, self.cfg.sigma_short, self.kernel_size, self.device)
        self.k_crowd = generate_annular_kernel(self.cfg.r_crowd, self.cfg.sigma_crowd, self.kernel_size, self.device)
        self.k_feed = generate_annular_kernel(self.cfg.r_feed, self.cfg.sigma_feed, self.kernel_size, self.device)
        
        # 5-point discrete Laplacian kernel
        laplacian = torch.tensor([[0.0, 1.0, 0.0],
                                  [1.0, -4.0, 1.0],
                                  [0.0, 1.0, 0.0]], device=self.device) / (self.cfg.dx ** 2)
        self.laplacian_kernel = laplacian.unsqueeze(0).unsqueeze(0)

    def init_state(self, batch_size: int = 1, radius: float = 24.0) -> torch.Tensor:
        """Initializes a batch of spherical protocells on the 2D grid."""
        H = W = self.cfg.grid_size
        coords = torch.linspace(-(W // 2), W // 2, W, device=self.device)
        y, x = torch.meshgrid(coords, coords, indexing="ij")
        r = torch.sqrt(x**2 + y**2)
        
        # Lipid membrane: annular shell
        m = torch.exp(-0.5 * ((r - radius) / 3.0) ** 2)
        # Protein: uniform inside
        c = 0.2 * torch.sigmoid((radius - r) / 2.0)
        # Nutrient: external ambient bath
        s = 0.8 * torch.sigmoid((r - (radius + 8.0)) / 4.0)
        
        state = torch.stack([m, c, s], dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return state

    def forward_step(self, state: torch.Tensor, params: Dict[str, Any] = None) -> torch.Tensor:
        """
        Executes one outer step (dt) composed of n_substeps for CFL stability.
        state: (B, 3, H, W) where channels are [M, C, S]
        """
        dt_sub = self.cfg.dt / self.cfg.n_substeps
        p_neck = params.get("alpha_neck", self.cfg.alpha_neck) if params else self.cfg.alpha_neck
        p_assim = params.get("gamma_assim", self.cfg.gamma_assim) if params else self.cfg.gamma_assim
        p_waste = params.get("lambda_waste", self.cfg.lambda_waste) if params else self.cfg.lambda_waste
        
        curr_state = state
        pad = self.kernel_size // 2
        
        # Compute spatial convolutions once per outer step (saves FFT compute)
        m_pad = F.pad(curr_state[:, 0:1], (pad, pad, pad, pad), mode="circular")
        c_pad = F.pad(curr_state[:, 1:2], (pad, pad, pad, pad), mode="circular")
        s_pad = F.pad(curr_state[:, 2:3], (pad, pad, pad, pad), mode="circular")
        
        u_short = F.conv2d(m_pad, self.k_short)
        u_crowd = F.conv2d(c_pad, self.k_crowd)
        u_feed = F.conv2d(s_pad, self.k_feed)
        
        # Non-linear growth activation
        growth = 2.0 * torch.exp(-0.5 * ((u_short - self.cfg.mu_lip) / self.cfg.sigma_lip) ** 2) - 1.0
        
        # Growth braking at furrow induced by crowding
        furrow_inhibition = 1.0 / (1.0 + p_neck * u_crowd)
        effective_growth = growth * furrow_inhibition
        
        # Sub-stepping integration
        for _ in range(self.cfg.n_substeps):
            # Diffusion via Laplacian
            m_l = F.conv2d(F.pad(curr_state[:, 0:1], (1, 1, 1, 1), mode="circular"), self.laplacian_kernel)
            c_l = F.conv2d(F.pad(curr_state[:, 1:2], (1, 1, 1, 1), mode="circular"), self.laplacian_kernel)
            s_l = F.conv2d(F.pad(curr_state[:, 2:3], (1, 1, 1, 1), mode="circular"), self.laplacian_kernel)
            
            # Channel derivatives
            dM = self.cfg.diff_m * m_l + 0.1 * effective_growth + p_assim * curr_state[:, 0:1] * u_feed - p_waste * curr_state[:, 0:1]
            dC = self.cfg.diff_c * c_l + self.cfg.eta_expr * curr_state[:, 0:1] - self.cfg.delta_decay * curr_state[:, 1:2]
            dS = self.cfg.diff_s * s_l - p_assim * curr_state[:, 0:1] * u_feed
            
            # Euler update
            new_m = torch.clamp(curr_state[:, 0:1] + dt_sub * dM, 0.0, 1.0)
            new_c = torch.clamp(curr_state[:, 1:2] + dt_sub * dC, 0.0, 1.0)
            new_s = torch.clamp(curr_state[:, 2:3] + dt_sub * dS, 0.0, 1.0)
            curr_state = torch.cat([new_m, new_c, new_s], dim=1)
            
        return curr_state

    def rollout(self, state: torch.Tensor, steps: int, params: Dict[str, Any] = None) -> torch.Tensor:
        """Rolls out the simulation for T steps forward."""
        trajectory = [state]
        curr = state
        for _ in range(steps):
            curr = self.forward_step(curr, params)
            trajectory.append(curr)
        return torch.stack(trajectory, dim=1)
