"""
SteerableMorphology: Real-time control interface and differentiable digital twin
for steering continuous cellular automata and synthetic protocell morphogenesis.
Intelligent Interfaces Group (IIG).
"""

__version__ = "0.1.0"
__author__ = "Intelligent Interfaces Group"

from steerablemorphology.simulation.engine import SpudLeniaSimulator, SimulationConfig
from steerablemorphology.simulation.kernels import generate_annular_kernel
from steerablemorphology.control.lyapunov import compute_finite_time_lyapunov
from steerablemorphology.control.filter import PredictiveStabilityFilter

__all__ = [
    "SpudLeniaSimulator",
    "SimulationConfig",
    "generate_annular_kernel",
    "compute_finite_time_lyapunov",
    "PredictiveStabilityFilter",
]
