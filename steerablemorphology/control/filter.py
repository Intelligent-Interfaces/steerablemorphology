import torch
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from steerablemorphology.simulation.engine import SpudLeniaSimulator
from steerablemorphology.control.lyapunov import compute_finite_time_lyapunov

@dataclass
class StabilityReport:
    lambda_val: float
    is_warning: bool
    is_critical: bool
    recommended_throttle: float
    predicted_biomass: float

class PredictiveStabilityFilter:
    """
    Closed-loop predictive stability filter that monitors unrolled trajectories
    and throttles actuators before non-linear bifurcation boundaries are crossed.
    """
    def __init__(self, warning_threshold: float = -0.05, critical_threshold: float = 0.0):
        self.warning_thresh = warning_threshold
        self.critical_thresh = critical_threshold
        
    def evaluate(
        self,
        sim: SpudLeniaSimulator,
        state: torch.Tensor,
        current_params: Dict[str, Any],
        horizon_steps: int = 120
    ) -> Tuple[StabilityReport, Dict[str, Any]]:
        """
        Evaluates stability and returns safe throttled parameters.
        """
        lam, pred_state = compute_finite_time_lyapunov(
            sim, state, horizon_steps=horizon_steps, params=current_params
        )
        
        pred_biomass = torch.sum(pred_state[:, 0]).item()
        is_warning = lam >= self.warning_thresh
        is_critical = lam >= self.critical_thresh
        
        safe_params = dict(current_params)
        throttle = 1.0
        
        if is_critical:
            # Throttle feeding to halt volume expansion
            throttle = 0.35
            safe_params["gamma_assim"] = safe_params.get("gamma_assim", 0.85) * throttle
            safe_params["lambda_waste"] = safe_params.get("lambda_waste", 0.020) * 1.5
        elif is_warning:
            throttle = 0.70
            safe_params["gamma_assim"] = safe_params.get("gamma_assim", 0.85) * throttle
            
        report = StabilityReport(
            lambda_val=lam,
            is_warning=is_warning,
            is_critical=is_critical,
            recommended_throttle=throttle,
            predicted_biomass=pred_biomass
        )
        return report, safe_params
