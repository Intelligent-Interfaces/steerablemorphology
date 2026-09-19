import pytest
import torch
from steerablemorphology.simulation.engine import SpudLeniaSimulator, SimulationConfig
from steerablemorphology.control.lyapunov import compute_finite_time_lyapunov
from steerablemorphology.control.filter import PredictiveStabilityFilter
from steerablemorphology.hardware.serial_bridge import MicrofluidicSerialBridge

def test_simulator_initialization():
    cfg = SimulationConfig(grid_size=64, device="cpu")
    sim = SpudLeniaSimulator(cfg)
    state = sim.init_state(batch_size=1)
    assert state.shape == (1, 3, 64, 64)
    assert torch.all(state >= 0.0) and torch.all(state <= 1.0)

def test_forward_step():
    cfg = SimulationConfig(grid_size=64, device="cpu")
    sim = SpudLeniaSimulator(cfg)
    state = sim.init_state(batch_size=1)
    next_state = sim.forward_step(state)
    assert next_state.shape == (1, 3, 64, 64)
    assert torch.all(next_state >= 0.0) and torch.all(next_state <= 1.0)

def test_lyapunov_estimation():
    cfg = SimulationConfig(grid_size=64, device="cpu")
    sim = SpudLeniaSimulator(cfg)
    state = sim.init_state(batch_size=1)
    lam, final_state = compute_finite_time_lyapunov(sim, state, horizon_steps=10)
    assert isinstance(lam, float)
    assert final_state.shape == (1, 3, 64, 64)

def test_stability_filter():
    cfg = SimulationConfig(grid_size=64, device="cpu")
    sim = SpudLeniaSimulator(cfg)
    state = sim.init_state(batch_size=1)
    filt = PredictiveStabilityFilter()
    params = {"gamma_assim": 0.85, "lambda_waste": 0.020}
    report, safe_params = filt.evaluate(sim, state, params, horizon_steps=10)
    assert hasattr(report, "lambda_val")
    assert "gamma_assim" in safe_params

def test_serial_bridge_mock():
    bridge = MicrofluidicSerialBridge(mock=True)
    assert bridge.connect() is True
    pkt = bridge.send_actuation_packet(q_syringe=0.5, p_laser=25.0, q_dial=1.2)
    assert pkt["type"] == "ACTUATE"
    assert pkt["q_syringe_ul_min"] == 0.5
