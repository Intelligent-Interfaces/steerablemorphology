import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from steerablemorphology.simulation.engine import SpudLeniaSimulator, SimulationConfig

def run_latency_benchmark():
    grids = [128, 256, 512]
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=== SteerableMorphology Computational Latency Benchmark ===")
    print(f"Hardware Device: {device}\n")
    print(f"{'Lattice Grid':<18} | {'Step Time':<12} | {'Unroll (T=120)':<16} | {'Lookahead':<12} | {'VRAM / Memory'}")
    print("-" * 75)

    for g in grids:
        cfg = SimulationConfig(grid_size=g, device=device)
        sim = SpudLeniaSimulator(cfg)
        state = sim.init_state(batch_size=1)

        # Warmup
        for _ in range(5):
            state = sim.forward_step(state)

        # Measure single step
        n_iters = 30
        t0 = time.perf_counter()
        for _ in range(n_iters):
            state = sim.forward_step(state)
        step_time_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

        # Measure 120-step unroll
        t1 = time.perf_counter()
        _ = sim.rollout(state, steps=120)
        unroll_time_ms = (time.perf_counter() - t1) * 1000.0

        lookahead_s = 120 * cfg.dt
        vram_mb = (state.element_size() * state.nelement() * 10) / (1024 * 1024)

        print(f"{f'{g}x{g}':<18} | {f'{step_time_ms:.2f} ms':<12} | {f'{unroll_time_ms:.1f} ms':<16} | {f'{lookahead_s:.1f} s':<12} | {f'{vram_mb:.1f} MB'}")

if __name__ == "__main__":
    run_latency_benchmark()
