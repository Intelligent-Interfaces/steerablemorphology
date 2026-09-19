# SteerableMorphology: Real-Time Control & Differentiable Digital Twins for Synthetic Protocells

<!-- rumdl-disable MD033 MD041 -->
<div align="center">
  <img src="papers/ALIFE2027/spudlenia_framework.png" alt="SteerableMorphology Framework" width="720"></img>
</div>

<div align="center">
  <a href="https://github.com/intelligent-interfaces/steerablemorphology"><img alt="PyPI - Python Version" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg"></a>
  <a href="https://github.com/intelligent-interfaces/steerablemorphology/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
  <a href="papers/ALIFE2027/main.pdf"><img alt="Paper - ALIFE 2027" src="https://img.shields.io/badge/paper-ALIFE%202027-B31B1B.svg"></a>
  <a href="papers/ICLR2027/main.pdf"><img alt="Paper - ICLR 2027" src="https://img.shields.io/badge/paper-ICLR%202027-purple.svg"></a>
  <a href="https://cgen-teleoperation.vercel.app"><img alt="Live Web Demo" src="https://img.shields.io/badge/demo-live%20visualizer-teal.svg"></a>
</div>
<!-- rumdl-enable MD033 MD041 -->

**SteerableMorphology** is an open-source library, predictive digital twin, and cybernetic control interface developed at the **Intelligent Interfaces Group (MIT Media Lab)**. It couples differentiable continuous cellular automata (such as [SpudLenia](https://github.com/intelligent-interfaces/spudlenia)) to benchtop microfluidic hardware, enabling real-time steering of synthetic protocell cytokinesis and emergent biological self-organization. 🧬

---

## Overview 🔎

Can we steer living and synthetic self-organizing matter in real time?

In continuous artificial life, reaction-diffusion fields and continuous cellular automata autonomously generate complex morphogenesis. Yet translating these *in silico* models to physical wet-lab protocells—such as Giant Unilamellar Vesicles (GUVs)—reveals a catastrophic control bottleneck:

1. **Fluid Transport Latency**: Physical microfluidic perfusion lines impose fluid transit delays of $5\text{--}30\,\text{seconds}$.
2. **Brittle Bifurcation Boundaries**: Protocells operate far from thermodynamic equilibrium. Instantaneous microscopic observation is too late: once osmotic swelling is visually detected, membrane rupture (lysis) is already physically irreversible.
3. **Discrete vs. Continuous Actuation**: Discrete numerical inputs excite high-wavenumber spatial oscillations that destroy vesicle stability.

**SteerableMorphology** resolves this bottleneck by deploying an online **differentiable digital twin** that unrolls trajectories $14.4\,\text{seconds}$ ahead of physical time. By estimating finite-time local Lyapunov exponents ($\lambda$), the system warns operators and throttles actuators $8.4\,\text{seconds}$ before non-linear instabilities occur. 🔬

---

## Why SteerableMorphology? 💡

### Fast & Differentiable 🚀
Built on PyTorch with spatial Fast Fourier Transform (FFT) convolutions, the simulator evaluates individual steps in $1.18\,\text{ms}$ on consumer GPUs. An entire 120-step predictive lookahead trajectory unrolls in just $141.6\,\text{ms}$, well within the microfluidic control latency budget.

### Predictive Stability Filtering 🛡️
Instead of relying on post-hoc static thresholds, SteerableMorphology computes local finite-time Lyapunov exponents:
$$\lambda(t) = \frac{1}{T_{\text{pred}} \Delta t} \ln \frac{\|\mathbf{A}_{\text{pert}}(t + T_{\text{pred}}) - \mathbf{A}_{\text{base}}(t + T_{\text{pred}})\|_2}{\|\delta \mathbf{A}_0\|_2}$$
Trajectories undergoing exponential divergence ($\lambda > 0$) trigger automatic pump throttling and software resistance, slashing osmotic rupture rates from $58.4\%$ to $7.6\%$.

### Ergonomic Tactile Controls 🎛️
Inspired by modular audio synthesis consoles and ecological affordance theory, the interactive console (`cgen`) maps continuous rotary potentiometers and motorized faders to continuous field parameters. Physical resistance acts as a mechanical low-pass filter on operator input velocity ($\|\dot{\boldsymbol{\theta}}\| \le v_{\max}$), preventing destructive step-transients.

### Open Biomaker Wet-Lab Pipeline 🌿
Designed in collaboration with the open-science prototyping ethos of the **MIT Media Lab Community Biotechnology Initiative (CBI)**, the platform replaces million-dollar cleanroom microfluidics with desktop laser-cut PMMA acrylic trapping arrays, cell-free TX-TL protein expression, and open-source microcontroller serial bridging.

---

## Quickstart 💻

### Installation

```bash
git clone https://github.com/intelligent-interfaces/steerablemorphology.git
cd steerablemorphology
pip install -e .
```

### Python API: Running Predictive Rollouts

```python
import torch
from steerablemorphology import SpudLeniaSimulator, SimulationConfig
from steerablemorphology.control import PredictiveStabilityFilter

# 1. Initialize differentiable simulator on GPU/MPS/CPU
cfg = SimulationConfig(grid_size=256, dt=0.12, n_substeps=10)
sim = SpudLeniaSimulator(cfg)

# 2. Spawn synthetic protocell state (Channels: [Lipid, Protein, Nutrient])
state = sim.init_state(batch_size=1)

# 3. Evaluate online predictive stability over 120 forward steps (14.4s lookahead)
filter_engine = PredictiveStabilityFilter()
params = {"alpha_neck": 1.25, "gamma_assim": 0.85, "lambda_waste": 0.020}

report, safe_params = filter_engine.evaluate(sim, state, params, horizon_steps=120)

print(f"Lyapunov Exponent (lambda): {report.lambda_val:.4f} s^-1")
print(f"Instability Warning: {report.is_warning} | Critical: {report.is_critical}")
print(f"Recommended Pump Throttle: {report.recommended_throttle * 100:.1f}%")
```

---

## Benchmarks 📊

### 1. Computational Latency & Predictive Horizon

Benchmarked on an Apple M1 Pro GPU (MPS / WebGL):

| Lattice Grid ($H \times W$) | Step Latency | Render Rate | BPTT Unroll ($T_{\text{pred}}=120$) | Lookahead Horizon ($T_{\text{lead}}$) | VRAM Footprint |
|---|---|---|---|---|---|
| $128 \times 128$ | $0.34\,\text{ms}$ | $60\,\text{FPS}$ | $40.8\,\text{ms}$ | $14.4\,\text{s}$ | $12.4\,\text{MB}$ |
| **$256 \times 256$ (Nominal)** | **$1.18\,\text{ms}$** | **$60\,\text{FPS}$** | **$141.6\,\text{ms}$** | **$14.4\,\text{s}$** | **$38.2\,\text{MB}$** |
| $512 \times 512$ | $4.42\,\text{ms}$ | $58\,\text{FPS}$ | $530.4\,\text{ms}$ | $14.4\,\text{s}$ | $142.8\,\text{MB}$ |

### 2. Stochastic Flow Perturbation Benchmark (500 Trials)

Comparison under random nutrient surges ($Q_{\text{syringe}} \in [0.8, 1.8]\,\mu\text{L/min}$) and pump drift:

| Metric | Static Threshold Baseline | Predictive Lyapunov (Ours) | Relative Improvement |
|---|---|---|---|
| **Warning Lead Time** | $0.9 \pm 0.4\,\text{s}$ | **$8.4 \pm 1.2\,\text{s}$** | **$+7.5\,\text{s}$ earlier warning** |
| **Averted Lysis / Rupture** | $41.6\%$ | **$92.4\%$** | **$+50.8\%$ recovery** |
| **Osmotic Rupture Rate ($\mathcal{R}_{\text{lysis}}$)** | $58.4\%$ | **$7.6\%$** | **$-87.0\%$ fewer failures** |
| **Starvation Rate ($\mathcal{R}_{\text{starve}}$)** | $14.2\%$ | **$2.8\%$** | **$-80.3\%$ fewer dissolutions** |
| **Successful Cytokinesis Yield ($\mathcal{Y}_{\text{cyto}}$)** | $27.4\%$ | **$89.6\%$** | **$+62.2\%$ division yield** |
| **False Alarm Rate** | $18.7\%$ | **$3.2\%$** | **$-82.9\%$ spurious warnings** |

---

## Closed-Loop Latency Budget ⏱️

$$\tau_{\text{closed-loop}} = \tau_{\text{input}} + \tau_{\text{BPTT}} + \tau_{\text{serial}} + \tau_{\text{pump}} \approx 10\,\text{ms} + 142\,\text{ms} + 20\,\text{ms} + 120\,\text{ms} \approx 292\,\text{ms}$$

Because the entire closed-loop latency is under **$0.3\,\text{seconds}$**, the digital twin preempts non-linear bifurcation boundaries well before physical fluids traverse the perfusion tubing ($5\text{--}30\,\text{seconds}$).

---

## Interactive Web Visualizer 🌐

The repository includes a standalone, client-side WebGL / WebSerial interface in [`web/`](web/):

- **Real-time continuous canvas**: Displays multi-scale lipid bilayer density, crowding gradients, and nutrient baths.
- **Physical actuator faders**: Ergonomic control rails with software resistance against unstable parameter regimes.
- **WebSerial integration**: Direct connection to USB microcontrollers running syringe pumps.

Launch locally:
```bash
cd web
python3 -m http.server 8000
# Open http://localhost:8000 in Chrome / Edge (supporting WebSerial)
```

---

## Repository Structure 📁

```
steerablemorphology/
├── README.md                  # CAX-style project documentation
├── LICENSE                    # MIT License
├── pyproject.toml             # Package metadata and build system
├── requirements.txt           # Python dependencies
├── papers/
│   ├── ALIFE2027/             # Camera-ready ALIFE 2027 manuscript & PDF
│   └── ICLR2027/              # ICLR 2027 manuscript & PDF
├── steerablemorphology/       # Core Python library
│   ├── simulation/            # Differentiable continuous PDE solver & kernels
│   ├── control/               # Online finite-time Lyapunov stability engine
│   └── hardware/              # Microfluidic serial communications bridge
├── web/                       # Standalone WebGL / WebSerial console
├── cad/                       # Laser-cut PMMA GUV trapping array blueprints
├── benchmarks/                # Latency & 500-trial stability benchmark scripts
└── tests/                     # Automated pytest verification suite
```

---

## Publications & Citations 📖

If you find this work helpful in your research, please cite our papers:

```bibtex
@inproceedings{oduniyi2027alife,
  title={Steering Emergent Morphogenesis: A Real-Time Control Interface and Differentiable Digital Twin for Synthetic Protocells},
  author={Oduniyi, Erick},
  booktitle={Artificial Life Conference Proceedings (ALIFE 2027)},
  year={2027}
}

@article{oduniyi2027cgen,
  title={Controlling Emergence: Cybernetic Teleoperation, Skeuomorphic Affordances, and Differentiable Digital Twins for Wet-Lab Neural Cellular Automata},
  author={Oduniyi, Erick and others},
  journal={Preprint},
  year={2027}
}
```

---

## License

Distributed under the **MIT License**. Developed by the **Intelligent Interfaces Group (IIG)** at the MIT Media Lab.
