# Open-Source Biomaker Microfluidic Hardware

This directory contains specifications and fabrication blueprints for the low-cost, PDMS-free giant unilamellar vesicle (GUV) trapping arrays grounded in the rapid biomaker prototyping ethos of the **MIT Media Lab Community Biotechnology Initiative (CBI)**.

## Specifications

| Component | Material / Method | Thickness / Dimension | Purpose |
|---|---|---|---|
| **Base Plate** | PMMA (Cast Acrylic) | 1.5 mm | Structural support and fluidic manifold |
| **Fluidic Channels** | Double-sided PSA Tape | 50 $\mu$m | Flow layer and biocompatible bonding |
| **Pillar Arrays** | Desktop CO2 Laser Ablation | 20 $\mu$m gap | Hydrodynamic single-GUV immobilization |
| **Cover Slip** | Borosilicate Glass #1.5 | 170 $\mu$m | Confocal / epifluorescence optical access |
| **Fluidic Barbs** | 3D Printed (Biocompatible Resin) | 1/16" barb | Perfusion tubing interface ($Q = 0.5\,\mu\text{L/min}$) |

## Rapid Desktop Fabrication Workflow

1. **Laser Cutting PMMA**:
   - Cut base manifold geometries and port holes using a 40W desktop CO2 laser cutter (speed 15 mm/s, power 65%).
2. **Channel Patterning in PSA Tape**:
   - Laser-cut the fluidic channels, bypass loops, and 20 $\mu$m trapping pillar geometries directly into 50 $\mu$m acrylic transfer tape.
3. **Solvent-Free Lamination**:
   - Align and bond the tape layer between the PMMA manifold and cleaned #1.5 borosilicate glass cover slip using a roller press at 45°C.
4. **Perfusion Coupling**:
   - Connect 1/16" PTFE tubing to syringe pumps and place the assembled chip on an inverted microscope stage.
