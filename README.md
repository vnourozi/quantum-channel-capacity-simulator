# Quantum Channel Capacity Simulator
Numerical tools and reproducible experiments for capacity bounds of qubit quantum channels, including **flagged-extension upper bounds**, **coherent information lower bounds**, and **SDP-based upper bounds**.

This repository accompanies the paper:

**V. Nourozi**, *Flagged Extensions and Numerical Simulations for Quantum Channel Capacity: Bridging Theory and Computation*  
arXiv:2506.03429 (2025)  
https://arxiv.org/abs/2506.03429

---

## Overview
This codebase provides simulations and figure-generation scripts for:
- **Amplitude Damping (AD)** channel (parameterized by damping rate `γ`)
- **Generalized Amplitude Damping (GADC)** channel (parameterized by `γ` and thermal bath population `N_th`)
- **Depolarizing** channel (parameterized by depolarizing probability `p`)

It supports generating **journal-ready figures** comparing:
- One-shot coherent information (optimized over qubit inputs)
- Reverse coherent information (diagnostic/comparison curve)
- Flagged-extension capacity upper bounds (single-letter computable bounds)
- SDP-based additive upper bound **QΓ** (when enabled / available)

---

