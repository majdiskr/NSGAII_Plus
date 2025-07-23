# NSGA-II+ for Vehicular Fog Computing

This repository contains MATLAB implementation of the NSGA-II+ algorithm with:
- Adaptive population sizing
- Dynamic convergence
- Pareto-based selection
- Custom mutation/crossover

## How to Run

1. Load your input data (task workload, deadlines, etc.)
2. Execute `MainNSGA.m` or `NSGA2_2.m`
3. Results will include delay, energy, convergence time

## Requirements

- MATLAB R2021a+
- Parallel Computing Toolbox (optional)

## Hardware Specifications
- CPU: Intel Core i7-12700H @ 2.30GHz
- RAM: 16 GB
- OS: Windows 11 (64-bit)
- Software: MATLAB R2023a
- Toolboxes: Parallel Computing Toolbox (used for `parfor`)

## Simulation Settings
- Max Iterations: 100
- Population Sizes: 50–200 (adaptive)
- Number of Vehicles: {50, 100, 200}
- Number of Tasks: {100, 200, 400}
- Task Workload: Uniform(200–500) million cycles
- Deadlines: 100–500 ms
- Seed: `rng(42)` for reproducibility

## 📊 Dataset
- Synthetic workloads generated with controlled variability
- Emulates real-world vehicular fog computing with deadline pressure

## 📂 Notes
- All code is vectorized for efficiency.
- Performance is benchmarked using delay, energy, and execution time metrics.

## Author

Majdi Mustafa Sukkar - Marwadi  University
