# sciml-pcb-thermal

Physics-informed machine learning for PCB thermal dynamics with adaptive solver and dataset generation.

## Overview

This project develops and benchmarks methods for learning PCB thermal transients with:
- **Physics-based ground truth**: FD discretization of 2D heat equation with conductive + radiative coupling
- **Resolution-independent formulation**: Heater/domain sizes in physical units, not grid-dependent
- **Transient + steady-state solvers**: Newton iteration (steady) + adaptive RK45/Radau (transient)
- **Dataset generation**: Random problem parameters for training Neural ODEs / PINNs
- **Modular architecture**: Shared physics code + per-method implementations

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Examples
```bash
# Steady-state solution
python experiments/run_experiment.py --experiment steady --display

# Transient solution
python experiments/run_experiment.py --experiment transient --time 10.0 --dt 0.1 --display

# Generate dataset
python experiments/run_experiment.py --experiment dataset --n_samples 100 --time 10.0 --dt 0.1
```

### Validation
```bash
# Check mesh resolution convergence
python tests/test_validate_resolution.py
```

## Project Structure

```
project-name/
├── README.md                      # This file
├── requirements.txt               # Dependencies
├── environment.yml                # Conda environment
├── .gitignore
│
├── data/
│   └── README.md                  # Data download/preparation
│
├── common/                        # Shared physics and utilities
│   ├── __init__.py
│   ├── pcb_physics.py            # PCBDomain, boundary conditions, RHS
│   ├── data_loader.py            # Dataset generation
│   └── config.py                  # Shared hyperparameters
│
├── methods/
│   └── pcb_solver/               # Physics-based solver method
│       ├── __init__.py
│       ├── solver.py              # PCB_solver_main implementation
│       ├── test_cases.py          # PCB_case_1, PCB_case_2
│       ├── train.py               # Training/example entry point
│       ├── config.yaml            # Method-specific config
│       └── README.md
│
├── experiments/
│   ├── run_experiment.py         # Unified CLI entry point
│   └── benchmark.py               # Comparison script (future)
│
├── tests/
│   ├── test_validate_resolution.py  # Mesh convergence study
│   └── test_interface.py            # Interface conformance (future)
│
├── results/                       # Gitignored outputs
│   └── README.md
│
└── notebooks/
    └── exploration/              # Per-person exploratory notebooks
```

## Physics Model

### Problem
2D transient heat equation on square PCB with:
- **Conduction**: diffusive heat spreading (k = 15 W/(m·K) default)
- **Radiation**: Stefan-Boltzmann cooling (ε = 0.8 default)
- **Boundary conditions**: Dirichlet at corners + adiabatic elsewhere
- **Heat sources**: Resistive heaters as spatially-distributed patches

### Discretization
**Spatial (FD)**: Regular nx × ny grid with node numbering `id = i + nx*j`

**Temporal (transient)**: Adaptive ODE integration (RK45 with Radau fallback)

**Capacitance**: Accounts for boundary cell fractions:
- Interior: C_ij = ρ·c_p · dx·dy·t
- Edge: C_ij = ρ·c_p · (dx/2)·dy·t or dx·(dy/2)·t
- Corner: C_ij = ρ·c_p · (dx/2)·(dy/2)·t

### Key Parameters
- Domain size: 0.1 m × 0.1 m
- PCB thickness: 0.001 m (1 mm)
- Material: Cu PCB (k=15 W/(m·K), ρ·c_p=2.43e6 J/(m³·K))
- Environment: 250 K (default), can vary
- Heater sizes: Physical units (e.g., 10 mm × 10 mm), resolution-independent

## Usage Examples

### Single Test Case
```python
from methods.pcb_solver.test_cases import PCB_case_1, PCB_case_2

# Steady-state
T, interfaces, heaters = PCB_case_1(display=True)

# Transient
T_traj, t_array, _, _ = PCB_case_2(
    solver='transient', time=10.0, dt=0.1, display=True
)
```

### Dataset Generation
```python
from common.data_loader import generate_dataset

dataset = generate_dataset(
    n_samples=100,
    time=10.0,
    dt=0.1,
    m=3,                          # mesh refinement
    Tenv_range=(250, 350),        # environment temp [K]
    Q_range=(0.1, 5.0),           # heater power [W]
    T_bc_range=(250, 350),        # interface temp [K]
    uniform_time_points=True,
    n_time_samples=100
)

# Each sample contains: Q, T_bc, Tenv, T_init, t, T (trajectory), X (coordinates)
```

### CLI Usage
```bash
# Steady-state with interface temps = 300 K
python experiments/run_experiment.py --experiment steady \
  --T_interfaces 300 300 300 300 --display

# Transient with custom physics
python experiments/run_experiment.py --experiment transient \
  --board_k 20 --time 5.0 --dt 0.05 --display

# Dataset: 50 samples, spatially-varying initial conditions
python experiments/run_experiment.py --experiment dataset \
  --n_samples 50 --time 10.0 --T_init_spatial --verbose
```

## Convergence & Validation

**Mesh resolution convergence** is validated in `tests/test_validate_resolution.py`:
- Runs transient solver at nx = [20, 40, 100, 200]
- Compares T_center(t) curves
- Plots final temperature fields
- Checks convergence (differences < 0.01 K)

Run: `python tests/test_validate_resolution.py`

## Method Development

To add a new method (e.g., Neural ODE model):
1. Create folder `methods/neural_ode/`
2. Inherit from base interface (TBD: `common.BaseModel`)
3. Use `common.pcb_physics` for physics evaluation
4. Use `common.data_loader.generate_dataset` for training data
5. Add method-specific `config.yaml` and `README.md`

See `methods/pcb_solver/README.md` for physics solver example.

## Configuration

### Global (shared physics)
Edit `common/config.py` for default parameters.

### Method-specific
Each method has `config.yaml` (e.g., `methods/pcb_solver/config.yaml`).

### Experiment CLI
Override any parameter via command-line flags:
```bash
python experiments/run_experiment.py --board_k 20 --Tenv 300 --time 5.0
```

## Development Notes

- **Circular imports**: `common/data_loader.py` imports from `methods/pcb_solver/test_cases.py`
  to use PCB_case_2 for dataset generation. Imported inside function to avoid load-time issues.
- **Resolution-independence**: All heater/domain sizes in physical metres, not grid cells.
  Validated by running same physical case at different resolutions.
- **Transient stiffness**: RK45 (explicit) attempts first; falls back to Radau (implicit) if needed.

## Dependencies

- **numpy** ≥ 1.20 — numerical arrays
- **scipy** — sparse matrices, ODE integration, linear algebra
- **matplotlib** — visualization
- **pyyaml** — configuration files

See `requirements.txt` for pinned versions.

## References

- Heat equation with radiation: https://en.wikipedia.org/wiki/Heat_equation
- Finite difference discretization: Strikwerda, "Finite Difference Schemes and Partial Differential Equations"
- Stefan-Boltzmann radiation: https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law

## License

[TBD]

## Contact

Ismael Gallo Lopez (ismael.gallo@tudelft.nl)
