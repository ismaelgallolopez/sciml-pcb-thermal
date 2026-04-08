# PCB Thermal Solver

Physics-based solver for transient heat transfer in square PCBs with radiative and conductive coupling.

## Overview

This solver implements a finite-difference discretization of the 2D heat equation on a PCB with:
- **Conductive coupling** between adjacent nodes (thermal conductivity)
- **Radiative coupling** to environment (Stefan-Boltzmann radiation)
- **Dirichlet boundary conditions** at corner interface nodes
- **Heat sources** from resistive heaters modeled as spatially-distributed patches

The solver supports both:
- **Steady-state**: Newton iteration for nonlinear radiative-conductive balance
- **Transient**: Adaptive ODE integration (RK45 with Radau fallback for stiff systems)

## Problem Formulation

### Spatial Discretization (FD)
Grid: nx × ny nodes on domain [0, L] × [0, L]

Node ordering: `id = i + nx*j` (x-fast, y-slow)

Capacitance matrix C accounts for boundary cell fractions:
- Corner nodes: C_ij = ρ·c_p · (dx/2)·(dy/2)·t
- Edge nodes: C_ij = ρ·c_p · (dx/2)·dy·t or dx·(dy/2)·t
- Interior nodes: C_ij = ρ·c_p · dx·dy·t

Conduction: K_ij = k · (face_area / |center_dist|)

Radiation: R_ij = ε · (surface area per face)

### Temporal Discretization (Transient)

dT/dt = (1/C) · [Q - K·T - σ·E·(T⁴ - T_env⁴)]

where σ = 5.67e-8 W/(m²·K⁴) (Stefan-Boltzmann constant)

## Usage

### Single Case
```python
from test_cases import PCB_case_1, PCB_case_2

# Steady-state
T, interfaces, heaters = PCB_case_1(
    L=0.1,           # board size [m]
    thickness=0.001, # [m]
    m=3,             # mesh refinement (nodes = (4*m+1)²)
    board_k=1,       # thermal conductivity [W/(m·K)]
    display=True
)

# Transient
T_traj, t_array, interfaces, heaters = PCB_case_2(
    solver='transient',
    L=0.1,
    time=10.0,  # [s]
    dt=0.1,     # max timestep [s]
    T_init=298.0,
    display=True
)
```

### Dataset Generation
```python
from common.data_loader import generate_dataset

dataset = generate_dataset(
    n_samples=100,
    time=10.0,
    dt=0.1,
    m=3,  # mesh refinement
    Tenv_range=(250, 350),
    Q_range=(0.1, 5.0),
    T_bc_range=(250, 350),
    uniform_time_points=True,
    n_time_samples=100
)

# Each sample contains:
# 'Q': heater powers [W]
# 'T_bc': interface temperatures [K]
# 'Tenv': environment temperature [K]
# 'T_init': initial condition
# 't': time array
# 'T': trajectory (n_steps, n_nodes)
# 'X': spatial coordinates (n_nodes, 2)
```

## Configuration

See `config.yaml` for method-specific parameters:
- Physics (material properties, emissivity)
- Mesh (refinement level, total nodes)
- Solver (tolerances, max iterations)
- Boundary conditions and heat sources
- Simulation parameters (time range, sampling)

## Implementation Details

### Steady-State Solver
Newton iteration with sparse matrix solves (UMFPACK).

### Transient Solver
- **Primary method**: RK45 (explicit, non-stiff problems)
- **Fallback**: Radau (implicit, stiff systems)
- **Dense output**: enables uniform time resampling for ML

### Boundary Conditions
- **Dirichlet (corner nodes)**: fixed temperature
- **Radiation edges**: exposed side faces radiate to T_env
- **Implicit BCs**: adiabatic elsewhere

## Files

- `solver.py` - Core `PCB_solver_main()` function
- `test_cases.py` - Test cases: `PCB_case_1()`, `PCB_case_2()`
- `train.py` - Example usage
- `config.yaml` - Method configuration

## Dependencies

- numpy, scipy (sparse matrices, ODE integration)
- matplotlib (visualization)

See root `requirements.txt` for full list.
