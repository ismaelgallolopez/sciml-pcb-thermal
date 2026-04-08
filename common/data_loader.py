"""Dataset generation and loading utilities."""

import numpy as np
from .pcb_physics import get_node_coordinates


def generate_dataset(n_samples, time, dt, L=0.1, thickness=0.001, m=3, board_k=15, board_c=900, board_rho=2700,
                     ir_emmisivity=0.8, Tenv_range=(250, 350), Q_range=(0.1, 5.0),
                     T_bc_range=(250, 350), T_init_range=(290, 310), T_init_spatial=False,
                     return_coordinates=True, uniform_time_points=True, n_time_samples=0, verbose=True, heater_size=None):
    """
    Generate a diverse dataset for training Neural ODEs or PINNs by varying:
    - Heater powers (Q_heaters)
    - Interface temperatures (T_interfaces)
    - Environment temperature (Tenv)
    - Initial temperature field (T_init)

    Parameters:
    -----------
    n_samples : int
        Number of trajectories to generate.
    time : float
        Simulation end time [s].
    dt : float
        Upper bound on integration step size [s]. If dt=0, solver chooses adaptively.
    L : float
        PCB side length [m].
    thickness : float
        PCB thickness [m].
    m : int
        Mesh refinement factor (total nodes = (4*m+1)²).
    board_k, board_c, board_rho : float
        Material properties.
    ir_emmisivity : float
        Infrared emissivity.
    Tenv_range : tuple
        (min, max) for uniform sampling of Tenv [K].
    Q_range : tuple
        (min, max) for uniform sampling of each heater power [W].
    T_bc_range : tuple
        (min, max) for uniform sampling of interface temperatures [K].
    T_init_range : tuple
        (min, max) for uniform sampling of initial temperature field [K].
    T_init_spatial : bool
        If False (default): uniform initial field — T_init is a scalar.
        If True: spatially-varying initial field — T_init is sampled per node for richer variation.
    return_coordinates : bool
        If True, include spatial node coordinates X in each sample.
    uniform_time_points : bool
        If True (default): resample trajectories to uniform time grid using dense output.
        If False: keep adaptive time points from integrator (ragged arrays).
    n_time_samples : int
        Number of uniform time points. If 0, auto-set to ceil(time / dt) if dt > 0 else 100.
    verbose : bool
        If True, print progress.
    heater_size : float or None
        Physical size of heater patches [m]. If None, defaults to L/10.

    Returns:
    --------
    dataset : list of dicts
        Each dict contains:
        {   'Q': [Q0, Q1, Q2, Q3],              # Heater powers [W]
            'T_bc': [T0, T1, T2, T3],          # Interface temperatures [K]
            'Tenv': Tenv,                      # Environment temperature [K]
            'T_init': T_init,                  # Initial temp field (scalar or array) [K]
            't': t_array,                      # Time points (uniform or adaptive)
            'T': T_traj,                       # Trajectories: shape (n_steps, n_nodes)
            'X': X (optional)                  # Node coordinates: shape (n_nodes, 2)
        }
    """
    # Avoid circular import: import here
    from methods.pcb_solver.test_cases import PCB_case_2

    dataset = []
    n_nodes = (4*m + 1)**2

    # Set heater size: default to L/10 (10% of board side length, independent of mesh)
    if heater_size is None:
        heater_size = L / 10

    # Auto-compute n_time_samples if uniform_time_points is True and n_time_samples is 0
    if uniform_time_points and n_time_samples <= 0:
        n_time_samples = max(10, int(np.ceil(time / dt)) if dt > 0 else 100)

    for sample_idx in range(n_samples):
        if verbose and (sample_idx + 1) % max(1, n_samples // 10) == 0:
            print(f"Generating sample {sample_idx + 1}/{n_samples}...")

        # Random parameters
        Q_heaters = np.random.uniform(Q_range[0], Q_range[1], 4).tolist()
        T_interfaces = np.random.uniform(T_bc_range[0], T_bc_range[1], 4).tolist()
        Tenv = np.random.uniform(Tenv_range[0], Tenv_range[1])

        # Initial condition: uniform or spatially-varying
        if T_init_spatial:
            T_init = np.random.uniform(T_init_range[0], T_init_range[1], n_nodes)
        else:
            T_init = np.random.uniform(T_init_range[0], T_init_range[1])  # scalar → uniform field

        # Solve transient problem
        # Pass n_uniform_samples to trigger uniform time resampling
        n_uniform = n_time_samples if uniform_time_points else 0
        T_traj, t_arr, interfaces_dict, heaters_dict = PCB_case_2(
            solver='transient',
            L=L,
            thickness=thickness,
            m=m,
            board_k=board_k,
            board_c=board_c,
            board_rho=board_rho,
            ir_emmisivity=ir_emmisivity,
            T_interfaces=T_interfaces,
            Q_heaters=Q_heaters,
            Tenv=Tenv,
            time=time,
            dt=dt,
            T_init=T_init,
            display=False,
            n_uniform_samples=n_uniform,
            heater_size=heater_size
        )

        sample = {
            'Q': Q_heaters,
            'T_bc': T_interfaces,
            'Tenv': Tenv,
            'T_init': T_init,
            't': t_arr,
            'T': T_traj          # Shape: (n_steps, n_nodes)
        }

        # Optionally include spatial coordinates (useful for PINNs)
        if return_coordinates:
            n = 4*m + 1
            X = get_node_coordinates(n, n, L, L)
            sample['X'] = X

        dataset.append(sample)

    if verbose:
        print(f"Dataset generation complete. Total samples: {n_samples}")
        if uniform_time_points:
            print(f"  All trajectories resampled to {n_time_samples} uniform time points.")
        else:
            print(f"  Trajectories use adaptive time points (ragged arrays for batching).")

    return dataset
