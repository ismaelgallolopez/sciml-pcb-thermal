"""PCB thermal solver implementation."""

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import colormaps

from common.pcb_physics import PCBDomain, RadiativeBC, pcb_rhs


def PCB_solver_main(solver: str, Lx: float, Ly: float, thickness: float, nx: int, ny: int, board_k: float, ir_emmisivity: float,
                    Tenv: float, interfaces: dict, heaters: dict, C: np.ndarray = None, K_domain: sparse.spmatrix = None, R_domain: sparse.spmatrix = None,
                    display: bool = False, maxiters: int = 1000, objtol: float = 0.01, board_c: float = 900, board_rho: float = 2700,
                    time: float = 0.0, dt: float = 0.0, T_init=298.0, n_uniform_samples: int = 0):
    '''
    PCB thermal solver for rectangular PCB in radiative environment at temperature Tenv.

    26---27---28---29---30
    |    |    |    |    |
    20---21---22---23---24
    |    |    |    |    |
    15---16---17---18---19
    |    |    |    |    |
    10---11---12---13---14
    |    |    |    |    |    y
    5----6----7----8----9    ^
    |    |    |    |    |    |
    0----1----2----3----4    ---> x

    Parameters:
    -----------
    solver : str
        'steady' or 'transient'
    Lx, Ly : float
        Domain dimensions [m]
    thickness : float
        PCB thickness [m]
    nx, ny : int
        Number of nodes
    board_k : float
        Thermal conductivity [W/(K·m)]
    ir_emmisivity : float
        Infrared emissivity
    Tenv : float
        Environment temperature [K]
    interfaces : dict
        {node_id: temperature [K]}
    heaters : dict
        {node_id: power [W]}
    C : np.ndarray
        Per-node lumped capacitance [J/K]
    K_domain, R_domain : sparse matrices
        Pre-computed domain matrices
    display : bool
        Display results
    maxiters : int
        Max iterations for steady-state
    objtol : float
        Tolerance for steady-state
    board_c : float
        Specific heat capacity [J/(kg·K)]
    board_rho : float
        Density [kg/m³]
    time : float
        End time for transient [s]
    dt : float
        Max time step [s]
    T_init : float or ndarray
        Initial condition
    n_uniform_samples : int
        Number of uniform time samples (transient only)

    Returns:
    --------
    T : ndarray
        Temperature field
    time_array : ndarray or None
        Time points (transient) or None (steady-state)
    '''

    n_nodes = nx * ny
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # Use provided domain matrices, or create them if needed (for backward compatibility)
    if C is None or K_domain is None or R_domain is None:
        domain = PCBDomain(Lx=Lx, Ly=Ly, thickness=thickness, k_xy=board_k,
                          rho_cp=board_rho * board_c, emissivity=ir_emmisivity)
        _, _, C, K_domain, R_domain = domain.discretize(nx, ny)

    # Adjust K and E matrices for interface boundary conditions
    # For steady-state: interface rows become identity (Dirichlet trick)
    # For transient: interface rows are zeroed out and dTdt is forced to 0 in RHS
    K = K_domain.tolil() if solver == 'steady' else K_domain.tolil()
    E = R_domain.tolil()

    if solver == 'steady':
        # Set interface rows to identity for Dirichlet boundary conditions
        for nid in interfaces:
            K[nid, :] = 0.0
            K[nid, nid] = 1.0
    else:
        # For transient: zero out interface rows (dTdt will be forced to 0 in RHS)
        for nid in interfaces:
            K[nid, :] = 0.0

    K = K.tocsr()
    E = E.tocsr()

    # Build heat source vector {Q}
    Q = np.zeros(n_nodes, dtype=np.double)
    for nid in range(n_nodes):
        if nid in interfaces:
            if solver == 'steady':
                Q[nid] = interfaces[nid]  # Dirichlet values for steady-state
            # else transient: leave Q[nid] = 0 (K row is zero and dTdt forced to 0)
        elif nid in heaters:
            Q[nid] = heaters[nid]

    # Solving nonlinear equation via Newton iteration (steady) or adaptive integration (transient)
    Boltzmann_cte = 5.67e-8
    tol = 100
    it = 0

    if solver == 'steady':
        if isinstance(T_init, (int, float)):
            T = np.full(n_nodes, T_init, dtype=np.double)
        elif isinstance(T_init, np.ndarray):
            T = T_init.copy()
        else:
            print("T_init must be a float or a numpy array.")
            exit(1)
        while tol > objtol and it < maxiters:
            b = Q - K.__matmul__(T) - Boltzmann_cte * E.__matmul__(T**4 - Tenv**4)
            A = K + 4 * Boltzmann_cte * E.multiply(T**3)
            dT = sparse.linalg.spsolve(A, b)
            T += dT
            tol = max(abs(dT))
            it = it + 1

        if tol > objtol:
            print("ERROR in PCB SOLVER MAIN. Convergence was not reached.")
            exit(1)

        if display:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
            psm = ax.pcolormesh(T.reshape(ny, nx), cmap=colormaps['jet'], rasterized=True, vmin=np.min(T), vmax=np.max(T))
            fig.colorbar(psm, ax=ax)
            plt.title('Temperature field')
            plt.show()
        return T, None

    elif solver == 'transient':
        if isinstance(T_init, (int, float)):
            T_init_vec = np.full(n_nodes, T_init, dtype=np.double)
        elif isinstance(T_init, np.ndarray):
            T_init_vec = T_init.copy()
        else:
            print("T_init must be a float or a numpy array.")
            exit(1)

        interface_nodes_id = np.array(list(interfaces.keys()))
        heater_nodes_id = np.array(list(heaters.keys()))

        # Set interface nodes to their boundary condition values
        for i_node in interfaces:
            T_init_vec[i_node] = interfaces[i_node]

        # Compute radiative forcing vector F_rad for environment radiation at T_env
        rad_bc = RadiativeBC(Tenv, ir_emmisivity)
        E_aug, F_rad = rad_bc.augment_radiation(E, nx, ny, dx, dy, thickness)

        # Adaptive integration with stiffness detection + fallback to Radau
        # First attempt with RK45 (explicit, good for non-stiff to mildly stiff problems)
        sol = solve_ivp(
            fun=lambda t, T: pcb_rhs(t, T, K, E_aug, Q, F_rad, interface_nodes_id, C),
            t_span=(0.0, time),
            y0=T_init_vec,
            method='RK45',
            dense_output=True,
            max_step=dt if dt > 0 else np.inf,
            rtol=1e-6,
            atol=1e-8
        )
        method = 'RK45'

        # Check integration status; fallback to Radau (implicit) if stiff
        if sol.status != 0:
            if display:
                print(f"RK45 returned status {sol.status}: {sol.message}. Retrying with Radau (implicit stiff solver)...")
            sol = solve_ivp(
                fun=lambda t, T: pcb_rhs(t, T, K, E_aug, Q, F_rad, interface_nodes_id, C),
                t_span=(0.0, time),
                y0=T_init_vec,
                method='Radau',
                dense_output=True,
                max_step=dt if dt > 0 else np.inf,
                rtol=1e-6,
                atol=1e-8
            )
            method = 'Radau'

        # Optionally resample to uniform time grid for consistent batching (e.g., for PyTorch DataLoader)
        if n_uniform_samples > 0:
            time_uniform = np.linspace(0.0, time, n_uniform_samples)
            T_uniform = sol.sol(time_uniform)  # Use dense output to evaluate at arbitrary times
            T_array = T_uniform.T  # Shape: (n_uniform_samples, n_nodes)
            time_array = time_uniform
        else:
            # Keep adaptive time points from integrator
            T_array = sol.y.T  # Shape: (n_timesteps, n_nodes)
            time_array = sol.t  # Shape: (n_timesteps,)

        T = T_array[-1]     # Final temperature state

        if display:
            fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
            psm = ax1.pcolormesh(T.reshape(ny, nx), cmap=colormaps['jet'], rasterized=True, vmin=np.min(T), vmax=np.max(T))
            fig1.colorbar(psm, ax=ax1)
            plt.title('Temperature field at time ' + str(time) + ' [s]')
            plt.show()

            fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
            ax2.plot(time_array, T_array[:, heater_nodes_id])
            plt.title(f"Heaters temperature (method: {method})")
            plt.xlabel("time [s]")
            plt.ylabel("temperature [K]")
            legend = list(heaters.keys())
            plt.legend([str(l) for l in legend])
            plt.show()

        return T_array, time_array
