"""Test cases and example setups for PCB thermal solver."""

import numpy as np
from common.pcb_physics import PCBDomain, HeaterPatch, initial_condition
from .solver import PCB_solver_main


def PCB_case_1(L: float = 0.1, thickness: float = 0.001, m: int = 3, board_k: float = 1, ir_emmisivity: float = 0.8,
               T_interfaces: list = [250, 250, 250, 250], Q_heaters: list = [1.0, 1.0, 1.0, 1.0], Tenv: float = 250,
               display: bool = False, T_init: float = 298.0):
    """
    Test case 1: steady-state solution.
    Square PCB with 4 symmetric heaters at [(L/4,L/2),(L/2,L/4),(3*L/4,L/2),(L/2,3*L/4)]
    and 4 interface nodes at corners.

    Parameters:
    -----------
    L : float
        PCB side length [m]
    thickness : float
        PCB thickness [m]
    m : int
        Mesh refinement factor (nodes = 4*m+1)
    board_k : float
        Thermal conductivity [W/(K·m)]
    ir_emmisivity : float
        Infrared emissivity
    T_interfaces : list
        Corner temperatures [K]
    Q_heaters : list
        Heater powers [W]
    Tenv : float
        Environment temperature [K]
    display : bool
        Show results
    T_init : float
        Initial guess for Newton solver [K]

    Returns:
    --------
    T : ndarray
        Temperature field
    interfaces : dict
        {node_id: temperature}
    heaters : dict
        {node_id: power}
    """
    n = 4 * m + 1
    dx = L / (n - 1)
    dy = L / (n - 1)

    # Discretise domain to get node coordinates and matrices
    domain = PCBDomain(Lx=L, Ly=L, thickness=thickness, k_xy=board_k,
                       rho_cp=1.0, emissivity=ir_emmisivity)
    X, Y, C, K, R = domain.discretize(n, n)

    # Heater patches at physical locations; one-cell size → all power to nearest node
    patches = [
        HeaterPatch(L / 4, L / 2, dx, dy, Q_heaters[0]),
        HeaterPatch(L / 2, L / 4, dx, dy, Q_heaters[1]),
        HeaterPatch(3 * L / 4, L / 2, dx, dy, Q_heaters[2]),
        HeaterPatch(L / 2, 3 * L / 4, dx, dy, Q_heaters[3]),
    ]
    Q_vec = np.zeros(n * n)
    for p in patches:
        Q_vec += p.apply_sources(X, Y, dx, dy)
    heaters = {i: Q_vec[i] for i in range(n * n) if Q_vec[i] != 0.0}

    # Dirichlet BC: corner nodes assigned directly
    interfaces = {}
    interfaces[0] = T_interfaces[0]  # Bottom-left (0, 0)
    interfaces[n - 1] = T_interfaces[1]  # Bottom-right (n-1, 0)
    interfaces[n * (n - 1) + (n - 1)] = T_interfaces[2]  # Top-right (n-1, n-1)
    interfaces[n * (n - 1)] = T_interfaces[3]  # Top-left (0, n-1)

    # Initial condition sampled at node centres
    T0 = initial_condition(X, Y, T_init)

    T, _ = PCB_solver_main(solver='steady', Lx=L, Ly=L, thickness=thickness,
                           nx=n, ny=n, board_k=board_k, ir_emmisivity=ir_emmisivity,
                           Tenv=Tenv, interfaces=interfaces, heaters=heaters,
                           C=C, K_domain=K, R_domain=R, display=display, T_init=T0)

    return T, interfaces, heaters


def PCB_case_2(solver: str = 'steady', L: float = 0.1, thickness: float = 0.001, m: int = 3, board_k: float = 15,
               board_c: float = 900, board_rho: float = 2700, ir_emmisivity: float = 0.8,
               T_interfaces: list = [250, 250, 250, 250], Q_heaters: list = [1.0, 1.0, 1.0, 1.0], Tenv: float = 250,
               display: bool = False, time: float = 0.0, dt: float = 0.0, T_init: float = 298.0,
               n_uniform_samples: int = 0, heater_size: float = None):
    """
    Test case 2: steady or transient solution.
    Square PCB with 4 heaters at [(L/4,L/2),(L/2,L/4),(L/4,3*L/4),(3*L/4,3*L/4)]
    and 4 corner interface nodes.

    Parameters:
    -----------
    solver : str
        'steady' or 'transient'
    L : float
        PCB side length [m]
    thickness : float
        PCB thickness [m]
    m : int
        Mesh refinement factor
    board_k, board_c, board_rho : float
        Material properties
    ir_emmisivity : float
        Infrared emissivity
    T_interfaces : list
        Corner temperatures [K]
    Q_heaters : list
        Heater powers [W]
    Tenv : float
        Environment temperature [K]
    display : bool
        Show results
    time : float
        Simulation time [s] (transient only)
    dt : float
        Max time step [s]
    T_init : float or ndarray
        Initial condition
    n_uniform_samples : int
        Number of uniform time samples (transient)
    heater_size : float or None
        Physical heater patch size [m]

    Returns:
    --------
    T : ndarray
        Temperature field or trajectory
    time_array : ndarray or None
        Time points (transient) or None (steady)
    interfaces : dict
        {node_id: temperature}
    heaters : dict
        {node_id: power}
    """
    n = 4 * m + 1
    dx = L / (n - 1)
    dy = L / (n - 1)

    # Set heater size: default to L/10 (10% of board side length, independent of mesh)
    if heater_size is None:
        heater_size = L / 10

    # Discretise domain to get node coordinates and matrices
    domain = PCBDomain(Lx=L, Ly=L, thickness=thickness, k_xy=board_k,
                       rho_cp=board_rho * board_c, emissivity=ir_emmisivity)
    X, Y, C_domain, K_domain, R_domain = domain.discretize(n, n)

    # Heater patches at physical locations with fixed physical size (resolution-independent)
    patches = [
        HeaterPatch(L / 4, L / 2, heater_size, heater_size, Q_heaters[0]),
        HeaterPatch(L / 2, L / 4, heater_size, heater_size, Q_heaters[1]),
        HeaterPatch(L / 4, 3 * L / 4, heater_size, heater_size, Q_heaters[2]),
        HeaterPatch(3 * L / 4, 3 * L / 4, heater_size, heater_size, Q_heaters[3]),
    ]
    Q_vec = np.zeros(n * n)
    for p in patches:
        Q_vec += p.apply_sources(X, Y, dx, dy)
    heaters = {i: Q_vec[i] for i in range(n * n) if Q_vec[i] != 0.0}

    # Dirichlet BC: corner nodes assigned directly
    interfaces = {}
    interfaces[0] = T_interfaces[0]  # Bottom-left (0, 0)
    interfaces[n - 1] = T_interfaces[1]  # Bottom-right (n-1, 0)
    interfaces[n * (n - 1) + (n - 1)] = T_interfaces[2]  # Top-right (n-1, n-1)
    interfaces[n * (n - 1)] = T_interfaces[3]  # Top-left (0, n-1)

    # Initial condition sampled at node centres
    T0 = initial_condition(X, Y, T_init)

    T, time_array = PCB_solver_main(solver=solver, Lx=L, Ly=L, thickness=thickness,
                                    nx=n, ny=n, board_k=board_k, board_c=board_c, board_rho=board_rho,
                                    ir_emmisivity=ir_emmisivity, Tenv=Tenv, interfaces=interfaces,
                                    heaters=heaters, C=C_domain, K_domain=K_domain, R_domain=R_domain,
                                    display=display, time=time, dt=dt, T_init=T0,
                                    n_uniform_samples=n_uniform_samples)

    return T, time_array, interfaces, heaters
