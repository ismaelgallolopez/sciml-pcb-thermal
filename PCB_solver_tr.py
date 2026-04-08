#%%

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
import matplotlib.pyplot as plt
from matplotlib import colormaps
np.set_printoptions(threshold=sys.maxsize)


# ---------------------------------------------------------------------------
# Resolution-independent physical abstractions
# ---------------------------------------------------------------------------

class PCBDomain:
    """PCB physical parameters; produces node coordinates and coupling matrices."""

    def __init__(self, Lx: float, Ly: float, thickness: float,
                 k_xy: float, rho_cp: float, emissivity: float):
        self.Lx = Lx
        self.Ly = Ly
        self.thickness = thickness
        self.k_xy = k_xy        # thermal conductivity [W/(m·K)]
        self.rho_cp = rho_cp    # volumetric heat capacity ρ·c_p [J/(m³·K)]
        self.emissivity = emissivity

    def discretize(self, nx: int, ny: int):
        """
        Discretise on an nx×ny regular grid.
        Node ordering: id = i + nx*j  (x-index fast, y-index slow).

        Returns
        -------
        X : (n,)     x-coordinates of node centres [m]
        Y : (n,)     y-coordinates of node centres [m]
        C : (n,)     lumped capacitance accounting for boundary node cell fractions [J/K]
                     Corner nodes: (dx/2)·(dy/2)·thickness
                     Edge nodes: (dx/2)·dy·thickness or dx·(dy/2)·thickness
                     Interior nodes: dx·dy·thickness
        K : sparse   conductive coupling K_ij = k · (face_area / centre_dist) [W/K]
        R : sparse   radiative factor    R_ij = ε · 2 · dx · dy               [m²]
                     (multiply by σ_SB externally to get [W/K⁴·m²])
        """
        dx = self.Lx / (nx - 1)
        dy = self.Ly / (ny - 1)
        n = nx * ny

        i_idx = np.tile(np.arange(nx), ny)    # x-index for each node
        j_idx = np.repeat(np.arange(ny), nx)  # y-index for each node
        X = i_idx.astype(float) * dx
        Y = j_idx.astype(float) * dy

        # Compute per-node capacitance accounting for boundary cell fractions
        C = np.zeros(n)
        for j in range(ny):
            for i in range(nx):
                nid = i + nx * j
                # Determine cell width and height for this node
                width_frac = 1.0 if 0 < i < nx - 1 else 0.5  # Edge nodes own half-width
                height_frac = 1.0 if 0 < j < ny - 1 else 0.5  # Edge nodes own half-height
                cell_area = width_frac * height_frac * dx * dy
                C[nid] = self.rho_cp * cell_area * self.thickness

        GLx = self.thickness * self.k_xy * dy / dx  # k · (dy·t) / dx
        GLy = self.thickness * self.k_xy * dx / dy  # k · (dx·t) / dy
        rows, cols, data = [], [], []
        for j in range(ny):
            for i in range(nx):
                nid = i + nx * j
                diag = 0.0
                for di, dj, GL in ((1, 0, GLx), (-1, 0, GLx),
                                   (0, 1, GLy),  (0, -1, GLy)):
                    ni2, nj2 = i + di, j + dj
                    if 0 <= ni2 < nx and 0 <= nj2 < ny:
                        rows.append(nid)
                        cols.append(ni2 + nx * nj2)
                        data.append(-GL)
                        diag += GL
                rows.append(nid); cols.append(nid); data.append(diag)
        K = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

        GR = 2.0 * dx * dy * self.emissivity   # top + bottom face area × ε
        R = sparse.diags(np.full(n, GR), format='csr')

        return X, Y, C, K, R


class HeaterPatch:
    """Rectangular power source defined by physical centre, size, and total power."""

    def __init__(self, x_center: float, y_center: float,
                 width: float, height: float, Q_dot_W: float):
        self.x_center = x_center
        self.y_center = y_center
        self.width    = width
        self.height   = height
        self.Q_dot_W  = Q_dot_W

    def apply_sources(self, X: np.ndarray, Y: np.ndarray,
                      dx: float, dy: float) -> np.ndarray:
        """
        Return Q vector with Q_dot_W distributed by fractional cell-overlap area.
        Node at (x_i, y_i) owns cell [x_i ± dx/2, y_i ± dy/2].
        Overlap is computed in physical metres and normalised by the patch area
        (self.width × self.height), so Q.sum() == Q_dot_W whenever the patch
        lies entirely within the domain.
        """
        patch_area = self.width * self.height
        if patch_area <= 0.0:
            return np.zeros(len(X))
        x0 = self.x_center - self.width  / 2
        x1 = self.x_center + self.width  / 2
        y0 = self.y_center - self.height / 2
        y1 = self.y_center + self.height / 2
        # Overlap lengths in metres (not normalised by cell size)
        ux = np.clip(np.minimum(X + dx/2, x1) - np.maximum(X - dx/2, x0), 0.0, None)
        uy = np.clip(np.minimum(Y + dy/2, y1) - np.maximum(Y - dy/2, y0), 0.0, None)
        weights = ux * uy          # overlap area per cell [m²]
        Q = weights * (self.Q_dot_W / patch_area)
        # Verify power conservation when the patch is fully inside the domain
        if abs(weights.sum() - patch_area) < 1e-9 * patch_area:
            assert abs(Q.sum() - self.Q_dot_W) < 1e-6 * self.Q_dot_W, (
                f"HeaterPatch power not conserved: Q.sum()={Q.sum():.6e} W, "
                f"Q_dot_W={self.Q_dot_W:.6e} W")
        return Q


class DirichletBC:
    """Fixed-temperature (Dirichlet) boundary condition at selected nodes."""

    def __init__(self, T_fixed: dict):
        # T_fixed: {node_id (int): temperature [K]}
        self.T_fixed = dict(T_fixed)

    @classmethod
    def from_physical(cls, coord_T_pairs, X: np.ndarray, Y: np.ndarray):
        """
        Construct by snapping physical (x, y) coordinates to their nearest nodes.

        coord_T_pairs : iterable of ((x, y), T) tuples
        """
        T_fixed = {}
        for (xc, yc), T in coord_T_pairs:
            nid = int(np.argmin((X - xc)**2 + (Y - yc)**2))
            T_fixed[nid] = float(T)
        return cls(T_fixed)

    def interface_ids(self) -> np.ndarray:
        return np.array(list(self.T_fixed.keys()), dtype=int)

    def as_interfaces_dict(self) -> dict:
        return dict(self.T_fixed)


class EdgeDirichletBC:
    """Fixed-temperature boundary condition using geometric edge selection.
    
    Selects nodes by edge name using physical bounds:
      left:   X < dx/2
      right:  X > Lx - dx/2
      bottom: Y < dy/2
      top:    Y > Ly - dy/2
    
    Works identically at any (nx, ny) without nearest-neighbor snapping.
    """

    def __init__(self, edge: str, T_fixed: float, Lx: float, Ly: float, nx: int, ny: int):
        if edge not in ("left", "right", "bottom", "top"):
            raise ValueError(f"edge must be 'left', 'right', 'bottom', or 'top', got '{edge}'")
        self.edge = edge
        self.T_fixed_value = float(T_fixed)
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny

    def get_nodes_and_temps(self, X: np.ndarray, Y: np.ndarray) -> dict:
        dx = self.Lx / (self.nx - 1)
        dy = self.Ly / (self.ny - 1)
        T_fixed_dict = {}
        for nid in range(len(X)):
            x, y = X[nid], Y[nid]
            is_on_edge = False
            if self.edge == "left":
                is_on_edge = x < dx / 2
            elif self.edge == "right":
                is_on_edge = x > self.Lx - dx / 2
            elif self.edge == "bottom":
                is_on_edge = y < dy / 2
            elif self.edge == "top":
                is_on_edge = y > self.Ly - dy / 2
            if is_on_edge:
                T_fixed_dict[nid] = self.T_fixed_value
        return T_fixed_dict

    def as_interfaces_dict(self, X: np.ndarray, Y: np.ndarray) -> dict:
        return self.get_nodes_and_temps(X, Y)


class RadiativeBC:
    """
    Radiative boundary condition for edge nodes.
    Each edge node radiates to T_env through its exposed PCB side face.
    R_env = emissivity × exposed_face_area  (σ_SB is applied by the solver).
    Call augment_radiation() to fold this into the R matrix from PCBDomain.
    """

    def __init__(self, T_env: float, emissivity: float):
        self.T_env      = T_env
        self.emissivity = emissivity

    def augment_radiation(self, R: sparse.spmatrix,
                          nx: int, ny: int,
                          dx: float, dy: float,
                          thickness: float) -> tuple:
        """
        Add exposed-side-face radiative factors on the diagonal for edge nodes.
        Also compute the forcing vector F_rad for radiation FROM the environment.
        
        Returns
        -------
        R_aug : sparse matrix (CSR)
            Augmented radiative coupling matrix.
        F_rad : ndarray
            Forcing vector where F_rad[i] = R_env_i * T_env**4 for boundary nodes,
            0 elsewhere. Must be multiplied by sigma_SB in the solver.
        """
        R_lil = R.tolil()
        F_rad = np.zeros(nx * ny)
        for j in range(ny):
            for i in range(nx):
                nid = i + nx * j
                extra = 0.0
                if i == 0 or i == nx - 1:
                    extra += self.emissivity * dy * thickness
                if j == 0 or j == ny - 1:
                    extra += self.emissivity * dx * thickness
                if extra > 0.0:
                    R_lil[nid, nid] = R_lil[nid, nid] + extra
                    F_rad[nid] = extra * (self.T_env ** 4)
        return R_lil.tocsr(), F_rad


def initial_condition(X: np.ndarray, Y: np.ndarray, params) -> np.ndarray:
    """
    Sample an initial temperature field at node centres.

    Parameters
    ----------
    X, Y   : node centre coordinate arrays (same ordering as solver)
    params : float    → uniform field  T0[i] = params  for all i
             ndarray  → spatially varying field (must match len(X))
    """
    if np.isscalar(params):
        return np.full(len(X), float(params))
    arr = np.asarray(params, dtype=float)
    if arr.shape != X.shape:
        raise ValueError(
            f"initial_condition: params shape {arr.shape} != node array shape {X.shape}")
    return arr.copy()


#####################################################################################################
####################################### ODE and RHS functions #######################################
#####################################################################################################

def pcb_rhs(t, T, K, E, Q, F_rad, interface_nodes_id, C):
    """
    Compute dT/dt for all nodes — the right-hand side of the heat ODE.
    This function is compatible with scipy.integrate.solve_ivp for adaptive time stepping.
    
    Parameters:
    -----------
    t : float
        Current time (required by solve_ivp interface).
    T : np.ndarray
        Temperature vector at current time.
    K : sparse matrix
        Conductivity coupling matrix.
    E : sparse matrix
        Radiation coupling matrix (augmented with environment radiation).
    Q : np.ndarray
        Heat source vector (includes heater powers).
    F_rad : np.ndarray
        Radiative forcing vector from environment: F_rad[i] = R_env_i * T_env**4.
        Multiplied by Boltzmann constant in this function.
    interface_nodes_id : np.ndarray
        Array of node indices corresponding to boundary conditions (interfaces).
    C : np.ndarray
        Per-node lumped capacitance [J/K], accounting for boundary cell fractions.
    
    Returns:
    --------
    dTdt : np.ndarray
        Temperature time derivative at each node.
    """
    Boltzmann = 5.67e-8
    dTdt = (Q - K.dot(T) - Boltzmann * E.dot(T**4) + Boltzmann * F_rad) / C
    dTdt[interface_nodes_id] = 0.0
    return dTdt


def get_node_coordinates(nx, ny, Lx, Ly):
    """
    Generate spatial coordinates for each node.
    Vectorized version for efficient computation on large meshes.
    
    Parameters:
    -----------
    nx, ny : int
        Number of nodes in x and y directions.
    Lx, Ly : float
        Domain dimensions [m].
    
    Returns:
    --------
    X : np.ndarray
        Shape (nx*ny, 2) array. X[i] = (x_coord, y_coord) for node i.
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    i_idx, j_idx = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    X = np.column_stack([i_idx.ravel() * dx, j_idx.ravel() * dy])
    return X


#####################################################################################################
######################################### PCB_case_1() ##############################################
#####################################################################################################

def PCB_case_1(L:float=0.1,thickness:float=0.001,m:int=3,board_k:float=1,ir_emmisivity:float=0.8,
                    T_interfaces:list=[250,250,250,250],Q_heaters:list=[1.0,1.0,1.0,1.0],Tenv:float=250,display:bool = False,
                    T_init:float=298.0):
    """
    Caso 1.
    PCB cuadrada de lado L con 4 heaters simétricamente colocados en coordenadas [(L/4,L/2),(L/2,L/4),(3*L/4,L/2),(L/2,3*L/4)]
    y con 4 nodos de interfaz situados en coordenadas [(0,0),(L,0),(L,L),(0,L)].
    Variables de entrada (unidades entre [], si no hay nada es adimensional):
                        -- L (int) = dimensiones de la placa. [m]
                        -- thickness (float) = espesor de la placa. [m]
                        -- m (int) = valor de refinamiento de malla. --> el número de nodos en x e y es n = 4*m+1. En el caso predeterminado son 12x12 nodos.
                        -- board_k (float) = conductividad térmica del material de la placa. [W/(K*m)]
                        -- ir_emmisivity (float) = emisividad infrarroja del recubrimiento óptico de la PCB (la pintura).
                        -- T_interfaces (lista de 4 elementos) = temperatura de las 4 interfaces. [K]
                        -- Q_heaters (lista de 4 elementos) = potencia disipada por los heaters. [W]
                        -- Tenv (float) = temperatura del entorno. [K]
                        -- display (bool) = mostrar las temperaturas.
                        -- T_init (float) = initial guess for the Newton solver [K].
    Variables de salida:
                        -- T (numpy.array con dimension n = nx*ny) = vector con las temperaturas de los nodos (más información mirar en la descripción de **PCB_solver_main()**).
                        -- interfaces (diccionario {key = id del nodo, value = temperatura del nodo [K]}) = temperatura de las interfaces.
                        -- heaters (diccionario {key = id del nodo, value = disipación del nodo [W]}) = potencia disipada por los heaters.
    """

    n = 4*m + 1
    dx = L / (n - 1)
    dy = L / (n - 1)

    # Discretise domain to get node coordinates and matrices
    domain = PCBDomain(Lx=L, Ly=L, thickness=thickness, k_xy=board_k,
                       rho_cp=1.0, emissivity=ir_emmisivity)
    X, Y, C, K, R = domain.discretize(n, n)

    # Heater patches at physical locations; one-cell size → all power to nearest node
    patches = [
        HeaterPatch(L/4,   L/2,   dx, dy, Q_heaters[0]),
        HeaterPatch(L/2,   L/4,   dx, dy, Q_heaters[1]),
        HeaterPatch(3*L/4, L/2,   dx, dy, Q_heaters[2]),
        HeaterPatch(L/2,   3*L/4, dx, dy, Q_heaters[3]),
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


#####################################################################################################
######################################### PCB_case_2() ##############################################
#####################################################################################################

def PCB_case_2(solver: str = 'steady', L:float=0.1, thickness:float=0.001, m:int=3, board_k:float=15, board_c:float=900, board_rho:float=2700, ir_emmisivity:float=0.8,
                    T_interfaces:list=[250,250,250,250], Q_heaters:list=[1.0,1.0,1.0,1.0], Tenv:float=250, display:bool=False, time:float=0.0, dt:float=0.0, T_init:float=298.0, n_uniform_samples:int=0, heater_size:float=None):
    """
    Caso 1
    PCB cuadrada de lado L con 4 heaters colocados en coordenadas [(L/4,L/2),(L/2,L/4),(L/4,3*L/4),(3*L/4,3*L/4)]
    y con 4 nodos de interfaz situados en coordenadas [(0,0),(L,0),(L,L),(0,L)].
    Variables de entrada (unidades entre [], si no hay nada es adimensional):
                        -- L (int) = dimensiones de la placa. [m]
                        -- thickness (float) = espesor de la placa. [m]
                        -- m (int) = valor de refinamiento de malla. --> el número de nodos en x e y es n = 4*m+1. En el caso predeterminado son 12x12 nodos.
                        -- board_k (float) = conductividad térmica del material de la placa. [W/(K*m)]
                        -- ir_emmisivity (float) = emisividad infrarroja del recubrimiento óptico de la PCB (la pintura). 
                        -- T_interfaces (lista de 4 elementos) = temperatura de las 4 interfaces (250 - 350 K). [K]
                        -- Q_heaters (lista de 4 elementos) = potencia disipada por los heaters (0.1 - 5.0 W). [W]
                        -- Tenv (float) = temperatura del entorno (250 - 350 K). [K]
                        -- display (bool) = mostrar las temperaturas.
    Variables de salida:
                        -- T (numpy.array con dimension n = nx*ny) = vector con las temperaturas de los nodos (más información mirar en la descripción de **PCB_solver_main()**).
                        -- interfaces (diccionario {key = id del nodo, value = temperatura del nodo [K]}) = temperatura de las interfaces.
                        -- heaters (diccionario {key = id del nodo, value = disipación del nodo [W]}) = potencia disipada por los heaters.
    """

    n = 4*m + 1
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
        HeaterPatch(L/4,   L/2,   heater_size, heater_size, Q_heaters[0]),
        HeaterPatch(L/2,   L/4,   heater_size, heater_size, Q_heaters[1]),
        HeaterPatch(L/4,   3*L/4, heater_size, heater_size, Q_heaters[2]),
        HeaterPatch(3*L/4, 3*L/4, heater_size, heater_size, Q_heaters[3]),
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
    


#####################################################################################################
####################################### PCB_solver_main() ###########################################
#####################################################################################################

def PCB_solver_main(solver:str, Lx:float, Ly:float, thickness:float, nx:int, ny:int, board_k:float, ir_emmisivity:float,
                    Tenv:float, interfaces:dict, heaters:dict, C:np.ndarray=None, K_domain:sparse.spmatrix=None, R_domain:sparse.spmatrix=None,
                    display:bool=False, maxiters:int=1000, objtol:float=0.01, board_c:float=900, board_rho:float=2700,
                    time:float=0.0, dt:float=0.0, T_init=298.0, n_uniform_samples:int=0):
    '''
    Función solver del problema de PCB rectangular en un entorno radiativo formado por un cuerpo negro a temperatura Tenv. 
    Los nodos van numerados siguiendo el esquema de la figura, los nodos se ordenan de forma creciente filas.

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

    Variables de entrada (unidades entre [], si no hay nada es adimensional):
                        -- Lx (int) = dimension x de la placa. [m]
                        -- Lx (int) = dimension y de la placa. [m]
                        -- thickness (float) = espesor de la placa. [m]
                        -- nx (int) = número de nodos en el eje x (en la figura de ejemplo son 5).
                        -- ny (int) = número de nodos en el eje y (en la figura de ejemplo son 6).
                        -- board_k (float) = conductividad térmica del material de la placa. [W/(K*m)]
                        -- ir_emmisivity (float) = emisividad infrarroja del recubrimiento óptico de la PCB (la pintura).
                        -- Tenv (float) = temperatura del entorno. [K]
                        -- interfaces (diccionario {key = id del nodo, value = temperatura del nodo [K]}) = temperatura de las interfaces.
                        -- heaters (diccionario {key = id del nodo, value = disipación del nodo [W]}) = potencia disipada por los heaters.
                        -- display (bool) = mostrar las temperaturas.
                        -- maxiters (int) = máximas iteraciones del solver. Mantener el valor predeterminado salvo si la convergencia es muy lenta (salta error en la linea 203). 
                        -- objtol (int) = tolerancia objetivo del solver. Mantener el valor predeterminado salvo si no se llega a convergencia (salta error en la linea 203).
                        -- dt (float) = upper bound on integration step size [s] for transient. If dt=0, adaptive stepping only (default). NOT a fixed time step.
                        -- n_uniform_samples (int) = For transient: if > 0, resample solution to n_uniform_samples uniform time points using dense output.
    Variables de salida:
                        -- T (numpy.array con dimension n = nx*ny) = vector con las temperaturas ordenadas como en la figura de ejemplo.
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
    
    # Resolución de la ecuación no lineal [K]{T} + Boltzmann_cte*[E]({T^4} - Tenv^4) = {Q} 
    # mediante la resolución iterativa de la ecuación [A]{dT_i} = {b}, donde:
    #           -- [A] = [K] + 4*Boltzmann_cte*[E].*{T_i^3} (.* = multiplicación elemento a elemento)
    #           -- {b} = {Q} - [K]*{T_i} - [E]*({T_i^4}-Tenv^4)
    #           -- {T_i+1} = {T_i} + {dT_i}
            
    Boltzmann_cte = 5.67E-8
    tol = 100
    it = 0

    if solver == 'steady':
        if isinstance(T_init, float) == True:
            T = np.full(n_nodes, T_init, dtype=np.double)
        elif isinstance(T_init, np.ndarray) == True: 
            T = T_init.copy()
        else:
            print("T_init must be a float or a numpy array.")
            exit(1)
        while tol > objtol and it < maxiters:
            b = Q - K.__matmul__(T) - Boltzmann_cte * E.__matmul__(T**4-Tenv**4)
            A = K + 4 * Boltzmann_cte * E.multiply(T**3)
            dT = sparse.linalg.spsolve(A,b)
            T += dT
            tol = max(abs(dT))
            it = it+1

        if tol > objtol:
            print("ERROR in PCB SOLVER MAIN. Convergence was not reached.")
            exit(1)

        if display == True:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
            psm = ax.pcolormesh(T.reshape(ny,nx), cmap=colormaps['jet'], rasterized=True, vmin=np.min(T), vmax=np.max(T))
            fig.colorbar(psm, ax=ax)
            plt.title('Temperature field')
            plt.show()
        return T, None
    
    elif solver == 'transient':
        if isinstance(T_init, float) == True:
            T_init_vec = np.full(n_nodes, T_init, dtype=np.double)
        elif isinstance(T_init, np.ndarray) == True: 
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
        
        if display == True:
            fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
            psm = ax1.pcolormesh(T.reshape(ny,nx), cmap=colormaps['jet'], rasterized=True, vmin=np.min(T), vmax=np.max(T))
            fig1.colorbar(psm, ax=ax1)
            plt.title('Temperature field at time '+str(time)+' [s]')
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

#####################################################################################################
####################################### Dataset Generation ##########################################
#####################################################################################################

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


#%%

######################################################################################## 
################# EJEMPLO DE USO CON LOS VALORES PREDETERMINADOS #######################
######################################################################################## 

if __name__ == '__main__':
    # Example 1: Single steady-state solution
    # T1, interfaces1, heaters1 = PCB_case_1(display=True)

    # Example 2: Single transient solution
    # T_init_random = np.random.uniform(290, 310, 169)
    # T2, time2, interfaces2, heaters2 = PCB_case_2(solver='transient', display=True, time=10, dt=0.1, T_init=T_init_random)

    # Example 3: Generate training dataset for Neural ODE / PINN
    dataset = generate_dataset(n_samples=10, time=10.0, dt=0.1, m=3, verbose=True)
    # For each sample in dataset:
    #   - sample['T'] has shape (n_steps, n_nodes) — full trajectory
    #   - sample['X'] has shape (n_nodes, 2) — spatial coordinates
    #   - sample['t'], sample['Q'], sample['T_bc'], sample['Tenv'] specify the problem
# %%
