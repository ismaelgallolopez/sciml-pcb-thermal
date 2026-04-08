"""PCB thermal physics: domains, boundary conditions, and ODE systems."""

import numpy as np
from scipy import sparse


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
