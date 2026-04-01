#%%

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
import matplotlib.pyplot as plt
from matplotlib import colormaps
np.set_printoptions(threshold=sys.maxsize)


#####################################################################################################
####################################### ODE and RHS functions #######################################
#####################################################################################################

def pcb_rhs(t, T, K, E, Q, interface_nodes_id, board_c, board_rho, thickness, dx, dy, Tenv):
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
        Radiation coupling matrix.
    Q : np.ndarray
        Heat source vector (includes heater powers).
    interface_nodes_id : np.ndarray
        Array of node indices corresponding to boundary conditions (interfaces).
    board_c : float
        Specific heat capacity [J/(kg·K)].
    board_rho : float
        Density [kg/m³].
    thickness : float
        PCB thickness [m].
    dx : float
        Grid spacing in x-direction [m].
    dy : float
        Grid spacing in y-direction [m].
    Tenv : float
        Environment temperature [K].
    
    Returns:
    --------
    dTdt : np.ndarray
        Temperature time derivative at each node.
    """
    Boltzmann = 5.67e-8
    dTdt = (Q - K.dot(T) - Boltzmann * E.dot(T**4 - Tenv**4)) / (board_c * board_rho * thickness * dx * dy)
    dTdt[interface_nodes_id] = 0.0  # BCs are fixed: dT/dt = 0 at interface nodes
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
                    T_interfaces:list=[250,250,250,250],Q_heaters:list=[1.0,1.0,1.0,1.0],Tenv:float=250,display:bool = False):
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
    Variables de salida:
                        -- T (numpy.array con dimension n = nx*ny) = vector con las temperaturas de los nodos (más información mirar en la descripción de **PCB_solver_main()**).
                        -- interfaces (diccionario {key = id del nodo, value = temperatura del nodo [K]}) = temperatura de las interfaces.
                        -- heaters (diccionario {key = id del nodo, value = disipación del nodo [W]}) = potencia disipada por los heaters.
    """

    n = 4*m+1

    id_Qnodes = [int((n-1)/4+(n-1)/2*n),int((n-1)/2+(n-1)/4*n),int(3*(n-1)/4+(n-1)/2*n),int((n-1)/2+3*(n-1)/4*n)]
    heaters = {id_Qnodes[0]:Q_heaters[0],id_Qnodes[1]:Q_heaters[1],id_Qnodes[2]:Q_heaters[2],id_Qnodes[3]:Q_heaters[3]}

    id_inodes = [0,n-1,n*n-1,n*n-n]
    interfaces = {id_inodes[0]:T_interfaces[0],id_inodes[1]:T_interfaces[1],id_inodes[2]:T_interfaces[2],id_inodes[3]:T_interfaces[3]}

    T, _ = PCB_solver_main(solver='steady', Lx=L, Ly=L, thickness=thickness,nx=n,ny=n,board_k=board_k,ir_emmisivity=ir_emmisivity,
                    Tenv=Tenv,interfaces=interfaces,heaters=heaters, display=display)
    
    return T,interfaces,heaters


#####################################################################################################
######################################### PCB_case_2() ##############################################
#####################################################################################################

def PCB_case_2(solver: str = 'steady', L:float=0.1,thickness:float=0.001,m:int=3,board_k:float=15, board_c:float=900, board_rho: float = 2700, ir_emmisivity:float=0.8,
                    T_interfaces:list=[250,250,250,250],Q_heaters:list=[1.0,1.0,1.0,1.0],Tenv:float=250,display:bool = False, time:float = 0.0, dt:float = 0.0, T_init:float = 298.0, n_uniform_samples:int = 0):
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

    n = 4*m+1

    id_Qnodes = [int((n-1)/4+(n-1)/2*n),int((n-1)/2+(n-1)/4*n),int((n-1)/4+3*(n-1)/4*n),int(3*(n-1)/4+3*(n-1)/4*n)]
    heaters = {id_Qnodes[0]:Q_heaters[0],id_Qnodes[1]:Q_heaters[1],id_Qnodes[2]:Q_heaters[2],id_Qnodes[3]:Q_heaters[3]}

    id_inodes = [0,n-1,n*n-1,n*n-n]
    interfaces = {id_inodes[0]:T_interfaces[0],id_inodes[1]:T_interfaces[1],id_inodes[2]:T_interfaces[2],id_inodes[3]:T_interfaces[3]}

    T, time_array = PCB_solver_main(solver = solver, Lx=L, Ly=L, thickness=thickness,nx=n,ny=n,board_k=board_k, board_c=board_c, board_rho=board_rho, ir_emmisivity=ir_emmisivity,
                    Tenv=Tenv,interfaces=interfaces,heaters=heaters, display=display, time=time, dt=dt, T_init = T_init, n_uniform_samples=n_uniform_samples)
    
    return T, time_array, interfaces, heaters
    


#####################################################################################################
####################################### PCB_solver_main() ###########################################
#####################################################################################################

def PCB_solver_main(solver:str, Lx:float,Ly:float,thickness:float,nx:int,ny:int,board_k:float,  ir_emmisivity:float,
                    Tenv:float,interfaces:dict,heaters:dict, display:bool = False, maxiters:int = 1000, objtol:int = 0.01, board_c:float=900, board_rho: float = 2700, time:float = 0.0, dt:float = 0.0,T_init = 298.0,
                    n_uniform_samples:int = 0):
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
    
    n_nodes = nx*ny # número total de nodos

    # cálculo de los GLs y GRs
    dx = Lx/(nx-1)
    dy = Ly/(ny-1)
    GLx = thickness*board_k*dy/dx
    GLy = thickness*board_k*dx/dy
    GR = 2*dx*dy*ir_emmisivity

    # Generación de la matriz de acoplamientos conductivos [K]. 
    # Note: For steady-state, interface rows are set to identity (Dirichlet trick).
    #       For transient, they are zeroed out and boundary conditions enforced via dTdt=0.
    K_cols = []
    K_rows = []
    K_data = []
    for j in range(ny):
        for i in range(nx):
            id = i + nx*j
            if id in interfaces:
                if solver == 'steady':
                    # Steady-state: diagonal entries for Dirichlet boundary conditions
                    K_rows.append(id)
                    K_cols.append(id)
                    K_data.append(1)
                # else: transient case — skip (leave row as zero)
            else:
                GLii = 0
                if i+1 < nx:
                    K_rows.append(id)
                    K_cols.append(id+1)
                    K_data.append(-GLx)
                    GLii += GLx
                if i-1 >= 0:
                    K_rows.append(id)
                    K_cols.append(id-1)
                    K_data.append(-GLx)
                    GLii += GLx
                if j+1 < ny:
                    K_rows.append(id)
                    K_cols.append(id+nx)
                    K_data.append(-GLy)
                    GLii += GLy
                if j-1 >= 0:
                    K_rows.append(id)
                    K_cols.append(id-nx)
                    K_data.append(-GLy)
                    GLii += GLy
                K_rows.append(id)
                K_cols.append(id)
                K_data.append(GLii)
    K = sparse.csr_matrix((K_data,(K_rows,K_cols)),shape=(n_nodes,n_nodes))

    # Creación de la matriz de acoplamientos radiativos [E]
    E_data = []
    E_id = []
    for id in range(n_nodes):
        if id not in interfaces:
            E_id.append(id)
            E_data.append(GR)
    E = sparse.csr_matrix((E_data,(E_id,E_id)),shape=(n_nodes,n_nodes))

    # Creación del vector {Q}.
    # Note: For transient, interface rows of K are zero and dTdt is forced to 0,
    # so Q[interface_nodes] should be 0 to avoid confusion. For steady-state,
    # Q[interface_nodes] holds the Dirichlet values for the Newton solver.
    Q = np.zeros(n_nodes,dtype=np.double)
    for id in range(n_nodes):
        if id in interfaces:
            if solver == 'steady':
                Q[id] = interfaces[id]
            # else transient: leave Q[id] = 0 (K row is zero and dTdt is forced to 0)
        elif id in heaters:
            Q[id] = heaters[id]
    
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
        
        # Adaptive integration with stiffness detection + fallback to Radau
        # First attempt with RK45 (explicit, good for non-stiff to mildly stiff problems)
        sol = solve_ivp(
            fun=lambda t, T: pcb_rhs(t, T, K, E, Q, interface_nodes_id, board_c, board_rho, thickness, dx, dy, Tenv),
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
                fun=lambda t, T: pcb_rhs(t, T, K, E, Q, interface_nodes_id, board_c, board_rho, thickness, dx, dy, Tenv),
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

def generate_dataset(n_samples, time, dt, L=0.1, m=3, board_k=15, board_c=900, board_rho=2700, 
                     ir_emmisivity=0.8, Tenv_range=(250, 350), Q_range=(0.1, 5.0), 
                     T_bc_range=(250, 350), T_init_range=(290, 310), T_init_spatial=False,
                     return_coordinates=True, uniform_time_points=True, n_time_samples=0, verbose=True):
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
            thickness=0.001,
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
            n_uniform_samples=n_uniform
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
