#%%

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import sys
import matplotlib.pyplot as plt
from matplotlib import colormaps
np.set_printoptions(threshold=sys.maxsize)


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

    T = PCB_solver_main(Lx=L, Ly=L, thickness=thickness,nx=n,ny=n,board_k=board_k,ir_emmisivity=ir_emmisivity,
                    Tenv=Tenv,interfaces=interfaces,heaters=heaters, display=display)
    
    return T,interfaces,heaters


#####################################################################################################
######################################### PCB_case_2() ##############################################
#####################################################################################################

def PCB_case_2(solver: str = 'steady', L:float=0.1,thickness:float=0.001,m:int=3,board_k:float=15, board_c:float=900, board_rho: float = 2700, ir_emmisivity:float=0.8,
                    T_interfaces:list=[250,250,250,250],Q_heaters:list=[1.0,1.0,1.0,1.0],Tenv:float=250,display:bool = False, time:float = 0.0, dt:float = 0.0, T_init:float = 298.0):
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
                    Tenv=Tenv,interfaces=interfaces,heaters=heaters, display=display, time=time, dt=dt, T_init = T_init)
    
    return T, time_array, interfaces, heaters
    


#####################################################################################################
####################################### PCB_solver_main() ###########################################
#####################################################################################################

def PCB_solver_main(solver:str, Lx:float,Ly:float,thickness:float,nx:int,ny:int,board_k:float,  ir_emmisivity:float,
                    Tenv:float,interfaces:dict,heaters:dict, display:bool = False, maxiters:int = 1000, objtol:int = 0.01, board_c:float=900, board_rho: float = 2700, time:float = 0.0, dt:float = 0.0,T_init = 298.0):
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
    K_cols = []
    K_rows = []
    K_data = []
    for j in range(ny):
        for i in range(nx):
            id = i + nx*j
            if id in interfaces:
                K_rows.append(id)
                K_cols.append(id)
                K_data.append(1)
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
    Q = np.zeros(n_nodes,dtype=np.double)
    for id in range(n_nodes):
        if id in interfaces:
            Q[id] = interfaces[id]
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
        alpha = board_k/(board_rho*board_c) # thermal difusivity.
        
        if alpha*dt/(0.5*(dx+dy))**2 > 0.5: # stability criterion for Euler scheeme. --> the criterion can be found here (eq 11): https://en.wikipedia.org/wiki/Von_Neumann_stability_analysis
            print("Excesive timestep size. Euler method becomes unstable.")
            exit(1)
        if isinstance(T_init, float) == True:
            T = np.full(n_nodes, T_init, dtype=np.double)
        elif isinstance(T_init, np.ndarray) == True: 
            T = T_init.copy()
        else:
            print("T_init must be a float or a numpy array.")
            exit(1)
        interface_nodes_id = np.array(list(interfaces.keys()))
        heater_nodes_id = np.array(list(heaters.keys()))
        
        for i_node in interfaces: # changing the temperature of the interface nodes from T_init to the proper temperature of the interface.
            T[i_node] = interfaces[i_node]
                        
        T_array = np.expand_dims(T.copy(), axis=0)
        time_array = np.array([0.0])
        ti = 0.0
        
        while ti < time:
            ti += dt
            dTdt = (Q - K.__matmul__(T) - Boltzmann_cte * E.__matmul__(T**4 - Tenv**4)) / (board_c * board_rho * thickness * dx * dy)
            dTdt[interface_nodes_id] = 0 # cancel de derivative term in the interface nodes. 
            dT = dTdt * dt
            T += dT
            T_array = np.concatenate((T_array, np.expand_dims(T.copy(), axis=0)), axis=0)
            time_array = np.append(time_array, ti)


        if display == True:
            fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
            psm = ax1.pcolormesh(T.reshape(ny,nx), cmap=colormaps['jet'], rasterized=True, vmin=np.min(T), vmax=np.max(T))
            fig1.colorbar(psm, ax=ax1)
            plt.title('Temperature field at time '+str(time)+' [s]')
            plt.show()

            fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
            ax2.plot(time_array,T_array[:,heater_nodes_id])
            plt.title("heaters temperature")
            plt.xlabel("time [s]")
            plt.ylabel("temperature [K]")
            legend = list(heaters.keys())
            plt.legend(legend)
            plt.show()

        return T_array, time_array

#%%

######################################################################################## 
################# EJEMPLO DE USO CON LOS VALORES PREDETERMINADOS #######################
######################################################################################## 
# T_init_random = np.random.uniform(70, 320, 169)
# T1,time1,interfaces1,heaters1 = PCB_case_2(solver='steady',display=True)
# T2, time2, interfaces2, heaters2 = PCB_case_2(solver = 'transient', display=True, time = 1000, dt = 1, T_init = T_init_random)
# %%
