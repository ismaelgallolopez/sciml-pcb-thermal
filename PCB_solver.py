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

def PCB_case_1(L:float=0.1,thickness:float=0.001,m:int=3,board_k:float=15,ir_emmisivity:float=0.8,
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

def PCB_case_2(L:float=0.1,thickness:float=0.001,m:int=3,board_k:float=15,ir_emmisivity:float=0.8,
                    T_interfaces:list=[250,250,250,250],Q_heaters:list=[1.0,1.0,1.0,1.0],Tenv:float=250,display:bool = False):
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

    T = PCB_solver_main(Lx=L, Ly=L, thickness=thickness,nx=n,ny=n,board_k=board_k,ir_emmisivity=ir_emmisivity,
                    Tenv=Tenv,interfaces=interfaces,heaters=heaters, display=display)
    
    return T,interfaces,heaters
    


#####################################################################################################
####################################### PCB_solver_main() ###########################################
#####################################################################################################

def PCB_solver_main(Lx:float,Ly:float,thickness:float,nx:int,ny:int,board_k:float,ir_emmisivity:float,
                    Tenv:float,interfaces:dict,heaters:dict, display:bool = False, maxiters:int = 1000, objtol:int = 0.01):
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
    T = np.full(n_nodes,Tenv,dtype=np.double)

    while tol > objtol and it < maxiters:
        b = Q-K.__matmul__(T) - Boltzmann_cte*E.__matmul__(T**4-Tenv**4)
        A = K+4*Boltzmann_cte*E.multiply(T**3)
        dT = sparse.linalg.spsolve(A,b)
        T += dT
        tol = max(abs(dT))
        it = it+1

    if tol > 0.01:
        print("ERROR in PCB SOLVER MAIN. Convergence was not reached.")
        exit(1)
    
    if display == True:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
        psm = ax.pcolormesh(T.reshape(ny,nx), cmap=colormaps['jet'], rasterized=True, vmin=np.min(T), vmax=np.max(T))
        fig.colorbar(psm, ax=ax)
        plt.title('Temperature field')
        plt.show()
    
    return T
#%%

######################################################################################## 
################# EJEMPLO DE USO CON LOS VALORES PREDETERMINADOS #######################
######################################################################################## 
T1,interfaces1,heaters1 = PCB_case_1(display=True)
T2,interfaces2,heaters2 = PCB_case_2(display=True)
# %%

T1

# %%
interfaces1
# %%
