"""Shared configuration and hyperparameters."""

# Physics parameters
BOARD_K = 15  # thermal conductivity [W/(m·K)]
BOARD_C = 900  # specific heat capacity [J/(kg·K)]
BOARD_RHO = 2700  # density [kg/m³]
IR_EMISSIVITY = 0.8  # infrared emissivity

# Geometry
DOMAIN_SIZE = 0.1  # [m]
PCB_THICKNESS = 0.001  # [m]
MESH_REFINEMENT = 3  # m: total nodes = (4*m+1)²

# Solver parameters
STEADY_STATE_MAXITERS = 1000
STEADY_STATE_TOL = 0.01
TRANSIENT_METHOD_PRIMARY = 'RK45'
TRANSIENT_METHOD_FALLBACK = 'Radau'
TRANSIENT_RTOL = 1e-6
TRANSIENT_ATOL = 1e-8

# Environment
TENV_DEFAULT = 250  # [K]
TENV_RANGE = (250, 350)  # [K]

# Heater configuration
HEATER_POWER_RANGE = (0.1, 5.0)  # [W]
HEATER_SIZE_FRACTION = 0.1  # fraction of domain size

# Boundary conditions
BC_TEMP_RANGE = (250, 350)  # [K]

# Initial conditions
INITIAL_TEMP_RANGE = (290, 310)  # [K]

# Data generation
DATASET_DEFAULT_SAMPLES = 10
DATASET_TIME_RANGE = (1.0, 100.0)  # [s]
DATASET_DT = 0.1  # [s]
