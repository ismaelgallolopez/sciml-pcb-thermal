"""Mesh resolution convergence validation."""

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '..')  # Add parent directory to path

from common.pcb_physics import PCBDomain, HeaterPatch, EdgeDirichletBC
from methods.pcb_solver.solver import PCB_solver_main


def run_convergence_study():
    """Run transient simulation at three resolutions and plot T_center vs time."""

    # Physical parameters (same for all resolutions)
    L = 0.1              # domain side length [m]
    thickness = 0.001    # PCB thickness [m]
    board_k = 15         # thermal conductivity [W/(m·K)]
    board_c = 900        # specific heat [J/(kg·K)]
    board_rho = 2700     # density [kg/m³]
    ir_emmisivity = 0.8  # infrared emissivity
    Tenv = 250.0         # environment temperature [K]
    T_bc_value = 250.0   # boundary condition temperature [K]
    Q_heater = 1.0       # heater power [W]
    time_final = 10.0    # simulation end time [s]
    dt = 0.1             # time step upper bound [s]
    T_init = 298.0       # initial uniform temperature [K]

    # EXPLICIT heater size: L/10 (10% of board side) - RESOLUTION INDEPENDENT
    heater_size = L / 10  # Fixed at 10 mm for L=100mm board

    # Three resolutions to test
    nx_values = [20, 40, 100, 200]
    results = {}

    print("Running convergence study...")
    print(f"Physical domain: L={L*1000:.0f}mm, thickness={thickness*1000:.2f}mm")
    print(f"Heater: size={heater_size*1000:.0f}mm × {heater_size*1000:.0f}mm at center")
    print(f"Heater power: Q={Q_heater:.2f} W")
    print(f"Boundary temperature: T_bc={T_bc_value:.1f} K")
    print(f"Environment: T_env={Tenv:.1f} K")
    print(f"Simulation time: {time_final:.1f} s\n")

    for nx in nx_values:
        ny = nx
        print(f"Solving for nx={nx}, ny={ny} ({nx*ny} nodes)...")

        dx = L / (nx - 1)
        dy = L / (ny - 1)
        n_nodes = nx * ny

        # Discretize domain
        domain = PCBDomain(Lx=L, Ly=L, thickness=thickness, k_xy=board_k,
                           rho_cp=board_rho * board_c, emissivity=ir_emmisivity)
        X, Y, C, K, R = domain.discretize(nx, ny)

        # Single heater at center with EXPLICIT fixed physical size
        heater_patch = HeaterPatch(L/2, L/2, heater_size, heater_size, Q_heater)
        Q_vec = heater_patch.apply_sources(X, Y, dx, dy)
        heaters = {i: Q_vec[i] for i in range(n_nodes) if Q_vec[i] > 0.0}

        # Use EdgeDirichletBC for all four edges
        interfaces_dict = {}
        for edge in ["left", "right", "bottom", "top"]:
            bc_edge = EdgeDirichletBC(edge, T_bc_value, L, L, nx, ny)
            interfaces_dict.update(bc_edge.as_interfaces_dict(X, Y))

        print(f"  Boundary nodes: {len(interfaces_dict)}")
        print(f"  Heater nodes: {len(heaters)} (heater physical size: {heater_size*1000:.0f}mm × {heater_size*1000:.0f}mm)")

        # Solve transient problem with uniform time resampling
        n_time_points = max(100, int(np.ceil(time_final / dt)))
        T_array, t_array = PCB_solver_main(
            solver='transient',
            Lx=L, Ly=L, thickness=thickness,
            nx=nx, ny=ny,
            board_k=board_k, board_c=board_c, board_rho=board_rho,
            ir_emmisivity=ir_emmisivity,
            Tenv=Tenv,
            interfaces=interfaces_dict,
            heaters=heaters,
            C=C, K_domain=K, R_domain=R,
            time=time_final,
            dt=dt,
            T_init=T_init,
            n_uniform_samples=n_time_points,
            display=False
        )

        # Extract temperature at board center
        center_idx = (nx // 2) + (ny // 2) * nx
        T_center = T_array[:, center_idx]

        results[nx] = {
            't': t_array,
            'T_center': T_center,
            'X': X,
            'Y': Y,
            'T_array': T_array
        }

        print(f"  T_center(t=0): {T_center[0]:.2f} K")
        print(f"  T_center(t={time_final}): {T_center[-1]:.2f} K\n")

    # Plot convergence
    print("Plotting convergence study...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Plot 1: Temperature vs time at board center for all resolutions
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(nx_values)))
    for nx, color in zip(nx_values, colors):
        t = results[nx]['t']
        T_c = results[nx]['T_center']
        ax.plot(t, T_c, color=color, linewidth=2, label=f'nx={nx} ({nx*nx} nodes)')

    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Temperature at board center [K]', fontsize=12)
    ax.set_title('Resolution Convergence Study\n(heater at center, T_bc=250K, T_env=250K)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Plot 2: Final temperature fields for all resolutions
    ax = axes[1]
    nx_finest = nx_values[-1]
    T_final = results[nx_finest]['T_array'][-1]
    X = results[nx_finest]['X']
    Y = results[nx_finest]['Y']

    psm = ax.scatter(X, Y, c=T_final, s=50, cmap='jet', alpha=0.7)
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title(f'Final Temperature Field (nx={nx_finest})', fontsize=12)
    ax.set_aspect('equal')
    cbar = plt.colorbar(psm, ax=ax)
    cbar.set_label('Temperature [K]', fontsize=11)

    plt.savefig('../results/convergence_study.png', dpi=150, bbox_inches='tight')
    print("Plot saved to 'results/convergence_study.png'")

    plt.show()

    # Print convergence analysis
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS")
    print("="*60)
    T_final_values = [results[nx]['T_center'][-1] for nx in nx_values]
    print(f"\nFinal T_center (at t={time_final} s):")
    for nx, T_f in zip(nx_values, T_final_values):
        print(f"  nx={nx:3d}: T_center={T_f:.4f} K")

    # Check convergence
    diffs = np.diff(T_final_values)
    print(f"\nDifferences between consecutive resolutions:")
    for i, (nx1, nx2, diff) in enumerate(zip(nx_values[:-1], nx_values[1:], diffs)):
        pct = 100 * abs(diff) / T_final_values[i]
        print(f"  nx={nx1} → nx={nx2}: ΔT={diff:+.6f} K ({pct:.3f}%)")

    convergence_ok = all(abs(d) < 0.01 for d in diffs)  # < 0.01 K difference
    if convergence_ok:
        print("\n✓ Convergence verified: differences < 0.01 K")
    else:
        print("\n✗ Not fully converged: differences >= 0.01 K (may be OK for coarse grids)")


if __name__ == '__main__':
    run_convergence_study()
