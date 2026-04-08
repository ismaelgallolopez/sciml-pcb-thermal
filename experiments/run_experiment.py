"""Unified experiment runner."""

import argparse
import sys
import numpy as np

sys.path.insert(0, '..')  # Add parent directory to path

from methods.pcb_solver.test_cases import PCB_case_1, PCB_case_2
from common.data_loader import generate_dataset


def run_steady_state(args):
    """Run steady-state test case."""
    print("Running steady-state test case...")
    T, interfaces, heaters = PCB_case_1(
        L=args.L,
        thickness=args.thickness,
        m=args.m,
        board_k=args.board_k,
        ir_emmisivity=args.ir_emissivity,
        T_interfaces=args.T_interfaces,
        Q_heaters=args.Q_heaters,
        Tenv=args.Tenv,
        display=args.display,
    )
    print(f"Temperature range: {np.min(T):.2f} K - {np.max(T):.2f} K")
    return T


def run_transient(args):
    """Run transient test case."""
    print("Running transient test case...")
    T_traj, t_array, interfaces, heaters = PCB_case_2(
        solver='transient',
        L=args.L,
        thickness=args.thickness,
        m=args.m,
        board_k=args.board_k,
        board_c=args.board_c,
        board_rho=args.board_rho,
        ir_emmisivity=args.ir_emissivity,
        T_interfaces=args.T_interfaces,
        Q_heaters=args.Q_heaters,
        Tenv=args.Tenv,
        display=args.display,
        time=args.time,
        dt=args.dt,
        T_init=args.T_init,
    )
    print(f"Time steps: {len(t_array)}")
    print(f"Final temperature range: {np.min(T_traj[-1]):.2f} K - {np.max(T_traj[-1]):.2f} K")
    return T_traj, t_array


def run_dataset(args):
    """Generate dataset."""
    print(f"Generating dataset ({args.n_samples} samples)...")
    dataset = generate_dataset(
        n_samples=args.n_samples,
        time=args.time,
        dt=args.dt,
        L=args.L,
        thickness=args.thickness,
        m=args.m,
        board_k=args.board_k,
        board_c=args.board_c,
        board_rho=args.board_rho,
        ir_emmisivity=args.ir_emissivity,
        Tenv_range=tuple(args.Tenv_range),
        Q_range=tuple(args.Q_range),
        T_bc_range=tuple(args.T_bc_range),
        T_init_range=tuple(args.T_init_range),
        T_init_spatial=args.T_init_spatial,
        return_coordinates=True,
        uniform_time_points=True,
        n_time_samples=args.n_time_samples,
        verbose=True,
    )
    print(f"Dataset generation complete: {len(dataset)} samples")
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="PCB Thermal Solver - Unified Experiment Runner"
    )

    # Common arguments
    parser.add_argument(
        '--experiment', type=str, default='steady',
        choices=['steady', 'transient', 'dataset'],
        help='Experiment type to run'
    )
    parser.add_argument('--display', action='store_true', help='Display plots')
    parser.add_argument('--L', type=float, default=0.1, help='PCB side length [m]')
    parser.add_argument('--thickness', type=float, default=0.001, help='PCB thickness [m]')
    parser.add_argument('--m', type=int, default=3, help='Mesh refinement factor')
    parser.add_argument('--board_k', type=float, default=15, help='Thermal conductivity [W/(m·K)]')
    parser.add_argument('--board_c', type=float, default=900, help='Specific heat [J/(kg·K)]')
    parser.add_argument('--board_rho', type=float, default=2700, help='Density [kg/m³]')
    parser.add_argument('--ir_emissivity', type=float, default=0.8, help='IR emissivity')
    parser.add_argument('--Tenv', type=float, default=250, help='Environment temp [K]')
    parser.add_argument('--T_interfaces', type=float, nargs=4, default=[250, 250, 250, 250],
                        help='Interface temperatures [K]')
    parser.add_argument('--Q_heaters', type=float, nargs=4, default=[1.0, 1.0, 1.0, 1.0],
                        help='Heater powers [W]')

    # Transient-specific arguments
    parser.add_argument('--time', type=float, default=10.0, help='Simulation time [s]')
    parser.add_argument('--dt', type=float, default=0.1, help='Max timestep [s]')
    parser.add_argument('--T_init', type=float, default=298.0, help='Initial temperature [K]')

    # Dataset-specific arguments
    parser.add_argument('--n_samples', type=int, default=10, help='Number of dataset samples')
    parser.add_argument('--n_time_samples', type=int, default=100, help='Number of time samples per trajectory')
    parser.add_argument('--Tenv_range', type=float, nargs=2, default=[250, 350],
                        help='Environment temperature range [K]')
    parser.add_argument('--Q_range', type=float, nargs=2, default=[0.1, 5.0],
                        help='Heater power range [W]')
    parser.add_argument('--T_bc_range', type=float, nargs=2, default=[250, 350],
                        help='Interface temperature range [K]')
    parser.add_argument('--T_init_range', type=float, nargs=2, default=[290, 310],
                        help='Initial temperature range [K]')
    parser.add_argument('--T_init_spatial', action='store_true',
                        help='Use spatially-varying initial conditions')

    args = parser.parse_args()

    if args.experiment == 'steady':
        run_steady_state(args)
    elif args.experiment == 'transient':
        run_transient(args)
    elif args.experiment == 'dataset':
        run_dataset(args)


if __name__ == '__main__':
    main()
