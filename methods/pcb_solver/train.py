"""Training and example script for PCB solver."""

import numpy as np
from test_cases import PCB_case_1, PCB_case_2
from common.data_loader import generate_dataset


def main_example():
    """Run example scenarios."""
    print("PCB Solver - Example Usage\n")

    # Example 1: Single steady-state solution
    print("Example 1: Steady-state solution...")
    T1, interfaces1, heaters1 = PCB_case_1(display=False)
    print(f"  Temperature range: {np.min(T1):.2f} K - {np.max(T1):.2f} K")

    # Example 2: Single transient solution
    print("\nExample 2: Transient solution...")
    T_init_random = np.random.uniform(290, 310, 169)
    T2, time2, interfaces2, heaters2 = PCB_case_2(
        solver='transient', display=False, time=10.0, dt=0.1, T_init=T_init_random
    )
    print(f"  Time points: {len(time2)}")
    print(f"  Final temperature range: {np.min(T2[-1]):.2f} K - {np.max(T2[-1]):.2f} K")

    # Example 3: Generate training dataset for Neural ODE / PINN
    print("\nExample 3: Dataset generation...")
    dataset = generate_dataset(n_samples=5, time=10.0, dt=0.1, m=3, verbose=True)
    print(f"  Dataset size: {len(dataset)} samples")
    print(f"  Sample keys: {dataset[0].keys()}")


if __name__ == '__main__':
    main_example()
