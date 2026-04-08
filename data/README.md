# Data

This directory holds datasets for training and evaluation. It is gitignored.

## Dataset Generation

Training datasets are generated on-the-fly using `common.data_loader.generate_dataset()`:

```python
from common.data_loader import generate_dataset

dataset = generate_dataset(
    n_samples=100,
    time=10.0,
    dt=0.1,
    m=3,
    Tenv_range=(250, 350),
    Q_range=(0.1, 5.0),
    T_bc_range=(250, 350),
    T_init_range=(290, 310),
    uniform_time_points=True,
    n_time_samples=100,
    verbose=True
)
```

Or via CLI:
```bash
python experiments/run_experiment.py --experiment dataset \
  --n_samples 100 --time 10.0 --dt 0.1
```

## Dataset Format

Each sample is a dictionary:
```python
{
    'Q': [Q0, Q1, Q2, Q3],        # Heater powers [W]
    'T_bc': [T0, T1, T2, T3],     # Interface temperatures [K]
    'Tenv': Tenv,                 # Environment temperature [K]
    'T_init': T_init,             # Initial condition (scalar or array)
    't': t_array,                 # Time points
    'T': T_traj,                  # Trajectories: shape (n_steps, n_nodes)
    'X': X                        # Node coordinates: shape (n_nodes, 2)
}
```

## Custom Datasets

To use pre-computed or external datasets:
1. Save as `.npy` or `.npz` (numpy) or `.pt` (PyTorch)
2. Place in this directory
3. Write a loader in your method's `train.py`

## Storage

To save generated datasets:
```python
import numpy as np

# Save entire dataset list as npz
np.savez('pcb_dataset.npz', dataset=dataset)

# Load it back
loaded = np.load('pcb_dataset.npz', allow_pickle=True)
dataset = loaded['dataset'].tolist()
```
