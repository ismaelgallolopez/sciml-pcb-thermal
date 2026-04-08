# Results

This directory holds outputs from experiments. It is gitignored.

## Folder Structure

Organize outputs by method:
```
results/
├── pcb_solver/           # Physics solver results
│   ├── convergence_study.png
│   ├── case1_steady.npy
│   └── case2_transient.npy
├── neural_ode/           # Neural ODE method (future)
│   ├── training_loss.png
│   ├── model.pt
│   └── predictions.npy
└── pinn/                 # PINN method (future)
    ├── training_loss.png
    ├── model.pt
    └── predictions.npy
```

## Outputs to Save

**Solver results:**
- Temperature fields (`.npy` or `.h5`)
- Time arrays
- Training/validation curves (`.png`)

**Model checkpoints:**
- Trained model weights (`.pt`, `.pth`, or `.h5`)
- Configuration file (`.yaml` or `.json`)

**Comparisons:**
- Benchmark plots (`.png`)
- Metrics table (`.csv`)
- Model predictions vs ground truth (`.npy`)

## Example: Saving Solver Results

```python
import numpy as np
from methods.pcb_solver.test_cases import PCB_case_1

T, interfaces, heaters = PCB_case_1()
np.save('results/pcb_solver/case1_steady_T.npy', T)
np.save('results/pcb_solver/case1_interfaces.npy', interfaces)
```

## Cleanup

To free space, delete large result files:
```bash
rm -f results/**/*.npy results/**/*.h5 results/**/*.pt
```

Note: This directory is gitignored, so results are local-only and not tracked in version control.
