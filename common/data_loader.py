import numpy as np

SEED = 42

def load_data():
    """
    Returns a dict with keys: X_train, y_train, X_val, y_val, X_test, y_test.
    TODO: replace this stub with the real dataset loading once the team agrees.
    """
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((100, 5))
    y = rng.standard_normal(100)
    return {
        "X_train": X[:60], "y_train": y[:60],
        "X_val":   X[60:80], "y_val":   y[60:80],
        "X_test":  X[80:],  "y_test":  y[80:],
    }
