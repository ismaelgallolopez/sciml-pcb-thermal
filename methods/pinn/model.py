from common.base_model import BaseModel

class PINN(BaseModel):
    """Physics-Informed Neural Network for the thermal PDE."""

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError