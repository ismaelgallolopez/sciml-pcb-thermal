import pytest
from common.base_model import BaseModel
from methods.pinn.model import PINN
from methods.deep_ritz.model import DeepRitz
from methods.neural_ode.model import NeuralODE
from methods.deeponet.model import DeepONet
from methods.fno.model import FNO

ALL_MODELS = [PINN, DeepRitz, NeuralODE, DeepONet, FNO]

@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_inherits_base(ModelClass):
    assert issubclass(ModelClass, BaseModel)

@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_has_required_methods(ModelClass):
    for method in ["fit", "predict", "save", "load"]:
        assert callable(getattr(ModelClass, method, None))