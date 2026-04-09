import pytest
from common.base_model import BaseModel
from methods.method_a.model import ModelA
from methods.method_b.model import ModelB
from methods.method_c.model import ModelC

@pytest.mark.parametrize("ModelClass", [ModelA, ModelB, ModelC])
def test_inherits_base(ModelClass):
    assert issubclass(ModelClass, BaseModel)

@pytest.mark.parametrize("ModelClass", [ModelA, ModelB, ModelC])
def test_has_required_methods(ModelClass):
    for method in ["fit", "predict", "save", "load"]:
        assert callable(getattr(ModelClass, method, None))