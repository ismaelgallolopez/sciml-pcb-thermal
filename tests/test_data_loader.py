from common.data_loader import load_data

def test_load_data_keys():
    data = load_data()
    expected = {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test"}
    assert expected.issubset(data.keys())

def test_load_data_reproducible():
    d1 = load_data()
    d2 = load_data()
    assert (d1["X_train"] == d2["X_train"]).all()