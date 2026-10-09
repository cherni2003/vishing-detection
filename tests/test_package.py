from importlib.util import find_spec

def test_vishing_detection_package_exists():
    assert find_spec("vishing_detection") is not None