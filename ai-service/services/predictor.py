"""Load prediction2.py from the folder with a space in its name."""
import importlib.util
from pathlib import Path

_PREDICTION2_PATH = (
    Path(__file__).resolve().parent.parent
    / "DiseasePrediction"
    / "prediction Model2"
    / "prediction2.py"
)


def _load_prediction2_module():
    spec = importlib.util.spec_from_file_location("prediction2", _PREDICTION2_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load predictor from {_PREDICTION2_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_prediction2 = None


def _module():
    global _prediction2
    if _prediction2 is None:
        _prediction2 = _load_prediction2_module()
    return _prediction2


def get_model_path():
    return _module().get_model_path()


def model_exists() -> bool:
    return _module().model_exists()


def predict_disease_detailed(image_path: str) -> dict:
    return _module().predict_disease_detailed(image_path)


def predict_disease(image_path: str):
    return _module().predict_disease(image_path)
