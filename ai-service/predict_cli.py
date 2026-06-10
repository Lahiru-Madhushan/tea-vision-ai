"""CLI wrapper for disease prediction — prints JSON only on stdout."""
import json
import os
import sys
import warnings
from pathlib import Path

# Suppress TensorFlow / Keras noise before any TF import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))


def _emit(payload: dict, code: int = 0) -> None:
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()
    sys.exit(code)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _emit({"error": "Image path required"}, 1)
    try:
        from services.predictor import predict_disease_detailed

        result = predict_disease_detailed(sys.argv[1])
        _emit(result)
    except Exception as exc:
        _emit({"error": str(exc)}, 1)
