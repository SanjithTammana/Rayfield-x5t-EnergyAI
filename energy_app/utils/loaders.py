from functools import lru_cache
import joblib
from ..settings import MODEL_DIR

@lru_cache(maxsize=None)
def load_bundle(name: str):
    """Return the saved model bundle dict for `name`. Raises FileNotFoundError if absent."""
    return joblib.load(MODEL_DIR / f"{name}.pkl")
