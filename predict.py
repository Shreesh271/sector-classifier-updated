import os
import time

import joblib

from utils import preprocess_definition

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

_tfidf = None
_sector_model = None
_subsector_model = None


def _load_artifacts() -> None:
    """Lazily load the TF-IDF vectorizer and both trained models into
    module-level caches.

    Raises:
        FileNotFoundError: If any required model artifact is missing.
    """
    global _tfidf, _sector_model, _subsector_model

    if _tfidf is not None and _sector_model is not None and _subsector_model is not None:
        return

    tfidf_path = os.path.join(MODELS_DIR, "tfidf.pkl")
    sector_path = os.path.join(MODELS_DIR, "sector_model.pkl")
    subsector_path = os.path.join(MODELS_DIR, "subsector_model.pkl")

    for path in (tfidf_path, sector_path, subsector_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model artifact not found: {path}. Run train.py first."
            )

    _tfidf = joblib.load(tfidf_path)
    _sector_model = joblib.load(sector_path)
    _subsector_model = joblib.load(subsector_path)


def predict(definition: str) -> dict:
    """Predict the Sector and Subsector for a given organization definition.

    Args:
        definition: Raw text describing the organization.

    Returns:
        Dictionary with keys "sector" and "subsector".

    Raises:
        ValueError: If the definition is empty or not a string.
    """
    if not isinstance(definition, str) or not definition.strip():
        raise ValueError("Definition must be a non-empty string.")

    _load_artifacts()

    start_time = time.perf_counter()
    clean_text = preprocess_definition(definition)
    features = _tfidf.transform([clean_text])

    sector = _sector_model.predict(features)[0]
    subsector = _subsector_model.predict(features)[0]
    prediction_time = round(
    time.perf_counter() - start_time,
    4
    )
    return {
    "sector": sector,
    "subsector": subsector,
    "clean_definition": clean_text,
    "prediction_time": prediction_time,
    "model_status": "success"
    }

if __name__ == "__main__":
    sample_definition = (
        "A government body responsible for regulating electricity "
        "generation, transmission, and distribution across the country."
    )
    result = predict(sample_definition)
    print(f"Definition: {sample_definition}")
    print(f"Predicted Sector   : {result['sector']}")
    print(f"Predicted Subsector: {result['subsector']}")
