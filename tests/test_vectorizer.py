import json
from pathlib import Path

import numpy as np


def test_unknown_location_maps_to_other():
    repo_root = Path(__file__).resolve().parent.parent
    preferred = repo_root / "artifacts" / "columns.json"
    columns_path = preferred if preferred.exists() else (repo_root / "columns.json")
    columns = json.loads(columns_path.read_text(encoding="utf-8"))

    from backend.domain.vectorizer import FeatureVectorizer

    vectorizer = FeatureVectorizer(
        feature_columns=columns["data_columns"],
        numeric_features=columns["numeric_features"],
    )

    x = vectorizer.vectorize(
        area=120.0,
        rooms=3.0,
        bathrooms=2.0,
        age=10.0,
        location="__UNKNOWN_LOCATION__",
    )

    other_idx = columns["data_columns"].index("other")
    assert x.shape == (len(columns["data_columns"]),)
    assert np.isclose(x[other_idx], 1.0)

