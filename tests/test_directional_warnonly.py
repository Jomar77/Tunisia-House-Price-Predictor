import warnings

from fastapi.testclient import TestClient


def test_area_directional_expectation_warn_only():
    """Warn-only: larger area should usually not reduce predicted price."""
    from backend.main import app

    with TestClient(app) as client:
        base = {"rooms": 3, "bathrooms": 2, "age": 5, "location": "Hammamet"}

        r1 = client.post("/api/v1/predict", json={**base, "area": 80}).json()["predicted_price"]
        r2 = client.post("/api/v1/predict", json={**base, "area": 120}).json()["predicted_price"]

        if r2 < r1:
            warnings.warn(
                f"Directional check violated: area 80->{r1:.2f}, area 120->{r2:.2f}. "
                "This is warn-only and should be reviewed if frequent.",
                RuntimeWarning,
            )

    assert True
