from fastapi.testclient import TestClient


def test_negative_inputs_rejected_422():
    from backend.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/predict",
            json={
                "area": -1,
                "rooms": 3,
                "bathrooms": 1,
                "age": 5,
                "location": "Hammamet",
            },
        )
        assert resp.status_code == 422


def test_unknown_location_is_accepted_via_other_fallback():
    from backend.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/predict",
            json={
                "area": 150,
                "rooms": 3,
                "bathrooms": 2,
                "age": 5,
                "location": "__UNKNOWN_LOCATION__",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_price" in data
        assert isinstance(data["predicted_price"], (int, float))

