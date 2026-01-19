"""scripts/verify_export.py

Parity testing for exported Safetensors linear regression.

This script proves the backend NumPy implementation (np.dot) matches a
scikit-learn LinearRegression reference built from the exported weights.

Constraints:
- No pickle/joblib deserialization
- Uses columns.json as the source of truth for feature ordering

Usage:
- `python scripts/verify_export.py`
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class Artifacts:
    feature_columns: list[str]
    numeric_features: list[str]
    coef: np.ndarray
    intercept: float


def _repo_root() -> Path:
    # scripts/verify_export.py -> repo_root
    return Path(__file__).resolve().parent.parent


def _resolve_artifact(repo_root: Path, filename: str) -> Path:
    preferred = repo_root / "artifacts" / filename
    if preferred.exists():
        return preferred
    return repo_root / filename


def load_artifacts(repo_root: Path) -> Artifacts:
    columns_path = _resolve_artifact(repo_root, "columns.json")
    model_path = _resolve_artifact(repo_root, "tunisia_home_prices_model.safetensors")

    with open(columns_path, "r", encoding="utf-8") as f:
        columns_data = json.load(f)

    feature_columns = list(columns_data["data_columns"])
    numeric_features = list(columns_data["numeric_features"])

    tensors = load_file(str(model_path))
    if "coef" not in tensors or "intercept" not in tensors:
        raise ValueError("Model file must contain 'coef' and 'intercept' tensors")

    coef = np.asarray(tensors["coef"]).reshape(-1)
    intercept_tensor = np.asarray(tensors["intercept"]).reshape(-1)
    if intercept_tensor.size != 1:
        raise ValueError(f"Expected intercept to have 1 value, got {intercept_tensor.size}")
    intercept = float(intercept_tensor[0])

    if coef.shape[0] != len(feature_columns):
        raise ValueError(
            f"coef length ({coef.shape[0]}) does not match columns.json length ({len(feature_columns)})"
        )

    return Artifacts(
        feature_columns=feature_columns,
        numeric_features=numeric_features,
        coef=coef,
        intercept=intercept,
    )


def build_sklearn_reference(artifacts: Artifacts) -> LinearRegression:
    """Build a sklearn LinearRegression object without training/pickles."""
    lr = LinearRegression()
    lr.coef_ = np.asarray(artifacts.coef, dtype=float)
    lr.intercept_ = float(artifacts.intercept)
    lr.n_features_in_ = int(len(artifacts.feature_columns))
    return lr


def build_vectorizer(artifacts: Artifacts):
    # Import from backend to ensure we parity-test the real production vectorizer.
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from backend.domain.vectorizer import FeatureVectorizer

    return FeatureVectorizer(
        feature_columns=artifacts.feature_columns,
        numeric_features=artifacts.numeric_features,
    )


def numpy_predict(x_vec: np.ndarray, coef: np.ndarray, intercept: float) -> float:
    return float(np.dot(x_vec, coef) + intercept)


def run_parity(
    *,
    artifacts: Artifacts,
    lr: LinearRegression,
    samples: int,
    seed: int,
    rtol: float,
    atol: float,
) -> int:
    vectorizer = build_vectorizer(artifacts)

    supported_locations = vectorizer.get_supported_locations()
    has_other = any(loc.lower() == "other" for loc in supported_locations)
    locations_no_other = [loc for loc in supported_locations if loc.lower() != "other"]
    if not locations_no_other:
        raise ValueError("No supported locations found (excluding 'other')")

    rng = np.random.default_rng(seed)

    # Deterministic golden cases (including an unknown location)
    golden_cases: list[dict] = [
        {"area": 150.0, "rooms": 3.0, "bathrooms": 2.0, "age": 5.0, "location": locations_no_other[0]},
        {"area": 80.0, "rooms": 2.0, "bathrooms": 1.0, "age": 20.0, "location": locations_no_other[-1]},
        {"area": 250.0, "rooms": 5.0, "bathrooms": 3.0, "age": 0.0, "location": "__UNKNOWN_LOCATION__"},
    ]

    def _random_case() -> dict:
        location = locations_no_other[int(rng.integers(0, len(locations_no_other)))]
        # Include a small fraction of unknown locations to exercise `other` fallback.
        if float(rng.random()) < 0.05:
            location = "__UNKNOWN_LOCATION__"
        return {
            "area": float(rng.uniform(20.0, 1200.0)),
            "rooms": float(rng.integers(1, 11)),
            "bathrooms": float(rng.integers(1, 8)),
            "age": float(rng.uniform(0.0, 120.0)),
            "location": location,
        }

    cases = golden_cases + [_random_case() for _ in range(samples)]

    worst = {
        "abs": -1.0,
        "rel": -1.0,
        "case": None,
        "y_numpy": None,
        "y_sklearn": None,
    }

    xs: list[np.ndarray] = []

    for case in cases:
        x = vectorizer.vectorize(**case)
        xs.append(x)

        y_np = numpy_predict(x, artifacts.coef, artifacts.intercept)
        y_sk = float(lr.predict(x.reshape(1, -1))[0])

        abs_err = abs(y_np - y_sk)
        denom = max(abs(y_sk), 1e-12)
        rel_err = abs_err / denom

        if abs_err > worst["abs"]:
            worst.update({"abs": abs_err, "rel": rel_err, "case": case, "y_numpy": y_np, "y_sklearn": y_sk})

        if not np.isclose(y_np, y_sk, rtol=rtol, atol=atol):
            print("FAIL: Parity mismatch")
            print(f"  case={case}")
            print(f"  y_numpy  ={y_np:.5f}")
            print(f"  y_sklearn={y_sk:.5f}")
            print(f"  abs_err={abs_err:.8f} rel_err={rel_err:.8f} (rtol={rtol} atol={atol})")
            if case["location"] == "__UNKNOWN_LOCATION__" and not has_other:
                print("  NOTE: unknown location used but 'other' column is missing")
            return 1

    # Batch parity check
    X = np.stack(xs, axis=0)
    y_np_batch = X @ artifacts.coef + artifacts.intercept
    y_sk_batch = lr.predict(X)
    np.testing.assert_allclose(y_np_batch, y_sk_batch, rtol=rtol, atol=atol)

    print("PASS: Export parity verified")
    print(f"  cases={len(cases)}")
    print(
        "  worst: "
        f"abs_err={worst['abs']:.8f} rel_err={worst['rel']:.8f} "
        f"y_numpy={float(worst['y_numpy']):.5f} y_sklearn={float(worst['y_sklearn']):.5f} "
        f"case={worst['case']}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Safetensors export parity vs sklearn predict().")
    parser.add_argument("--samples", type=int, default=1000, help="Number of randomized cases (in addition to golden cases)")
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance")
    args = parser.parse_args()

    repo_root = _repo_root()
    artifacts = load_artifacts(repo_root)
    lr = build_sklearn_reference(artifacts)

    return run_parity(
        artifacts=artifacts,
        lr=lr,
        samples=args.samples,
        seed=args.seed,
        rtol=args.rtol,
        atol=args.atol,
    )


if __name__ == "__main__":
    raise SystemExit(main())
