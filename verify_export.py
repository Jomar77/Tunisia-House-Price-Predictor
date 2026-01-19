"""verify_export.py

Parity testing for exported Safetensors linear regression.

This script proves the backend NumPy implementation (np.dot) matches a
scikit-learn LinearRegression reference built from the exported weights.

Constraints:
- No pickle/joblib deserialization
- Uses columns.json as the source of truth for feature ordering
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
    return Path(__file__).resolve().parent


def load_artifacts(repo_root: Path) -> Artifacts:
    columns_path = repo_root / "columns.json"
    model_path = repo_root / "tunisia_home_prices_model.safetensors"

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
    ys_numpy: list[float] = []
    ys_sklearn: list[float] = []

    for case in cases:
        x = vectorizer.vectorize(**case)
        xs.append(x)

        y_np = numpy_predict(x, artifacts.coef, artifacts.intercept)
        y_sk = float(lr.predict(x.reshape(1, -1))[0])
        ys_numpy.append(y_np)
        ys_sklearn.append(y_sk)

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
"""verify_export.py

Standalone verifier for exported inference artifacts.

What it checks:
- Loads `columns.json` (feature contract)
- Loads `tunisia_home_prices_model.safetensors` (coef + intercept)
- Constructs an sklearn `LinearRegression` with weights populated from safetensors
- Uses `FeatureVectorizer` to build feature vectors (including unknown location -> 'other')
- Compares sklearn prediction vs a manual numpy dot-product within tolerances

No pickle/joblib is used.
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

from backend.domain.vectorizer import FeatureVectorizer


@dataclass(frozen=True)
class TestCase:
    area: float
    rooms: float
    bathrooms: float
    age: float
    location: str


def _load_columns(columns_path: Path) -> tuple[list[str], list[str]]:
    with columns_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "data_columns" not in data or "numeric_features" not in data:
        raise ValueError(
            "columns.json must contain 'data_columns' and 'numeric_features' keys"
        )

    feature_columns = list(data["data_columns"])
    numeric_features = list(data["numeric_features"])

    if len(feature_columns) == 0:
        raise ValueError("columns.json has empty 'data_columns'")

    return feature_columns, numeric_features


def _load_weights(model_path: Path) -> tuple[np.ndarray, float]:
    tensors = load_file(str(model_path))

    if "coef" not in tensors:
        raise ValueError("Missing 'coef' tensor in model file")
    if "intercept" not in tensors:
        raise ValueError("Missing 'intercept' tensor in model file")

    coef = np.asarray(tensors["coef"], dtype=np.float64)
    intercept_arr = np.asarray(tensors["intercept"], dtype=np.float64)

    if coef.ndim != 1:
        raise ValueError(f"Expected coef to be 1D, got shape={coef.shape}")
    if intercept_arr.shape != (1,):
        raise ValueError(
            f"Expected intercept to have shape (1,), got shape={intercept_arr.shape}"
        )

    return coef, float(intercept_arr[0])


def _build_sklearn_lr(coef: np.ndarray, intercept: float) -> LinearRegression:
    lr = LinearRegression()
    # Populate fitted attributes (no training performed here)
    lr.coef_ = coef.astype(np.float64, copy=False)
    lr.intercept_ = float(intercept)
    lr.n_features_in_ = int(coef.shape[0])
    return lr


def _default_test_cases(
    supported_locations: list[str],
    include_unknown_location: bool,
) -> list[TestCase]:
    """Deterministic suite of test cases.

    Includes:
    - a few known locations
    - explicit 'other'
    - an unknown location that should map to 'other' when available
    """

    # Keep deterministic selection order.
    locations_to_use: list[str] = []

    # Prefer a few "real" locations (exclude the literal 'other' bucket for now).
    for loc in supported_locations:
        if loc == "other":
            continue
        locations_to_use.append(loc)
        if len(locations_to_use) >= 3:
            break

    # Always include explicit 'other' if present.
    if "other" in supported_locations:
        locations_to_use.append("other")

    # Add an unknown location case (vectorizer should map to 'other').
    if include_unknown_location:
        locations_to_use.append("__unknown_location__")

    # A small but varied grid of numeric values.
    areas = [50.0, 110.0, 250.0]
    rooms = [1.0, 3.0, 5.0]
    bathrooms = [1.0, 2.0]
    ages = [0.0, 8.0, 25.0]

    cases: list[TestCase] = []

    # A deterministic subset of the cartesian product to keep runtime fast.
    for i, loc in enumerate(locations_to_use):
        cases.append(TestCase(area=areas[i % len(areas)], rooms=rooms[0], bathrooms=bathrooms[0], age=ages[0], location=loc))
        cases.append(TestCase(area=areas[(i + 1) % len(areas)], rooms=rooms[1], bathrooms=bathrooms[1], age=ages[1], location=loc))
        cases.append(TestCase(area=areas[(i + 2) % len(areas)], rooms=rooms[2], bathrooms=bathrooms[0], age=ages[2], location=loc))

    # Add a couple of edge-ish cases.
    cases.append(TestCase(area=20.0, rooms=1.0, bathrooms=1.0, age=60.0, location=locations_to_use[0]))
    cases.append(TestCase(area=600.0, rooms=8.0, bathrooms=4.0, age=1.0, location=locations_to_use[0]))

    return cases


def _predict_manual(x: np.ndarray, coef: np.ndarray, intercept: float) -> float:
    return float(np.dot(x.astype(np.float64, copy=False), coef) + intercept)


def _relative_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def verify(
    *,
    columns_path: Path,
    model_path: Path,
    rtol: float,
    atol: float,
    include_unknown_location: bool,
) -> int:
    feature_columns, numeric_features = _load_columns(columns_path)
    coef, intercept = _load_weights(model_path)

    if coef.shape[0] != len(feature_columns):
        raise ValueError(
            f"coef length ({coef.shape[0]}) != number of columns ({len(feature_columns)})"
        )

    vectorizer = FeatureVectorizer(
        feature_columns=feature_columns,
        numeric_features=numeric_features,
    )

    lr = _build_sklearn_lr(coef, intercept)

    supported_locations = vectorizer.get_supported_locations()
    cases = _default_test_cases(supported_locations, include_unknown_location)

    worst_abs = -1.0
    worst_rel = -1.0
    worst_case: TestCase | None = None
    worst_vals: tuple[float, float] | None = None

    failures: list[tuple[TestCase, float, float, float]] = []

    for case in cases:
        x = vectorizer.vectorize(
            area=case.area,
            rooms=case.rooms,
            bathrooms=case.bathrooms,
            age=case.age,
            location=case.location,
        )

        manual = _predict_manual(x, coef, intercept)
        sklearn_pred = float(lr.predict(x.reshape(1, -1))[0])

        abs_diff = abs(sklearn_pred - manual)
        rel_diff = _relative_diff(sklearn_pred, manual)

        if abs_diff > worst_abs:
            worst_abs = abs_diff
            worst_rel = rel_diff
            worst_case = case
            worst_vals = (sklearn_pred, manual)

        if not np.isclose(sklearn_pred, manual, rtol=rtol, atol=atol):
            failures.append((case, sklearn_pred, manual, abs_diff))

    # Reporting
    print("verify_export.py: artifact verification")
    print(f"- columns: {columns_path}")
    print(f"- model:   {model_path}")
    print(f"- n_features: {len(feature_columns)}")
    print(f"- n_cases: {len(cases)}")
    print(f"- tolerances: rtol={rtol:g}, atol={atol:g}")

    if worst_case is not None and worst_vals is not None:
        sp, mp = worst_vals
        print(
            "- worst_case_abs_diff: "
            f"{worst_abs:.10g} (sklearn={sp:.5f}, manual={mp:.5f}, rel={worst_rel:.10g})"
        )
        print(
            "  worst_case_inputs: "
            f"area={worst_case.area}, rooms={worst_case.rooms}, "
            f"bathrooms={worst_case.bathrooms}, age={worst_case.age}, "
            f"location={worst_case.location!r}"
        )

    if failures:
        print(f"FAIL: {len(failures)} case(s) exceeded tolerance")
        # Print up to a small number of failures, worst-first by abs diff.
        failures_sorted = sorted(failures, key=lambda t: t[3], reverse=True)
        for case, sklearn_pred, manual, abs_diff in failures_sorted[:10]:
            print(
                "  case: "
                f"area={case.area}, rooms={case.rooms}, bathrooms={case.bathrooms}, "
                f"age={case.age}, location={case.location!r} "
                f"=> sklearn={sklearn_pred:.5f}, manual={manual:.5f}, abs_diff={abs_diff:.10g}"
            )
        return 1

    print("OK: sklearn prediction matches manual dot-product")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Verify exported columns.json + safetensors weights by comparing sklearn "
            "LinearRegression predictions against a manual numpy dot-product."
        )
    )
    p.add_argument(
        "--columns",
        type=Path,
        default=Path("columns.json"),
        help="Path to columns.json (default: columns.json)",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=Path("tunisia_home_prices_model.safetensors"),
        help="Path to model .safetensors (default: tunisia_home_prices_model.safetensors)",
    )
    p.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for np.isclose (default: 1e-5)",
    )
    p.add_argument(
        "--atol",
        type=float,
        default=1e-8,
        help="Absolute tolerance for np.isclose (default: 1e-8)",
    )
    p.add_argument(
        "--no-unknown-location",
        action="store_true",
        help="Disable the test case that uses an unknown location mapped to 'other'",
    )

    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        return verify(
            columns_path=args.columns,
            model_path=args.model,
            rtol=args.rtol,
            atol=args.atol,
            include_unknown_location=not args.no_unknown_location,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
