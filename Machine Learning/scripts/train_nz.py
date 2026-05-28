import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from safetensors.numpy import save_file


def train_and_export():
    # Resolve paths relative to repo root
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    # Dynamically run data preparation if available
    import sys
    if str(script_dir) not in sys.path:
        sys.path.append(str(script_dir))
    try:
        from prepare_nz_data import main as prepare_data
        prepare_data()
    except Exception as e:
        print(f"[Info] Automatic preprocessing check skipped/failed: {e}")

    data_path = repo_root / "Machine Learning" / "data" / "nz_homes.csv"
    artifacts_dir = repo_root / "artifacts"

    if not data_path.exists():
        print(f"[Error] NZ Housing dataset CSV not found at: {data_path}")
        print("Please place your real 'nz_homes.csv' file in that location.")
        print("Expected columns: 'land_area', 'building_area', 'bedrooms', 'bathrooms', 'age', 'location', 'property_type', and 'price'")
        return

    print(f"[Info] Reading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # Ensure standard lowercase column names
    df.columns = [col.lower().strip() for col in df.columns]

    # Check for required columns
    required_cols = {'land_area', 'building_area', 'bedrooms', 'bathrooms', 'age', 'location', 'property_type', 'price'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"[Error] Missing columns in CSV: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return

    # 2. Data Cleaning & Outlier Removal
    original_count = len(df)
    df = df[df['building_area'] > 10]
    df = df[df['land_area'] > 10]
    df = df[df['price'] > 10_000]
    cleaned_count = len(df)
    print(f"[Info] Cleaned outliers: kept {cleaned_count} of {original_count} rows ({original_count - cleaned_count} removed).")

    # 3. Categorical Bucketing for Locations
    location_stats = df['location'].value_counts()
    rare_locations = location_stats[location_stats < 5].index
    df['location'] = df['location'].apply(lambda x: 'other' if x in rare_locations else x)
    print(f"[Info] Location distribution: {df['location'].nunique()} unique locations (including 'other' fallback).")

    # 4. Feature Encoding & Scaling
    numeric_features = ["land_area", "building_area", "bedrooms", "bathrooms", "age"]

    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(df[numeric_features])
    df_scaled_numeric = pd.DataFrame(scaled_numeric, columns=numeric_features, index=df.index)

    df_location_encoded = pd.get_dummies(df['location'], dtype=float)
    df_prop_encoded = pd.get_dummies(df['property_type'].str.lower(), prefix='property_type', dtype=float)

    X = pd.concat([df_scaled_numeric, df_location_encoded, df_prop_encoded], axis=1)
    y = df['price'].values

    X.columns = [str(c) for c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )

    print("[Info] Training Multi-Layer Perceptron Regressor...")
    mlp.fit(X_train, y_train)

    train_score = mlp.score(X_train, y_train)
    test_score = mlp.score(X_test, y_test)
    print(f"[Info] Training completed! R2 score - Train: {train_score:.4f}, Test: {test_score:.4f}")

    tensors = {}
    for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        tensors[f"w_{i}"] = np.asarray(w, dtype=np.float32)
        tensors[f"b_{i}"] = np.asarray(b, dtype=np.float32)

    artifacts_dir.mkdir(exist_ok=True)

    model_export_path = artifacts_dir / "nz_home_prices_model.safetensors"
    save_file(tensors, str(model_export_path))
    print(f"[Info] Exported model weights to {model_export_path}")

    scaling_stats = {}
    for idx, feat in enumerate(numeric_features):
        scaling_stats[feat] = {
            "mean": float(scaler.mean_[idx]),
            "std": float(scaler.scale_[idx])
        }

    columns_payload = {
        "schema_version": 2,
        "numeric_features": numeric_features,
        "data_columns": list(X.columns),
        "scaling_stats": scaling_stats
    }

    columns_path = artifacts_dir / "columns.json"
    with open(columns_path, "w", encoding="utf-8") as f:
        json.dump(columns_payload, f, indent=2)
    print(f"[Info] Exported feature columns and scaling parameters to {columns_path}")

    district_medians = df.groupby('location')['price'].median().to_dict()

    metadata_payload = {
        "schema_version": 2,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "numeric_features": numeric_features,
        "n_training_rows": int(df.shape[0]),
        "n_features": int(len(X.columns)),
        "train_r2_score": float(train_score),
        "test_r2_score": float(test_score),
        "district_medians": district_medians
    }

    metadata_path = artifacts_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)
    print(f"[Info] Exported model training metadata to {metadata_path}")
    print("[Info] All NZ model training artifacts exported successfully! Ready for backend service execution.")


if __name__ == "__main__":
    train_and_export()