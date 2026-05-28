import re
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Define flexible mappings for various scraper outputs and open data sources
COLUMN_MAPPINGS = {
    'land_area': ['land_area', 'land_size', 'landsize', 'landarea', 'lot_size', 'lotsize', 'land_area_sqm', 'land_area_m2', 'land_size_sqm', 'section_size', 'land_sqm', 'land_m2'],
    'building_area': ['building_area', 'building_size', 'floor_area', 'floor_size', 'floorarea', 'buildingarea', 'house_size', 'building_area_sqm', 'floor_area_sqm', 'floor_area_m2', 'floor_sqm', 'floor_m2'],
    'bedrooms': ['bedrooms', 'bedroom', 'beds', 'bed', 'bedroom_count', 'no_bedrooms', 'room_count'],
    'bathrooms': ['bathrooms', 'bathroom', 'baths', 'bath', 'bathroom_count', 'no_bathrooms'],
    'age': ['age', 'house_age', 'property_age'],
    'year_built': ['year_built', 'yearbuilt', 'year_constructed', 'built_year', 'year', 'construction_year', 'built'],
    'location': ['location', 'suburb', 'district', 'city', 'region', 'address', 'address_suburb', 'town', 'area'],
    'property_type': ['property_type', 'propertytype', 'type', 'category', 'dwelling_type', 'prop_type', 'type_of_property'],
    'price': ['price', 'sale_price', 'price_nzd', 'sold_price', 'amount', 'display_price', 'cost', 'valuation', 'capital_value', 'cv', 'rv', 'rating_valuation']
}

def clean_numeric(val):
    """Converts dirty strings like '$1,250,000' or 'Asking $850k' into floats."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    
    val_str = str(val).lower().strip()
    
    # Remove standard punctuation and symbols
    val_str = val_str.replace('$', '').replace(',', '').replace('nzd', '')
    
    # Handle shorthand scales like K, M, B
    multiplier = 1.0
    if 'k' in val_str:
        multiplier = 1_000.0
        val_str = val_str.replace('k', '')
    elif 'm' in val_str:
        multiplier = 1_000_000.0
        val_str = val_str.replace('m', '')
        
    # Extract the first consecutive digit-like pattern
    match = re.search(r'[-+]?\d*\.\d+|\d+', val_str)
    if match:
        try:
            return float(match.group()) * multiplier
        except ValueError:
            return None
    return None

def process_raw_dataset(input_csv_path: Path, output_csv_path: Path):
    print(f"[Info] Reading raw dataset from: {input_csv_path}")
    
    try:
        # Detect delimiter or encoding issues
        df = pd.read_csv(input_csv_path, encoding='utf-8', on_bad_lines='skip')
    except UnicodeDecodeError:
        df = pd.read_csv(input_csv_path, encoding='latin1', on_bad_lines='skip')

    print(f"[Info] Original shape: {df.shape}")
    print(f"[Info] Available columns: {list(df.columns)}")
    
    # Standardize column header strings for mapping
    df.columns = [col.lower().strip() for col in df.columns]
    
    mapped_df = pd.DataFrame()
    detected_mappings = {}

    # Map raw columns to canonical training schema
    for canonical_col, synonyms in COLUMN_MAPPINGS.items():
        matched = False
        for synonym in synonyms:
            syn_lower = synonym.lower().strip()
            if syn_lower in df.columns:
                mapped_df[canonical_col] = df[syn_lower]
                detected_mappings[canonical_col] = synonym
                matched = True
                break
        
        if not matched:
            print(f"[Warning] Could not find match for required column: '{canonical_col}'")
            mapped_df[canonical_col] = None

    print(f"[Info] Columns mapped successfully: {detected_mappings}")

    # 1. Clean price (Critical)
    mapped_df['price'] = mapped_df['price'].apply(clean_numeric)
    original_count = len(mapped_df)
    mapped_df = mapped_df.dropna(subset=['price'])
    print(f"[Info] Cleaned price. Kept {len(mapped_df)} of {original_count} rows with valid prices.")

    # 2. Clean land and building area
    mapped_df['land_area'] = mapped_df['land_area'].apply(clean_numeric)
    mapped_df['building_area'] = mapped_df['building_area'].apply(clean_numeric)

    # Impute missing areas based on property characteristics or defaults
    median_land = mapped_df['land_area'].median() or 500.0
    median_building = mapped_df['building_area'].median() or 120.0
    mapped_df['land_area'] = mapped_df['land_area'].fillna(median_land)
    mapped_df['building_area'] = mapped_df['building_area'].fillna(median_building)

    # 3. Clean bedrooms and bathrooms
    mapped_df['bedrooms'] = mapped_df['bedrooms'].apply(clean_numeric).fillna(3).astype(int)
    mapped_df['bathrooms'] = mapped_df['bathrooms'].apply(clean_numeric).fillna(2).astype(int)

    # 4. Handle Age / Year Built
    mapped_df['age'] = mapped_df['age'].apply(clean_numeric)
    if 'year_built' in mapped_df.columns:
        mapped_df['year_built'] = mapped_df['year_built'].apply(clean_numeric)
        current_year = datetime.now().year
        # Compute age from year built if age is missing
        mapped_df['age'] = mapped_df.apply(
            lambda row: current_year - row['year_built'] if pd.isna(row['age']) and not pd.isna(row['year_built']) else row['age'],
            axis=1
        )
    
    # Impute default age if still missing
    median_age = mapped_df['age'].median() or 25.0
    mapped_df['age'] = mapped_df['age'].fillna(median_age).astype(int)

    # Remove temporary helper columns if present
    if 'year_built' in mapped_df.columns:
        mapped_df = mapped_df.drop(columns=['year_built'])

    # 5. Clean location
    mapped_df['location'] = mapped_df['location'].fillna('unknown').astype(str).str.lower().str.strip()

    # 6. Clean property type
    mapped_df['property_type'] = mapped_df['property_type'].fillna('house').astype(str).str.lower().str.strip()

    # Final validation check
    mapped_df = mapped_df[['land_area', 'building_area', 'bedrooms', 'bathrooms', 'age', 'location', 'property_type', 'price']]
    
    # Save the polished training dataset
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    mapped_df.to_csv(output_csv_path, index=False)
    
    print(f"[Success] Data preprocessing finished! Cleaned training dataset exported to: {output_csv_path}")
    print(f"[Success] Final shape of processed dataset: {mapped_df.shape}")

def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    data_dir = repo_root / "Machine Learning" / "data"
    
    target_csv = data_dir / "nz_homes.csv"
    
    # Scan for any CSV in the data directory that isn't the final nz_homes.csv or the legacy data.csv
    raw_files = [f for f in data_dir.glob("*.csv") if f.name not in ("nz_homes.csv", "data.csv")]
    
    if not raw_files:
        # Check if they directly placed it as nz_homes.csv
        if target_csv.exists():
            # Run the processor on the placed file itself to clean it in-place
            print(f"[Info] Found 'nz_homes.csv' in the data folder. Performing automatic cleaning...")
            temp_raw = data_dir / "nz_homes_raw_temp.csv"
            target_csv.rename(temp_raw)
            try:
                process_raw_dataset(temp_raw, target_csv)
                temp_raw.unlink()
            except Exception as e:
                # Restore original if cleaning failed
                temp_raw.rename(target_csv)
                print(f"[Error] Failed to automatically process the CSV: {e}")
                sys.exit(1)
        else:
            print("[Info] No raw CSV files found in 'Machine Learning/data/' to process.")
            print("Please drop your downloaded property CSV in the data folder and name it 'nz_homes.csv' or similar.")
            sys.exit(0)
    else:
        # Pick the most recently added raw CSV file
        raw_csv = max(raw_files, key=lambda f: f.stat().st_mtime)
        print(f"[Info] Detected raw dataset: {raw_csv.name}")
        process_raw_dataset(raw_csv, target_csv)

if __name__ == "__main__":
    main()
