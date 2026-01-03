import pandas as pd
import numpy as np
from pathlib import Path


# === PHASE 2: EDA ( current code ) ===
def analyze_data(df: pd.DataFrame) -> None:
    """Print data description for report."""
    print("=== BASIC INFO ===")
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))

    print("\n=== MISSING VALUES COUNT ===")
    missing_count = df.isnull().sum()
    print(missing_count)

    print("\n=== MISSING VALUES PERCENTAGE ===")
    missing_percent = (df.isnull().sum() / len(df)) * 100
    print(missing_percent.round(2))

    print("\n=== MISSING VALUES TABLE (for report) ===")
    missing_table = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing_count,
        "Missing %": missing_percent.round(2)
    }).sort_values("Missing Count", ascending=False)
    print(missing_table)

    print("\n=== UNIQUE COUNTS (for key columns) ===")
    key_cols = ["name", "location", "cuisines", "rate", "votes"]
    for col in key_cols:
        if col in df.columns:
            print(f"{col}: {df[col].nunique()} unique values")


# === PHASE 3: CLEANING FUNCTIONS (NEW) ===
def clean_zomato_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Zomato dataset for content-based recommender."""
    df = df.copy()

    # Drop useless columns (high missing or irrelevant)
    drop_cols = ["url", "phone", "dish_liked", "reviews_list", "menu_item"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Clean rate (15% missing)
    df["rate"] = (
        df["rate"]
        .astype(str)
        .str.replace("/5", "", regex=False)
        .replace(["NEW", "-", "nan"], np.nan)
        .astype(float)
    )

    # Clean approx_cost (0.67% missing)
    df["approx_cost"] = (
        df["approx_cost(for two people)"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace("nan", np.nan)
        .astype(float)
    )
    df = df.drop(columns=["approx_cost(for two people)"], errors="ignore")

    # Fill minor missing values
    df["rest_type"] = df["rest_type"].fillna("Casual Dining")
    df = df.dropna(subset=["cuisines", "location"])  # Critical for recommender

    # Text cleaning for TF-IDF
    df["cuisines"] = df["cuisines"].str.lower().str.strip()
    df["location"] = df["location"].str.lower().str.strip()
    df["name"] = df["name"].str.strip()

    # Remove exact duplicates
    df = df.drop_duplicates(subset=["name", "location", "cuisines"])

    return df


# === MAIN PIPELINE ===
def main():
    """Complete EDA + preprocessing pipeline."""
    raw_path = Path("/Users/rohit/Desktop/swayam_project/DATASET/zomato.csv")
    clean_path = Path("/Users/rohit/Desktop/swayam_project/DATASET/zomato_clean.csv")

    # Phase 2: EDA
    print("🔍 PHASE 2: DATA ANALYSIS")
    df_raw = pd.read_csv(raw_path)
    analyze_data(df_raw)

    # Phase 3: Preprocessing
    print("\n🧹 PHASE 3: CLEANING DATA")
    df_clean = clean_zomato_data(df_raw)

    # Save cleaned data
    clean_path.parent.mkdir(exist_ok=True)
    df_clean.to_csv(clean_path, index=False)

    print(f"\n SUCCESS!")
    print(f"Raw shape:    {df_raw.shape}")
    print(f"Clean shape:  {df_clean.shape}")
    print(f"Cleaned file: {clean_path}")
    print(f"Rows reduced by: {(1 - len(df_clean) / len(df_raw)) * 100:.1f}%")

    print(df_clean.head())


if __name__ == "__main__":
    main()
