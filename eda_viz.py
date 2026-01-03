import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load  clean data directly (skip model.df)
df_path = Path("DATASET/zomato_clean.csv")  # Or "DATASET/zomato_clean.csv"
if df_path.exists():
    df = pd.read_csv(df_path)
else:
    # Fallback: Load from model if CSV missing
    from src.content_based import ContentBasedRecommender
    model = ContentBasedRecommender()
    if hasattr(model, 'df') and model.df is not None:
        df = model.df
    else:
        print(" No data found. Run PREPROCESSING/preprocessing.py first.")
        exit(1)

print(f" Loaded {len(df):,} restaurants")

# Create folder
Path("DOCUMENTATION/IMAGES").mkdir(parents=True, exist_ok=True)

# Safe plots (handle missing columns)
if 'rate' in df.columns:
    # Rating histogram
    plt.figure(figsize=(10,6))
    df['rate'].hist(bins=20, edgecolor='black')
    plt.title(f'Restaurant Rating Distribution (n={len(df):,})')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.savefig('DOCUMENTATION/IMAGES/rating_hist.png', dpi=300, bbox_inches='tight')
    plt.close()

if 'location' in df.columns:
    # Top locations
    plt.figure(figsize=(12,6))
    top_loc = df['location'].value_counts().head(10)
    top_loc.plot(kind='bar', edgecolor='black')
    plt.title('Top 10 Locations by Restaurant Count')
    plt.xlabel('Location')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('DOCUMENTATION/IMAGES/top_locations.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Heatmap: Avg Cost by Top Locations + Rating Buckets (legible!)
if all(col in df.columns for col in ['approx_cost', 'rate', 'location']):
    # Top 10 locations only
    top_locs = df['location'].value_counts().head(10).index
    df_filtered = df[df['location'].isin(top_locs)].copy()

    # Rating + cost buckets
    df_filtered['rate_bucket'] = pd.cut(df_filtered['rate'], bins=5, labels=['0-1', '1-2', '2-3', '3-4', '4-5'])
    df_filtered['cost_bucket'] = pd.cut(df_filtered['approx_cost'], bins=5,
                                        labels=['₹0-200', '₹200-400', '₹400-600', '₹600-800', '₹800+'])

    # Pivot heatmap
    heatmap_data = df_filtered.groupby(['location', 'rate_bucket'])['approx_cost'].mean().unstack(fill_value=0)

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data.T, annot=True, cmap='YlOrRd', fmt='.0f', cbar_kws={'label': 'Avg Cost (₹)'})
    plt.title('Avg Cost Heatmap: Top 10 Locations by Rating Bucket')
    plt.xlabel('Location')
    plt.ylabel('Rating Bucket')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('DOCUMENTATION/IMAGES/location_rating_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Heatmap saved (top 10 locations + rating → avg cost)")

print("EDA plots saved: DOCUMENTATION/IMAGES/*.png")
print("Add to PROGRESS_REPORT.docx!")
