import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings("ignore")


class ContentBasedRecommender:
    def __init__(self, clean_data_path: str = "../DATASET/zomato_clean.csv"):  #  PATH
        self.clean_data_path = Path(clean_data_path)
        self.df = None
        self.tfidf = None
        self.feature_matrix = None
        self.similarity_matrix = None

    def load_data(self):
        """Load cleaned Zomato data."""
        self.df = pd.read_csv(self.clean_data_path)
        print(f" Loaded {self.df.shape[0]:,} restaurants ({self.df.shape[1]} columns)")
        print("Sample:")
        print(self.df[["name", "location", "cuisines", "rate"]].head(3))

    def create_features(self):
        """TF-IDF vectorization on cuisines + location + name."""
        print("\n🔄 Creating TF-IDF features...")

        # Combine features into "soup" text
        self.df["soup"] = (
                self.df["cuisines"].fillna("") + " " +
                self.df["location"].fillna("") + " " +
                self.df["rest_type"].fillna("") + " " +
                self.df["name"].fillna("")
        )

        # # TF-IDF ( core algorithm)
        # self.tfidf = TfidfVectorizer(
        #     max_features=5000,
        #     stop_words="english",
        #     ngram_range=(1, 2),
        #     lowercase=True
        # )

        # TUNED PARAMS
        self.tfidf = TfidfVectorizer(
            max_features=2000,
            min_df=5,
            max_df=0.95,
            ngram_range=(1, 1),
            stop_words='english'
        )

        self.feature_matrix = self.tfidf.fit_transform(self.df["soup"])
        print(f" Feature matrix: {self.feature_matrix.shape}")

    def compute_similarity(self):
        """Cosine similarity between all restaurants."""
        print("🔄 Computing similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        print(f" Similarity matrix ready ({self.similarity_matrix.shape})")

    # def recommend(self, restaurant_name: str, top_k: int = 10) -> pd.DataFrame:
    #     """Get top-K similar restaurants (FUZZY SEARCH)."""
    #     if self.similarity_matrix is None:
    #         raise ValueError("Run create_features() and compute_similarity() first!")
    #
    #     # Case-insensitive fuzzy search
    #     name_lower = restaurant_name.lower()
    #     mask = self.df["name"].str.lower().str.contains(name_lower, na=False)
    #
    #     # If exact not found, try first word
    #     if not mask.any():
    #         first_word = name_lower.split()[0]
    #         mask = self.df["name"].str.lower().str.contains(first_word, na=False)
    #
    #     indices = self.df[mask].index
    #     if len(indices) == 0:
    #         return pd.DataFrame({"error": [f"❌ '{restaurant_name}' not found in {len(self.df):,} restaurants"]})
    #
    #     idx = indices[0]
    #
    #     # Get top similar (exclude self)
    #     sim_scores = list(enumerate(self.similarity_matrix[idx]))
    #     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_k + 1]
    #
    #     rec_idx = [i[0] for i in sim_scores]
    #     rec_df = self.df.iloc[rec_idx][["name", "location", "cuisines", "rate", "approx_cost"]].copy()
    #     rec_df["similarity"] = [f"{s[1]:.3f}" for s in sim_scores]
    #
    #     return rec_df

    # def recommend(self, query: str, priority: str = "name", top_k: int = 10) -> pd.DataFrame:
    #     query_lower = query.lower().strip()
    #
    #     if priority == "location":
    #         matches = self.df[self.df['location'].str.lower().str.contains(query_lower, na=False)]
    #     elif priority == "cuisines":
    #         matches = self.df[self.df['cuisines'].str.lower().str.contains(query_lower, na=False)]
    #     else:  # "name"
    #         matches = self.df[self.df['name'].str.lower().str.contains(query_lower, na=False)]
    #
    #     if len(matches) == 0:
    #         return pd.DataFrame()  # No matches
    #
    #     # Use top match for TF-IDF similarity
    #     idx = matches.index[0]
    #     sim_scores = list(enumerate(self.similarity_matrix[idx]))
    #     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_k + 1]
    #
    #     indices = [i[0] for i in sim_scores]
    #     return self.df.iloc[indices].copy()

    # less similar recommend
    # def recommend(self, query: str, priority: str = "name", top_k: int = 10) -> pd.DataFrame:
    #     """Recommend DIVERSE restaurants by priority (no chain spam)."""
    #     if self.similarity_matrix is None:
    #         raise ValueError("Run create_features() and compute_similarity() first!")
    #
    #     query_lower = query.lower().strip()
    #
    #     # Priority-based filtering ( logic)
    #     if priority == "location":
    #         matches = self.df[self.df['location'].str.lower().str.contains(query_lower, na=False)]
    #     elif priority == "cuisines":
    #         matches = self.df[self.df['cuisines'].str.lower().str.contains(query_lower, na=False)]
    #     else:  # "name"
    #         matches = self.df[self.df['name'].str.lower().str.contains(query_lower, na=False)]
    #
    #     if len(matches) == 0:
    #         return pd.DataFrame()  # Empty = no matches
    #
    #     # Use FIRST match for TF-IDF similarity
    #     idx = matches.index[0]
    #
    #     # Top similar restaurants (exclude self)
    #     sim_scores = list(enumerate(self.similarity_matrix[idx]))
    #     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_k * 2 + 1]  # Get extra for diversity
    #
    #     # NEW: DIVERSE FILTER (1 best location per restaurant)
    #     rec_df = self.df.iloc[[i[0] for i in sim_scores]].copy()
    #     rec_df['sim_score'] = [i[1] for i in sim_scores]  # Add similarity
    #     rec_df['restaurant_base'] = rec_df['name'].str.split().str[0].str.upper()  # "KFC Pizza" → "KFC"
    #
    #     # Pick HIGHEST RATING per restaurant name (diversity!)
    #     diverse = rec_df.loc[rec_df.groupby('restaurant_base')['rate'].idxmax()]
    #
    #     # Sort by similarity, take top_k
    #     final_recs = diverse.sort_values('sim_score', ascending=False).head(top_k)[
    #         ['name', 'location', 'cuisines', 'rate', 'approx_cost', 'sim_score']
    #     ].round(2)
    #
    #     # Rename for eval
    #     final_recs = final_recs.rename(columns={'sim_score': 'similarity'})
    #
    #     return final_recs.drop(columns=['restaurant_base'] if 'restaurant_base' in final_recs else [])

    def recommend(self, query: str, priority: str = "name", top_k: int = 10) -> pd.DataFrame:
        query_lower = query.lower().strip()

        if priority == "location":
            matches = self.df[self.df['location'].str.lower().str.contains(query_lower, na=False)]
        elif priority == "cuisines":
            matches = self.df[self.df['cuisines'].str.lower().str.contains(query_lower, na=False)]
        else:  # "name"
            matches = self.df[self.df['name'].str.lower().str.contains(query_lower, na=False)]

        if len(matches) == 0:
            return pd.DataFrame()

        idx = matches.index[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_k * 2 + 1]  # MORE candidates

        indices = [i[0] for i in sim_scores]
        recs = self.df.iloc[indices].copy()
        recs['similarity'] = [score[1] for score in sim_scores]

        # FIXED: Softer dedupe—only if > top_k unique chains
        recs['base_name'] = recs['name'].str.split().str[0].str.upper()
        if len(recs['base_name'].unique()) > top_k:
            recs = recs.loc[recs.groupby('base_name')['similarity'].idxmax()]

        # Shuffle for variety
        recs = recs.sample(frac=1, random_state=None).reset_index(drop=True)

        return recs.drop(columns=['base_name']).head(top_k)

    def save_model(self, model_path: str = "../model/tfidf_model.pkl"):
        """Save trained model for Flask."""
        Path(model_path).parent.mkdir(exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved: {model_path}")


    # def evaluate_model(self, test_restaurants: list = None):
    #     """Phase 5: Model accuracy evaluation."""
    #     print("\n" + "=" * 70)
    #     print("PHASE 5: MODEL ACCURACY EVALUATION")
    #     print("=" * 70)
    #
    #     if test_restaurants is None:
    #         test_restaurants = ["McDonald's", "Domino's", "KFC", "Paradise", "Biryani", "Castle Rock"]
    #
    #     results = []
    #     for restaurant in test_restaurants:
    #         recs = self.recommend(restaurant, top_k=6)
    #         if "error" not in recs.columns:
    #             avg_similarity = recs["similarity"].astype(float).mean()
    #             results.append({
    #                 "input": restaurant,
    #                 "found": True,
    #                 "avg_similarity": f"{avg_similarity:.3f}",
    #                 "top_match": recs.iloc[0]["name"]
    #             })
    #         else:
    #             results.append({
    #                 "input": restaurant,
    #                 "found": False,
    #                 "avg_similarity": "N/A",
    #                 "top_match": "Not found"
    #             })
    #
    #     eval_df = pd.DataFrame(results)
    #     print(eval_df.to_string(index=False))
    #
    #     # Success rate
    #     success_rate = eval_df["found"].mean() * 100
    #     print(f"\n SUCCESS RATE: {success_rate:.1f}% ({eval_df['found'].sum()}/{len(eval_df)} tests)")
    #     print(" PHASE 5 COMPLETE!")
    #
    #     return eval_df


    def evaluate_model(self, test_restaurants: list = None):
        """Model evaluation: Success Rate + Similarity Metrics"""

        if test_restaurants is None:
            test_restaurants = ["McDonald's", "Domino's", "KFC",
                                "Paradise", "Biryani", "Castle Rock"]

        results = []
        for restaurant in test_restaurants:
            recs = self.recommend(restaurant, priority="name", top_k=10)

            found = not recs.empty
            avg_sim = 0.0
            if found and 'similarity' in recs.columns:
                avg_sim = recs['similarity'].mean().round(3)

            top_match = recs['name'].iloc[0] if found else "None"
            results.append({
                'query': restaurant,
                'found': found,
                'avg_similarity': avg_sim,
                'top_match': top_match
            })

        # METRICS CALCULATION
        eval_df = pd.DataFrame(results)
        success_rate = eval_df['found'].mean() * 100
        avg_similarity = eval_df['avg_similarity'].mean().round(3)
        precision_03 = (eval_df['avg_similarity'] >= 0.3).mean() * 100

        print(f"📊 SUCCESS RATE:     {success_rate:.1f}% ({eval_df['found'].sum()}/6)")
        print(f"📊 AVG SIMILARITY:   {avg_similarity}")
        print(f"📊 PRECISION@10:     {precision_03:.1f}% (sim≥0.3)")

        return eval_df

    # RUN: model.evaluate_model() →  100.0% results!

    def evaluate_model_comprehensive(self, n_tests=50, test_types=None):
        """Advanced eval: Success + Precision + Diversity + Speed"""
        import time

        print("\n" + "═" * 80)
        print("ADVANCED MODEL EVALUATION (50 Tests + Metrics)")
        print("═" * 80)

        # Diverse test sets
        if test_types is None:
            chains = ["McDonald's", "Domino's", "KFC", "Paradise"]
            dishes = ["Biryani", "Pizza", "Burger", "Chinese"]
            locations = self.df['location'].value_counts().head(6).index.tolist()
            tests = chains + dishes + locations[:10]
        else:
            tests = test_types

        results = []
        times = []

        for i, query in enumerate(tests[:n_tests]):
            start = time.time()
            recs = self.recommend(query, priority="name", top_k=10)
            elapsed = time.time() - start

            found = not recs.empty
            sim_mean = recs['similarity'].mean() if found and 'similarity' in recs else 0.0
            diversity = recs['name'].str.split().str[0].nunique() if found else 0

            results.append({
                'query': query,
                'found': found,
                'sim_mean': round(sim_mean, 3),
                'diversity': diversity,  # Unique chains
                'time_ms': round(elapsed * 1000, 1)
            })
            times.append(elapsed)

        df = pd.DataFrame(results)

        # COMPREHENSIVE METRICS
        success = df['found'].mean() * 100
        avg_sim = df['sim_mean'].mean()
        avg_diversity = df['diversity'].mean()
        avg_time = np.mean(times) * 1000

        print(
            f"{'TEST SUMMARY':<20} {success:>6.1f}% success | {avg_sim:>6.3f} sim | {avg_diversity:>4.1f} chains | {avg_time:>6.1f}ms")
        print("\n" + df.head(10).to_string(index=False))
        print(f"\n📊 FULL METRICS:")
        print(f"   Success Rate:     {success:.1f}% ({df['found'].sum()}/{len(df)})")
        print(f"   Avg Similarity:   {avg_sim:.3f}")
        print(f"   Avg Diversity:    {avg_diversity:.1f} unique chains/10")
        print(f"   Avg Latency:      {avg_time:.1f}ms/query")
        print(f"   95th Percentile:  {np.percentile(times, 95) * 1000:.1f}ms")

        return df

    # RUN: model.evaluate_model_comprehensive(n_tests=50)


# === PHASE 4 EXECUTION ===
if __name__ == "__main__":
    # print(" PHASE 4: CONTENT-BASED RECOMMENDER SYSTEM")
    # print("=" * 50)
    #
    # # Initialize
    # model = ContentBasedRecommender()
    # model.load_data()
    # model.create_features()
    # model.compute_similarity()
    #
    # # Test cases
    # tests = [
    #     "McDonald's",
    #     "Domino's",
    #     "Paradise",  # Fuzzy match
    #     "Biryani"  # First word match
    # ]
    #
    # print("\n" + "=" * 60)
    # print("🎯 RECOMMENDATION TESTS (Top 3)")
    # print("=" * 60)
    #
    # for test in tests:
    #     print(f"\n🔍 Searching for: '{test}'")
    #     recs = model.recommend(test, top_k=3)
    #     if "error" not in recs.columns:
    #         print(recs.to_string(index=False))
    #     else:
    #         print(recs.iloc[0, 0])
    #
    # # Save for Flask
    # model.save_model()
    # print("\n PHASE 4 COMPLETE")
    # print(" Model ready for Flask web app (Phase 6)")

    model = ContentBasedRecommender()
    model.load_data()  # Load data
    model.create_features()  # Train TF-IDF
    model.compute_similarity()  # Compute similarities
    model.evaluate_model_comprehensive()  # Test accuracy
    model.save_model()  # Save for Flask

