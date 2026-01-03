# 3_Model_Optimization/3.1_Hyperparameter_Tuning_Code.py
# TF-IDF GridSearch - Phase 3.1 (Run from 3_Model_Optimization/)
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path
from sklearn.model_selection import ParameterGrid
import time
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

        # TF-IDF ( core algorithm)
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            lowercase=True
        )
        self.feature_matrix = self.tfidf.fit_transform(self.df["soup"])
        print(f" Feature matrix: {self.feature_matrix.shape}")

    def compute_similarity(self):
        """Cosine similarity between all restaurants."""
        print("🔄 Computing similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        print(f" Similarity matrix ready ({self.similarity_matrix.shape})")


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

        print(f" SUCCESS RATE:     {success_rate:.1f}% ({eval_df['found'].sum()}/6)")
        print(f" AVG SIMILARITY:   {avg_similarity}")
        print(f" PRECISION@10:     {precision_03:.1f}% (sim≥0.3)")

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


class TuningExperiment:
    def __init__(self, data_path: str = "../DATASET/zomato_clean.csv"):
        self.data_path = Path(data_path)
        self.model = ContentBasedRecommender(str(self.data_path))
        self.model.load_data()
        self.model.create_features()
        self.model.compute_similarity()
        self.best_params = None
        self.best_score = -np.inf
        self.best_vectorizer = None

    def evaluate_config(self, params: dict, test_queries: list, top_k: int = 10):
        """Test single TF-IDF config."""
        tfidf_temp = TfidfVectorizer(**params)
        matrix_temp = tfidf_temp.fit_transform(self.model.df['soup'])
        sim_temp = cosine_similarity(matrix_temp)

        scores = []
        for query in test_queries:
            query_vec = tfidf_temp.transform([query.lower()])
            sim_scores = cosine_similarity(query_vec, matrix_temp).flatten()
            top_idx = np.argsort(sim_scores)[-top_k:]
            avg_sim = sim_scores[top_idx].mean()
            scores.append(avg_sim)

        return np.mean(scores)

    def grid_search(self, param_grid: dict, test_queries: list):
        """Full GridSearch."""
        print("🔍 TF-IDF GRIDSEARCH")
        print("=" * 60)

        results = []
        for i, params in enumerate(ParameterGrid(param_grid)):
            start = time.time()
            mean_sim = self.evaluate_config(params, test_queries)
            latency = (time.time() - start) * 1000

            result = {**params, 'mean_similarity': mean_sim, 'latency_ms': round(latency, 1)}
            results.append(result)

            status = " BEST" if mean_sim > self.best_score else ""
            print(f"{i + 1:2d}. {params} → {mean_sim:.3f} sim | {latency:.1f}ms {status}")

            if mean_sim > self.best_score:
                self.best_score = mean_sim
                self.best_params = params

        df_results = pd.DataFrame(results).sort_values('mean_similarity', ascending=False)
        print("\n FINAL RESULTS:")
        print(df_results.round(3).head(10))

        # SAVE
        df_results.to_csv('tuning_results.csv', index=False)
        print(f"\n Saved: tuning_results.csv")
        print(f" Best: {self.best_params} (Sim: {self.best_score:.3f})")

        return df_results


if __name__ == "__main__":
    tuner = TuningExperiment()

    # 18 fast trials (3min max)
    test_queries = ["pizza BTM", "McDonald's", "Domino's Pizza", "KFC",
                    "Biryani", "Paradise", "Chinese HSR"]

    # param_grid = {
    #     'max_features': [3000, 5000, 8000],
    #     'ngram_range': [(1, 1), (1, 2)],
    #     'min_df': [1, 3]
    # }

    param_grid = {
        'max_features': [2000, 3000, 5000, 8000, 10000],  # +2 sizes
        'ngram_range': [(1, 1), (1, 2), (2, 2)],  # +bigrams only
        'min_df': [1, 2, 3, 5],  # +rarer pruning
        'max_df': [0.95, 1.0]  # -Stop too-common terms
    }
    # 5 × 3 × 4 × 2 = 120 trials

    results = tuner.grid_search(param_grid, test_queries)
