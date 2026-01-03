from flask import Flask, render_template, request
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# CRITICAL: Import  model class (fixes pickle error)
from src.content_based import ContentBasedRecommender

app = Flask(__name__)

print("Loading model...")
model_path = Path("model/tfidf_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)
print("Model loaded! Ready for recommendations.")


# @app.route("/", methods=["GET", "POST"])
# def index():
#     recommendations = None
#     query = ""
#     error = None
#
#     if request.method == "POST":
#         query = request.form.get("restaurant", "").strip()
#         if query:
#             try:
#                 if hasattr(model, 'recommend'):
#                     recommendations = model.recommend(query, top_k=10)
#                 else:
#                     error = "Model recommend() not found. Run src/content_based.py first."
#             except Exception as e:
#                 error = f"Error: {str(e)}"
#
#     return render_template("home.html",
#                            recommendations=recommendations,
#                            query=query,
#                            error=error)

#
# @app.route("/", methods=["GET", "POST"])
# def index():
#     recommendations = None
#     query = ""
#     error = None
#
#     if request.method == "POST":
#         query = request.form.get("restaurant", "").strip()
#         if query:
#             try:
#                 if hasattr(model, 'recommend'):
#                     recommendations = model.recommend(query, top_k=10)
#                     if recommendations is not None:
#                         recommendations['rate_display'] = recommendations['rate'].fillna('New')
#                         recommendations['approx_cost'] = recommendations['approx_cost'].fillna(0).astype(int)
#                 else:
#                     error = "Model recommend() not found."
#             except Exception as e:
#                 error = f"Error: {str(e)}"
#
#     return render_template("home.html",
#                            recommendations=recommendations,
#                            query=query,
#                            error=error)
#
# @app.route("/", methods=["GET", "POST"])
# def index():
#     recommendations = None
#     query = ""
#     priority = "name"  # Default
#     error = None
#
#     if request.method == "POST":
#         query = request.form.get("restaurant", "").strip()
#         priority = request.form.get("priority", "name")
#
#         if query:
#             try:
#                 recommendations = model.recommend(query, priority=priority, top_k=10)
#                 if recommendations is not None:
#                     recommendations['rate_display'] = recommendations['rate'].fillna('New')
#                     recommendations['approx_cost'] = recommendations['approx_cost'].fillna(0).astype(int)
#             except Exception as e:
#                 error = f"Error: {str(e)}"
#
#     return render_template("home2.html",
#                            recommendations=recommendations,
#                            query=query,
#                            priority=priority,
#                            error=error)

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None
    query = ""
    priority = "name"  # Default
    error = None
    summary = {}  # NEW: For visualization

    if request.method == "POST":
        query = request.form.get("restaurant", "").strip()
        priority = request.form.get("priority", "name")

        if query:
            try:
                recommendations = model.recommend(query, priority=priority, top_k=10)
                if recommendations is not None and not recommendations.empty:
                    # Safe column checks prevent KeyError ( CODE UNCHANGED)
                    if 'rate' in recommendations.columns:
                        recommendations['rate_display'] = recommendations['rate'].fillna('New')
                    else:
                        recommendations['rate_display'] = 'New'

                    if 'approx_cost' in recommendations.columns:
                        recommendations['approx_cost'] = recommendations['approx_cost'].fillna(0).astype(int)
                    else:
                        recommendations['approx_cost'] = 0

                    # NEW: Summary stats for charts (6 lines)
                    valid_ratings = recommendations['rate'].dropna()
                    summary = {
                        # 'avg_rating': recommendations['rate'].fillna(0).mean().round(1),
                        'avg_rating': valid_ratings.mean().round(1) if len(valid_ratings) > 0 else 0,
                        'top_cost': int(recommendations['approx_cost'].max()),
                        'count': len(recommendations),
                        'rated_count': len(valid_ratings),  # NEW: Rated only
                        # 'names': recommendations['name'].tolist()[:5],
                        'names': [f"{name} ({loc})" for name, loc in
                                  zip(recommendations['name'].tolist()[:5], recommendations['location'].tolist()[:5])],
                        'ratings': recommendations['rate'].fillna(0).tolist()[:5]
                    }

                else:
                    error = f"No matches found for '{query}' with {priority} priority. Try 'BTM' (location), 'KFC' (name), or 'Biryani' (dish)."
            except Exception as e:
                error = f"Error: {str(e)}"

    return render_template("home2.html",
                           recommendations=recommendations,
                           query=query,
                           priority=priority,
                           error=error,
                           summary=summary)  # NEW: Pass to HTML


if __name__ == "__main__":
    print("🚀 Starting Flask app at http://localhost:5000")
    app.run(debug=True, port=5000, host='127.0.0.1')
