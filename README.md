

Movie Recommendation System — SVD Collaborative Filtering

A movie recommendation engine built using Matrix Factorization (SVD) on the MovieLens 100K dataset. The system learns latent user and movie preferences from historical ratings and predicts what a user will enjoy watching next.

 Results

| Metric | Score |
|--------|-------|
| RMSE (test set) | 0.7849 |
| MAE (test set) | 0.6321|
| CV RMSE (5-fold) | 0.7891 ± 0.0081|
| Dataset | 100K ratings, 943 users, 1682 movies |


How It Works

Collaborative filtering works on a simple idea: users who agreed in the past tend to agree in the future. Instead of knowing anything about movies themselves (genres, cast, etc.), the model learns purely from the pattern of who rated what.

SVD (Singular Value Decomposition)decomposes the sparse user-movie matrix into latent factor vectors. Each user gets a vector representing their hidden preferences, each movie gets a vector representing its hidden attributes. The dot product of these vectors predicts ratings.


predicted_rating(u, i) = global_mean + bias_u + bias_i + (user_factors_u · item_factors_i)

Project Structure

movie-recommender/
│
├── data/
│   ├── ratings.csv          100K user-movie ratings
│   ├── movies.csv           Movie metadata + genre flags  
│   └── users.csv            User demographics
│
├── src/
│   ├── recommender.py       Core ML: SVD training, evaluation, recommendations
│   └── visualize.py         All plots and charts
│
├── outputs/                 Generated plots and CSV results
│   ├── rating_distribution.png
│   ├── ratings_per_user.png
│   ├── ratings_per_movie.png
│   ├── top_movies.png
│   ├── genre_popularity.png
│   ├── rmse_vs_k.png
│   ├── actual_vs_predicted.png
│   ├── error_distribution.png
│   ├── test_predictions.csv
│   └── model_summary.csv
│
├── main.py                 Entry point — runs everything
├── requirements.txt
└── README.md


Setup & Run

bash
1. Clone the repo
git clone https://github.com/YOUR_USERNAME/movie-recommender.git
cd movie-recommender

2. Install dependencies
pip install -r requirements.txt

3. Run the full pipeline
python main.py

Dependencies

- Python 3.8+
- scikit-surprise (SVD)
- pandas, numpy
- matplotlib, seaborn
- scikit-learn


Sample Output

Dataset Overview
Total ratings   : 100,000
Unique users    : 943
Unique movies   : 1682
Matrix sparsity : 93.70%

Best params: n_factors=20, n_epochs=30, lr=0.01, reg=0.1
Test RMSE  : 0.7849
Test MAE   : 0.6321
CV RMSE    : 0.7891 ± 0.0081


What the Model Does

1. Loads100K ratings and runs basic EDA
2. Tunes hyperparameters (latent factors, learning rate, regularization) via grid search
3. Trains SVD on 80% of data, evaluates on 20% held-out
4. Validates with 5-fold cross-validation
5. Generates top-10 personalized recommendations for any user
6. Finds similar movies using cosine similarity on learned item vectors
7. Saves all plots and metrics to /outputs

Dataset

MovieLens 100K format — 100,000 ratings from 943 users on 1,682 movies. Ratings are on a 1–5 scale.
Original dataset: https://grouplens.org/datasets/movielens/100k/


Author: Bhukya Rakesh
