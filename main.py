"""
Movie Recommendation System -- SVD Collaborative Filtering
Dataset : MovieLens 100K format (943 users, 1682 movies, 100K ratings)
Method  : Matrix Factorization via SVD (Surprise library)
Author  : Bhukya Rakesh

import os, sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.recommender import (
    load_data, basic_stats, build_surprise_dataset,
    tune_svd, train_final_model, cross_val_eval,
    get_recommendations, get_similar_movies, get_predictions_df
)
from src.visualize import (
    plot_rating_distribution, plot_ratings_per_user,
    plot_ratings_per_movie, plot_top_movies,
    plot_genre_popularity, plot_rmse_vs_k,
    plot_prediction_vs_actual, plot_error_distribution
)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
OUT  = os.path.join(BASE, 'outputs')
os.makedirs(OUT, exist_ok=True)


def main():
    
    1. Load data
    print("\nLoading data...")
    ratings, movies, users = load_data(
        os.path.join(DATA, 'ratings.csv'),
        os.path.join(DATA, 'movies.csv'),
        os.path.join(DATA, 'users.csv')
    )
    basic_stats(ratings, movies, users)
    print("\nGenerating EDA plots...")
    plot_rating_distribution(ratings,      save_path=os.path.join(OUT, 'rating_distribution.png'))
    plot_ratings_per_user(ratings,         save_path=os.path.join(OUT, 'ratings_per_user.png'))
    plot_ratings_per_movie(ratings,        save_path=os.path.join(OUT, 'ratings_per_movie.png'))
    plot_top_movies(ratings, movies,       save_path=os.path.join(OUT, 'top_movies.png'))
    plot_genre_popularity(movies, ratings, save_path=os.path.join(OUT, 'genre_popularity.png'))

    # ------------------------------------------------------------------
    # 3. Build Surprise dataset
    # ------------------------------------------------------------------
    data = build_surprise_dataset(ratings)

   
    # 4. Hyperparameter tuning
    # ------------------------------------------------------------------
    print("\nHyperparameter tuning...")
    best_params, best_cv_rmse = tune_svd(data)

    # ------------------------------------------------------------------
    # 5. Train/test split evaluation
    # ------------------------------------------------------------------
    print("\nTraining final model...")
    model, trainset, preds, rmse, mae = train_final_model(data, best_params)

    print(f"\n{'='*52}")
    print("  Model Evaluation (80/20 split)")
    print(f"{'='*52}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  Evaluated on {len(preds):,} test ratings")
    print(f"{'='*52}")

    # ------------------------------------------------------------------
    # 6. 5-fold Cross validation
    # ------------------------------------------------------------------
    print("\nRunning 5-fold cross-validation...")
    cv_results = cross_val_eval(data, best_params, cv=5)
    cv_rmse_mean = np.mean(cv_results['test_rmse'])
    cv_rmse_std  = np.std(cv_results['test_rmse'])
    cv_mae_mean  = np.mean(cv_results['test_mae'])
    print(f"  CV RMSE : {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}")
    print(f"  CV MAE  : {cv_mae_mean:.4f}")

    # ------------------------------------------------------------------
    # 7. Visualize predictions
    # ------------------------------------------------------------------
    preds_df = get_predictions_df(preds)
    plot_prediction_vs_actual(preds_df, save_path=os.path.join(OUT, 'actual_vs_predicted.png'))
    plot_error_distribution(preds_df,   save_path=os.path.join(OUT, 'error_distribution.png'))

    # k tuning curve -- reuse CV results across k values
    print("\nGenerating RMSE vs k curve...")
    from surprise import SVD as SurpriseSVD
    from surprise.model_selection import cross_validate as cv_fn
    k_vals, k_rmse = [], []
    for k in [10, 20, 30, 50, 75, 100, 150]:
        m = SurpriseSVD(n_factors=k, n_epochs=20, lr_all=best_params['lr_all'],
                        reg_all=best_params['reg_all'], random_state=42)
        r = cv_fn(m, data, measures=['rmse'], cv=3, verbose=False)
        k_vals.append(k)
        k_rmse.append(np.mean(r['test_rmse']))
        print(f"  k={k:4d}  RMSE={k_rmse[-1]:.4f}")
    plot_rmse_vs_k(k_rmse, k_vals, save_path=os.path.join(OUT, 'rmse_vs_k.png'))

    # ------------------------------------------------------------------
    # 8. Sample recommendations
    # ------------------------------------------------------------------
    print(f"\n{'='*52}")
    print("  Sample Top-10 Recommendations")
    print(f"{'='*52}")
    for uid in [1, 42, 100, 250, 500]:
        n_rated = ratings[ratings['user_id']==uid].shape[0]
        avg_r   = ratings[ratings['user_id']==uid]['rating'].mean()
        print(f"\nUser {uid}  (rated {n_rated} movies, avg={avg_r:.2f})")
        recs = get_recommendations(uid, model, ratings, movies, n=10)
        print(recs.to_string())

    # ------------------------------------------------------------------
    # 9. Similar movies
    # ------------------------------------------------------------------
    sample_mid = ratings['movie_id'].value_counts().index[0]
    sample_title = movies[movies['movie_id']==sample_mid]['title'].values[0]
    print(f"\n{'='*52}")
    print(f"  Movies Similar to: '{sample_title}' (id={sample_mid})")
    print(f"{'='*52}")
    sim_movies = get_similar_movies(sample_mid, model, movies, n=8)
    print(sim_movies.to_string())

    # ------------------------------------------------------------------
    # 10. Save outputs
    # ------------------------------------------------------------------
    preds_df.to_csv(os.path.join(OUT, 'test_predictions.csv'), index=False)
    pd.DataFrame({'k': k_vals, 'RMSE': [round(r,4) for r in k_rmse]}).to_csv(
        os.path.join(OUT, 'rmse_vs_k.csv'), index=False)
    metrics_summary = pd.DataFrame([{
        'best_k':        best_params['n_factors'],
        'n_epochs':      best_params['n_epochs'],
        'lr':            best_params['lr_all'],
        'reg':           best_params['reg_all'],
        'test_RMSE':     round(rmse, 4),
        'test_MAE':      round(mae, 4),
        'cv_RMSE_mean':  round(cv_rmse_mean, 4),
        'cv_RMSE_std':   round(cv_rmse_std, 4),
    }])
    metrics_summary.to_csv(os.path.join(OUT, 'model_summary.csv'), index=False)
    print(f"\nAll outputs saved to /outputs/")
    print("Done!\n")


if __name__ == '__main__':
    main()
