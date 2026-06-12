import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120


def plot_rating_distribution(ratings, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = ratings['rating'].value_counts().sort_index()
    colors = ['#d9534f','#f0ad4e','#aec7e8','#5cb85c','#337ab7']
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+300,
                f'{val:,}', ha='center', va='bottom', fontsize=10)
    ax.set_xlabel('Rating', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Rating Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks([1,2,3,4,5])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches='tight'); print(f"Saved: {save_path}")
    plt.close()


def plot_ratings_per_user(ratings, save_path=None):
    per_user = ratings.groupby('user_id')['rating'].count()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(per_user, bins=50, color='#5bc0de', edgecolor='white', linewidth=0.5)
    ax.axvline(per_user.mean(), color='#d9534f', linestyle='--', lw=1.5,
               label=f'Mean = {per_user.mean():.0f}')
    ax.set_xlabel('Ratings per User', fontsize=12)
    ax.set_ylabel('Number of Users', fontsize=12)
    ax.set_title('Ratings per User Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches='tight'); print(f"Saved: {save_path}")
    plt.close()


def plot_ratings_per_movie(ratings, save_path=None):
    per_movie = ratings.groupby('movie_id')['rating'].count()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(per_movie, bins=60, color='#337ab7', edgecolor='white', linewidth=0.5)
    ax.axvline(per_movie.mean(), color='#f0ad4e', linestyle='--', lw=1.5,
               label=f'Mean = {per_movie.mean():.0f}')
    ax.set_xlabel('Ratings per Movie', fontsize=12)
    ax.set_ylabel('Number of Movies', fontsize=12)
    ax.set_title('Ratings per Movie Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches='tight'); print(f"Saved: {save_path}")
    plt.close()


def plot_top_movies(ratings, movies, save_path=None, n=15):
    counts = ratings.groupby('movie_id')['rating'].count()
    popular = counts[counts >= 40].index
    avg = ratings[ratings['movie_id'].isin(popular)].groupby('movie_id')['rating'].mean()
    top = avg.sort_values(ascending=False).head(n).reset_index()
    top = top.merge(movies[['movie_id','title']], on='movie_id')

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top['title'], top['rating'], color='#5cb85c', edgecolor='white')
    for bar, val in zip(bars, top['rating']):
        ax.text(bar.get_width()+0.02, bar.get_y()+bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=9)
    ax.set_xlabel('Average Rating', fontsize=12)
    ax.set_title(f'Top {n} Highest Rated Movies (min 40 ratings)', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0, 5.6)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches='tight'); print(f"Saved: {save_path}")
    plt.close()


def plot_genre_popularity(movies, ratings, save_path=None):
    genres = ['Action','Adventure','Animation','Children','Comedy','Crime',
              'Documentary','Drama','Fantasy','Horror','Musical','Mystery',
              'Romance','Sci-Fi','Thriller','War','Western']
    gc = {}
    for g in genres:
        if g in movies.columns:
            mids = movies[movies[g]==1]['movie_id']
            gc[g] = ratings[ratings['movie_id'].isin(mids)].shape[0]
    gc_df = pd.DataFrame({'genre':list(gc.keys()),'count':list(gc.values())}).sort_values('count')
    fig, ax = plt.subplots(figsize=(9,7))
    ax.barh(gc_df['genre'], gc_df['count'], color='#9b59b6', edgecolor='white')
    ax.set_xlabel('Total Ratings', fontsize=12)
    ax.set_title('Total Ratings by Genre', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches='tight'); print(f"Saved: {save_path}")
    plt.close()


def plot_rmse_vs_k(rmse_list, k_values, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, rmse_list, marker='o', color='#337ab7', linewidth=2, markersize=7)
    best_k = k_values[int(np.argmin(rmse_list))]
    best_r = min(rmse_list)
    ax.axvline(best_k, color='#d9534f', linestyle='--', lw=1.3,
               label=f'Best k={best_k}  RMSE={best_r:.4f}')
    ax.set_xlabel('Latent Factors (k)', fontsize=12)
    ax.set_ylabel('RMSE (3-fold CV)', fontsize=12)
    ax.set_title('SVD: RMSE vs Number of Latent Factors', fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches='tight'); print(f"Saved: {save_path}")
    plt.close()


def plot_prediction_vs_actual(preds_df, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(preds_df['actual'], preds_df['predicted'],
               alpha=0.25, color='#337ab7', s=12, label='Predictions')
    ax.plot([1,5],[1,5], color='#d9534f', lw=1.5, linestyle='--', label='Perfect fit')
    ax.set_xlabel('Actual Rating', fontsize=12)
    ax.set_ylabel('Predicted Rating', fontsize=12)
    ax.set_title('Actual vs Predicted Ratings', fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches='tight'); print(f"Saved: {save_path}")
    plt.close()


def plot_error_distribution(preds_df, save_path=None):
    errors = preds_df['actual'] - preds_df['predicted']
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=60, color='#f0ad4e', edgecolor='white', linewidth=0.4)
    ax.axvline(0, color='#d9534f', linestyle='--', lw=1.5, label='Zero error')
    ax.axvline(errors.mean(), color='#337ab7', linestyle='-', lw=1.5,
               label=f'Mean error = {errors.mean():.3f}')
    ax.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Prediction Errors', fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches='tight'); print(f"Saved: {save_path}")
    plt.close()
