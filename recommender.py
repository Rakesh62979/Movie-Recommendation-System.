import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV
import warnings
warnings.filterwarnings('ignore')


def load_data(ratings_path, movies_path, users_path):
    ratings = pd.read_csv(ratings_path)
    movies  = pd.read_csv(movies_path)
    users   = pd.read_csv(users_path)
    return ratings, movies, users


def basic_stats(ratings, movies, users):
    print("=" * 52)
    print("  Dataset Overview")
    print("=" * 52)
    print(f"  Total ratings   : {len(ratings):,}")
    print(f"  Unique users    : {ratings['user_id'].nunique()}")
    print(f"  Unique movies   : {ratings['movie_id'].nunique()}")
    print(f"  Rating scale    : {ratings['rating'].min()} to {ratings['rating'].max()}")
    print(f"  Avg rating      : {ratings['rating'].mean():.3f}")
    sparsity = 1 - (len(ratings) / (ratings['user_id'].nunique() * ratings['movie_id'].nunique()))
    print(f"  Matrix sparsity : {sparsity*100:.2f}%")
    print("=" * 52)


def build_surprise_dataset(ratings):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
    return data


def tune_svd(data):
    # grid search over key hyperparameters
    param_grid = {
        'n_factors': [20, 50, 100],
        'n_epochs':  [20, 30],
        'lr_all':    [0.005, 0.01],
        'reg_all':   [0.02, 0.1]
    }
    print("Running grid search (this takes ~2 min)...")
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
    gs.fit(data)
    best_params = gs.best_params['rmse']
    best_score  = gs.best_score['rmse']
    print(f"Best params : {best_params}")
    print(f"Best CV RMSE: {best_score:.4f}")
    return best_params, best_score


def train_final_model(data, params, test_size=0.2):
    trainset, testset = train_test_split(data, test_size=test_size, random_state=7)
    model = SVD(
        n_factors=params['n_factors'],
        n_epochs=params['n_epochs'],
        lr_all=params['lr_all'],
        reg_all=params['reg_all'],
        random_state=42
    )
    model.fit(trainset)
    preds = model.test(testset)
    rmse = accuracy.rmse(preds, verbose=False)
    mae  = accuracy.mae(preds, verbose=False)
    return model, trainset, preds, rmse, mae


def cross_val_eval(data, params, cv=5):
    model = SVD(
        n_factors=params['n_factors'],
        n_epochs=params['n_epochs'],
        lr_all=params['lr_all'],
        reg_all=params['reg_all'],
        random_state=42
    )
    results = cross_validate(model, data, measures=['rmse','mae'], cv=cv, verbose=False)
    return results


def get_recommendations(user_id, model, ratings, movies, n=10):
    # all movies this user hasn't rated yet
    rated = set(ratings[ratings['user_id'] == user_id]['movie_id'].tolist())
    all_movies = set(ratings['movie_id'].unique())
    unrated = list(all_movies - rated)

    preds = [(mid, model.predict(user_id, mid).est) for mid in unrated]
    preds.sort(key=lambda x: x[1], reverse=True)
    top = preds[:n]

    rec_df = pd.DataFrame(top, columns=['movie_id', 'predicted_rating'])
    rec_df = rec_df.merge(movies[['movie_id', 'title']], on='movie_id', how='left')
    rec_df = rec_df[['movie_id', 'title', 'predicted_rating']].reset_index(drop=True)
    rec_df['predicted_rating'] = rec_df['predicted_rating'].round(3)
    rec_df.index += 1
    return rec_df


def get_similar_movies(movie_id, model, movies, n=8):
    # compare item latent vectors using cosine similarity
    qi = model.qi  # item factors matrix
    inner_id = model.trainset.to_inner_iid(movie_id)
    movie_vec = qi[inner_id]

    sims = []
    for iid in range(len(qi)):
        if iid == inner_id:
            continue
        v = qi[iid]
        cos_sim = np.dot(movie_vec, v) / (np.linalg.norm(movie_vec) * np.linalg.norm(v) + 1e-9)
        raw_id = model.trainset.to_raw_iid(iid)
        sims.append((raw_id, cos_sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    top = sims[:n]

    sim_df = pd.DataFrame(top, columns=['movie_id', 'similarity'])
    sim_df = sim_df.merge(movies[['movie_id', 'title']], on='movie_id', how='left')
    sim_df = sim_df[['movie_id', 'title', 'similarity']].reset_index(drop=True)
    sim_df['similarity'] = sim_df['similarity'].round(4)
    sim_df.index += 1
    return sim_df


def get_predictions_df(preds):
    rows = []
    for p in preds:
        rows.append({
            'user_id':    p.uid,
            'movie_id':   p.iid,
            'actual':     p.r_ui,
            'predicted':  round(p.est, 3),
            'error':      round(abs(p.r_ui - p.est), 3)
        })
    return pd.DataFrame(rows)
