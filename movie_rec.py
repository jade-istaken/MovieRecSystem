#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random

import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from plotting import generate_coverage_report
from grid_search_viz import generate_grid_search_report
from baselines import generate_baseline_report
#global behavioral variables go here 
dataset_folder = 'dataset'
min_ratings = 40 #threshold for minimum ratings
num_neighbors = 350
alpha=0.90


class HybridUserClusterKNNRecommender(BaseEstimator, ClusterMixin):
    def __init__(self, n_neighbors=10, min_ratings=20, alpha=0.9):
        self.n_neighbors = n_neighbors
        self.min_ratings = min_ratings
        self.alpha = alpha  # Weight for CF (1-alpha for Content)

        # CF estimators
        self._scaler = StandardScaler()
        self._meanshift = MeanShift(bin_seeding=True, cluster_all=True, n_jobs=-1)
        self._nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
        
        # Content estimators
        self._genre_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        self._movie_genre_matrix = None
        self._user_genre_profiles = None
        self._all_movie_ids = None
        self._movie_to_all_idx = None

    def fit(self, ratings_df, movies_df, y=None):
        """Collaborative Filtering Setup"""
        user_movie = ratings_df.pivot_table(index='UserID', columns='MovieID', values='Rating')
        active_users = user_movie.count(axis=1) >= self.min_ratings
        active_movies = user_movie.count(axis=0) >= self.min_ratings
        df = user_movie.loc[active_users, active_movies].fillna(0)
        if df.empty:
            raise ValueError("No users/movies meet min_ratings threshold.")

        self._user_ids = [int(uid) for uid in df.index]
        self._movie_ids = [int(mid) for mid in df.columns]
        self._X = df.values
        # Precompute per-user mean rating (ignores 0/unrated)
        user_rating_counts = np.sum(self._X > 0, axis=1)
        self._user_means = np.sum(self._X, axis=1) / np.maximum(user_rating_counts, 1)
        
        self._movies_df = movies_df.copy()
        self._global_mean = float(np.mean(self._X[self._X > 0]))
        self._valid_user_ids = set(self._user_ids)
        self._valid_movie_ids = set(self._movie_ids)
        self._user_to_idx = {uid: i for i, uid in enumerate(self._user_ids)}
        self._movie_to_idx = {mid: i for i, mid in enumerate(self._movie_ids)}

        self._X_scaled = self._scaler.fit_transform(self._X)
        self._meanshift.fit(self._X_scaled)
        self._user_to_cluster = dict(zip(self._user_ids, self._meanshift.labels_))
        self._n_clusters_found_ = len(np.unique(self._meanshift.labels_))
        # print(f"MeanShift found {self._n_clusters_found_} taste clusters.")
        self._nn.fit(self._X_scaled)
        self._compute_biases(ratings_df, lambda_reg=15)
        """Content-Based Setup"""
        self._all_movie_ids = movies_df['MovieID'].tolist()
        self._movie_to_all_idx = {mid: i for i, mid in enumerate(self._all_movie_ids)}
        
        genre_texts = movies_df['Genres'].str.replace('|', ' ', regex=False).fillna('Unknown')
        # Convert to dense: genre space is ~20 dims, so this is tiny
        self._movie_genre_matrix = self._genre_vectorizer.fit_transform(genre_texts).toarray()
        
        self._user_genre_profiles = self._compute_user_profiles(ratings_df)

        self.is_fitted_ = True
        return self

    def _compute_user_profiles(self, ratings_df):
        """Efficiently compute weighted genre profiles for all training users."""
        valid_ratings = ratings_df[
            ratings_df['UserID'].isin(self._valid_user_ids) & 
            ratings_df['MovieID'].isin(self._movie_to_all_idx)
        ].copy()
        
        if valid_ratings.empty:
            return np.zeros((len(self._user_ids), self._movie_genre_matrix.shape[1]))
            
        valid_ratings['RowIdx'] = valid_ratings['UserID'].map(self._user_to_idx)
        valid_ratings['ColIdx'] = valid_ratings['MovieID'].map(self._movie_to_all_idx)
        valid_ratings = valid_ratings.dropna(subset=['RowIdx', 'ColIdx'])
        
        rows = valid_ratings['RowIdx'].astype(int).values
        cols = valid_ratings['ColIdx'].astype(int).values
        data = valid_ratings['Rating'].values
        
        # Build dense user-movie-rating matrix
        user_movie_dense = np.zeros((len(self._user_ids), len(self._all_movie_ids)))
        user_movie_dense[rows, cols] = data
        
        # Project into genre space
        profiles = user_movie_dense @ self._movie_genre_matrix
        
        # L2 normalize for cosine similarity
        norms = np.linalg.norm(profiles, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return profiles / norms

    def _predict_content_batch(self, user_ids, movie_ids):
        """Vectorized content-based rating prediction."""
        u_indices = np.array([self._user_to_idx.get(u, -1) for u in user_ids])
        m_indices = np.array([self._movie_to_all_idx.get(m, -1) for m in movie_ids])
        
        valid_mask = (u_indices >= 0) & (m_indices >= 0)
        content_preds = np.full(len(user_ids), self._global_mean)
        
        if np.any(valid_mask):
            profiles = self._user_genre_profiles[u_indices[valid_mask]]
            movie_vecs = self._movie_genre_matrix[m_indices[valid_mask]]
            
            #L2-normalized vectors: cosine similarity = element-wise dot product
            sims = np.sum(profiles * movie_vecs, axis=1)
            
            # Map similarity [-1, 1] to Rating [1, 5]
            # (TF-IDF norms ensure sims in [0, 1] for non-negative features)
            content_preds[valid_mask] = np.clip(1 + 4 * sims, 1, 5)
            
        return content_preds

    def predict_batch(self, user_ids, movie_ids, alpha=None):
        """Hybrid rating prediction: CF + Content (bug-free)"""
        if not getattr(self, 'is_fitted_', False):
            raise RuntimeError("Estimator not fitted. Call .fit() first.")
            
        user_ids_arr = np.array([int(u) for u in user_ids], dtype=int)
        movie_ids_arr = np.array([int(m) for m in movie_ids], dtype=int)
        
        # Initialize with global mean
        cf_preds = np.full(len(user_ids_arr), self._global_mean)
        content_preds = np.full(len(user_ids_arr), self._global_mean)
        
        # 1. Identify valid (user, movie) pairs
        valid_mask = np.array([(u in self._valid_user_ids) and (m in self._valid_movie_ids) 
                               for u, m in zip(user_ids_arr, movie_ids_arr)])
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return cf_preds
            
        v_users = user_ids_arr[valid_indices]
        v_movies = movie_ids_arr[valid_indices]
        
        # 2. CF: Find neighbors & compute weighted ratings
        unique_users, inv_idx = np.unique(v_users, return_inverse=True)
        unique_full_idx = [self._user_to_idx[u] for u in unique_users]
        X_scaled_batch = self._scaler.transform(self._X[unique_full_idx])
        dists, neighs = self._nn.kneighbors(X_scaled_batch)
        sims = 1 - dists
        
        # Precompute safe item means
        item_cnt = np.sum(self._X > 0, axis=0)
        item_means = np.where(item_cnt > 0, np.sum(self._X, axis=0) / item_cnt, self._global_mean)
        
        # Allocate CF predictions only for valid pairs
        cf_preds_valid = np.full(len(valid_indices), self._global_mean)
        
        for u_idx, u_id in enumerate(unique_users):
            pos = np.where(inv_idx == u_idx)[0]
            m_idx = np.array([self._movie_to_idx[mid] for mid in v_movies[pos]])
            
            u_full_idx = unique_full_idx[u_idx]
            target_user_mean = self._user_means[u_full_idx]
            
            neigh_ratings = self._X[neighs[u_idx]][:, m_idx]
            neighbor_means = self._user_means[neighs[u_idx]][:, None]  # Shape: (k, 1)
            
            # ✅ CENTERED CF: Subtract neighbor baselines before weighting
            centered_ratings = neigh_ratings - neighbor_means
            valid_mask_ratings = neigh_ratings > 0
            w = sims[u_idx][:, None] * valid_mask_ratings
            
            num = np.sum(w * centered_ratings, axis=0)
            den = np.sum(w, axis=0)
            
            # Predict = target_user_mean + weighted_adjustment
            fallback = np.where(den > 0, target_user_mean, item_means[m_idx])
            cf_preds_valid[pos] = np.where(den > 0, target_user_mean + num / np.maximum(den, 1e-9), fallback)            
                # ✅ CORRECT ASSIGNMENT: No chained indexing copy bug
        cf_preds[valid_indices] = cf_preds_valid
        
        # 3. Content prediction (for valid pairs)
        content_preds_valid = np.full(len(valid_indices), self._global_mean)
        u_content_idx = np.array([self._user_to_idx.get(u, -1) for u in v_users])
        m_content_idx = np.array([self._movie_to_all_idx.get(m, -1) for m in v_movies])
        
        c_valid = (u_content_idx >= 0) & (m_content_idx >= 0)
        if np.any(c_valid):
            profiles = self._user_genre_profiles[u_content_idx[c_valid]]
            movie_vecs = self._movie_genre_matrix[m_content_idx[c_valid]]
            sims_c = np.sum(profiles * movie_vecs, axis=1)
            # Center content around global mean to avoid dragging CF down
            content_preds_valid[c_valid] = self._global_mean + (sims_c - sims_c.mean()) * 1.5
            
        content_preds[valid_indices] = content_preds_valid
        
        # 4. Sanitize & Blend
        cf_preds = np.nan_to_num(cf_preds, nan=self._global_mean, posinf=5.0, neginf=1.0)
        content_preds = np.nan_to_num(content_preds, nan=self._global_mean, posinf=5.0, neginf=1.0)
        
        alpha = alpha
        final_preds = alpha * cf_preds + (1 - alpha) * content_preds
        return np.clip(final_preds, 1, 5)

    def predict(self, user_id, movie_id, alpha=alpha):
        """Predict rating for a user-movie pair. 
        Returns float for scalar inputs, numpy array for list/array inputs."""
        preds = self.predict_batch(np.atleast_1d(user_id), np.atleast_1d(movie_id), alpha=alpha)
        # Return scalar if single input, else return full array
        return preds[0] if np.ndim(user_id) == 0 else preds

    def recommend(self, user_id, n_rec=5, alpha=alpha):
        """Hybrid top-N recommendations."""
        if not getattr(self, 'is_fitted_', False):
            raise RuntimeError("Estimator not fitted. Call .fit() first.")
        if user_id not in self._user_to_cluster:
            return ["User not in training set or too few ratings"]
            
        u_idx = self._user_ids.index(user_id)
        target_rated = np.where(self._X[u_idx] > 0)[0]
        unseen_mask = np.ones(len(self._movie_ids), dtype=bool)
        unseen_mask[target_rated] = False
        
        if unseen_mask.sum() == 0:
            return ["User has rated all available movies"]
            
        unseen_movie_ids = np.array(self._movie_ids)[unseen_mask]
        unseen_movie_indices = np.where(unseen_mask)[0]
        
        # Predict hybrid scores for all unseen movies at once
        user_batch = np.full(len(unseen_movie_ids), user_id)
        scores = self.predict_batch(user_batch, unseen_movie_ids, alpha=alpha)
        
        top_indices = np.argsort(scores)[::-1][:n_rec]
        top_mid = unseen_movie_ids[top_indices]
        
        return self._movies_df[self._movies_df['MovieID'].isin(top_mid)][['MovieID', 'Title']].values.tolist()
    def _compute_biases(self, ratings_df, lambda_reg=15):
        """Compute user and item biases via regularized alternating least squares."""
        # Initialize
        self._user_bias = np.zeros(len(self._user_ids))
        self._item_bias = np.zeros(len(self._movie_ids))
        
        # Build lookup for fast access
        user_to_row = {uid: i for i, uid in enumerate(self._user_ids)}
        movie_to_col = {mid: j for j, mid in enumerate(self._movie_ids)}
        
        # Filter to active users/movies only
        active_ratings = ratings_df[
            (ratings_df['UserID'].isin(self._user_ids)) & 
            (ratings_df['MovieID'].isin(self._movie_ids))
        ].copy()
        
        if active_ratings.empty:
            return
            
        # Convert to 0-based indices
        active_ratings['u_idx'] = active_ratings['UserID'].map(user_to_row)
        active_ratings['i_idx'] = active_ratings['MovieID'].map(movie_to_col)
        active_ratings = active_ratings.dropna(subset=['u_idx', 'i_idx'])
        
        # Alternating Least Squares (5 iterations is usually enough)
        for _ in range(5):
            # Update user biases
            for u_idx in range(len(self._user_ids)):
                user_data = active_ratings[active_ratings['u_idx'] == u_idx]
                if len(user_data) == 0:
                    continue
                residuals = user_data['Rating'].values - self._global_mean - self._item_bias[user_data['i_idx'].values]
                self._user_bias[u_idx] = residuals.sum() / (lambda_reg + len(user_data))
            
            # Update item biases
            for i_idx in range(len(self._movie_ids)):
                item_data = active_ratings[active_ratings['i_idx'] == i_idx]
                if len(item_data) == 0:
                    continue
                residuals = item_data['Rating'].values - self._global_mean - self._user_bias[item_data['u_idx'].values]
                self._item_bias[i_idx] = residuals.sum() / (lambda_reg + len(item_data))
    
    def predict_bias(self, user_id, movie_id):
        """Bias baseline prediction: μ + b_u + b_i"""
        if not getattr(self, 'is_fitted_', False):
            return self._global_mean
            
        # Cold-start fallback
        if user_id not in self._user_to_idx or movie_id not in self._movie_to_idx:
            return self._global_mean
            
        u_idx = self._user_to_idx[user_id]
        m_idx = self._movie_to_idx[movie_id]
        
        pred = self._global_mean + self._user_bias[u_idx] + self._item_bias[m_idx]
        return np.clip(pred, 1, 5)  # Ensure rating stays in valid range

def cross_validate_recommender(model_class, ratings_df, movies_df, n_splits=5, alpha=alpha, **model_params):
    """Run K-Fold CV on explicit ratings. Returns list of fold RMSEs."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(ratings_df)):
        train_df = ratings_df.iloc[train_idx]
        test_df = ratings_df.iloc[test_idx]
        
        try:
            # Instantiate & fit fresh model per fold
            model = model_class(**model_params)
            model.fit(train_df, movies_df)
            
            # Filter test to only users/movies seen during training
            valid_test = test_df[
                (test_df['UserID'].isin(model._user_ids)) & 
                (test_df['MovieID'].isin(model._movie_ids))
            ]
            
            if len(valid_test) < 20:
                print(f"⚠️ Fold {fold_idx+1}: Skipped (too few valid test pairs)")
                continue
                
            # Fast vectorized prediction
            preds = model.predict_batch(valid_test['UserID'].values, valid_test['MovieID'].values, alpha=alpha)
            rmse = np.sqrt(mean_squared_error(valid_test['Rating'].values, preds))
            fold_scores.append(rmse)
            # print(f"Fold {fold_idx+1}: RMSE = {rmse:.3f} ({len(valid_test)} pairs)")
            
        except Exception as e:
            print(f"Fold {fold_idx+1} failed: {e}")
            
    if not fold_scores:
        raise RuntimeError("No folds produced valid results. Check min_ratings threshold or data sparsity.")
        
    mean_rmse = np.mean(fold_scores)
    std_rmse = np.std(fold_scores)
    print("Cross-Validation Complete:")
    print(f"   Mean RMSE: {mean_rmse:.3f} ± {std_rmse:.3f}")
    return fold_scores

def load_data(dataset_folder):
    user_path = os.path.join(dataset_folder, 'users.dat')
    movie_path = os.path.join(dataset_folder, 'movies.dat')
    rating_path = os.path.join(dataset_folder, 'ratings.dat')
    users = pd.read_csv(user_path, sep='::', engine='python', header=None,
                        names=['UserID','Gender','Age','Occupation','Zip'])
    movies = pd.read_csv(movie_path, sep='::', engine='python', header=None, encoding='latin-1',
                         names=['MovieID','Title','Genres'])
    ratings = pd.read_csv(rating_path, sep='::', engine='python', header=None,
                          names=['UserID','MovieID','Rating','Timestamp'])

    return users, movies, ratings

def wcss(X, max_k):
    wcss = []
    for i in range(1, max_k+1):
        kmeans = KMeans(n_clusters=i, init='k-means++')
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return wcss

def kmeans_elbow(X, scaler):
    wcss_vals = []
    max_k=30
    #because kmeans is semi-stochastic, taking averages is useful in finding the elbow
    for i in range(1, 15):
        print(f"WCSS pass {i} of 15")
        wcss_vals.append(wcss(X, max_k))
    avg_wcss = np.mean(wcss_vals,axis=0)
    plt.plot(range(1,max_k+1), avg_wcss)
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.show()

def evaluate_with_coverage(ratings_df, movies_df, min_r, alpha=0.7):
    """Evaluate RMSE + report coverage metrics."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import numpy as np
    
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
    model = HybridUserClusterKNNRecommender(n_neighbors=10, min_ratings=min_r, alpha=alpha)
    model.fit(train_df, movies_df)
    
    # Filter test to seen users/movies
    valid_test = test_df[
        (test_df['UserID'].isin(model._user_ids)) & 
        (test_df['MovieID'].isin(model._movie_ids))
    ]
    
    # Compute metrics
    preds = model.predict_batch(valid_test['UserID'].values, valid_test['MovieID'].values)
    rmse = np.sqrt(mean_squared_error(valid_test['Rating'], preds))
    
    # Coverage metrics
    user_coverage = len(model._user_ids) / ratings_df['UserID'].nunique() * 100
    movie_coverage = len(model._movie_ids) / movies_df['MovieID'].nunique() * 100
    interaction_coverage = len(valid_test) / len(test_df) * 100
    
    return {
        'min_ratings': min_r,
        'rmse': rmse,
        'user_coverage_pct': user_coverage,
        'movie_coverage_pct': movie_coverage,
        'interaction_coverage_pct': interaction_coverage,
        'n_users': len(model._user_ids),
        'n_movies': len(model._movie_ids)
    }

# this lets us get evaluation metrics easily
def run_mr_sweep(ratings, movies):
    results = []
    for mr in [5, 10, 20, 50, 100, 200, 500]:
        metrics = evaluate_with_coverage(ratings, movies, min_r=mr, alpha=0.7)
        results.append(metrics)
        print(f"min_ratings={mr:3d} | RMSE={metrics['rmse']:.3f} | "
              f"Users:{metrics['user_coverage_pct']:5.1f}% | "
              f"Movies:{metrics['movie_coverage_pct']:5.1f}% | "
              f"Interactions:{metrics['interaction_coverage_pct']:5.1f}%")

#this helps to tune in the default alpha value
def run_alpha_sweep(train_ratings, test_ratings, movies):
    model = HybridUserClusterKNNRecommender(n_neighbors=num_neighbors, min_ratings=min_ratings)
    model.fit(train_ratings, movies)
    
    #ensure valid_test is properly aligned
    valid_test = test_ratings[
        (test_ratings['UserID'].isin(model._user_ids)) & 
        (test_ratings['MovieID'].isin(model._movie_ids))
    ]
    
    # test different levels of content/CF blending
    alphas = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
    alpha_scores = []
    for a in alphas:
        preds = model.predict_batch(valid_test['UserID'].values, valid_test['MovieID'].values, alpha=a)
        rmse = np.sqrt(mean_squared_error(valid_test['Rating'], preds))
        nans = np.isnan(preds).sum()
        print(f"α={a:.2f} | RMSE={rmse:.3f} | NaNs={nans}")
        alpha_scores.append(rmse)
    plt.plot(alphas, alpha_scores)
    return alpha_scores

def run_alpha_sweep_cross(ratings,movies):
    alphas = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
    alpha_scores = []
    alpha_stds = []
    for a in alphas:
        score = cross_validate_recommender(HybridUserClusterKNNRecommender, ratings, movies,5, alpha=a, n_neighbors=num_neighbors, min_ratings=min_ratings)
        mean_rmse = np.mean(score)
        rmse_std = np.std(score)
        print(f"α={a:.2f} | RMSE={mean_rmse:.3f} +/- {rmse_std:.3f}")
        alpha_scores.append(mean_rmse)
        alpha_stds.append(rmse_std)
    plt.plot(alphas, alpha_scores)
    plt.xlabel("α values")
    plt.ylabel("RMSE")
    plt.title("α-blending vs. RMSE")
    return alpha_scores, alpha_stds
        

    
def main():
    users, movies, ratings = load_data(dataset_folder)
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)
    
    #commented out block of code for finding the best min_Ratings number
    # report = generate_coverage_report(
    # ratings_df=ratings,
    # movies_df=movies,
    # alpha=alpha,
    # output_plot=True,  # Saves PNG/PDF automatically
    # verbose=True
    # )
    
    # # Access results
    # print(f"Best RMSE: {report['results_df']['rmse'].min():.3f}")
    # if 'sweet_spot' in report:
    #     print(f"Sweet spot at min_ratings={report['sweet_spot']['min_ratings']}")
    
    #generate grid search results
    # report = generate_grid_search_report(
    #     model_class=HybridUserClusterKNNRecommender,
    #     ratings_df=ratings,
    #     movies_df=movies,
    #     min_ratings_values=[30, 40, 50],
    #     n_neighbors_values=[200, 250, 300, 350, 400, 450, 500],
    #     alpha=0.75,
    #     n_splits=5,  # Use 5 for final report
    #     verbose=True
    # )
    
    # run_alpha_sweep_cross(ratings, movies)
    # cross_validate_recommender(HybridUserClusterKNNRecommender, ratings, movies, n_neighbors=num_neighbors, min_ratings=min_ratings)
    
    #baseline comparisons 
    report = generate_baseline_report(
        ratings_df=ratings,
        movies_df=movies,
        hybrid_rmse=0.897,        # Your model's RMSE
        hybrid_coverage=91.9,     # Your model's coverage
        min_ratings_values=[40],
        verbose=True
    )
    
    # Access results programmatically
    print(report['summary'])
    print(f"\nBest baseline RMSE: {report['results_df']['mean_rmse'].min():.3f}")

    
if __name__ == '__main__':
    main()