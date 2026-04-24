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

#global behavioral variables go here 
dataset_folder = 'dataset'
min_ratings = 5 #threshold for minimum ratings
num_neighbors = 13
num_clusters = 15

class UserClusterKNNRecommender(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=15, n_neighbors=10, min_ratings=20, random_state=42):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.min_ratings = min_ratings
        self.random_state = random_state

        # Internal sklearn estimators
        self._scaler = StandardScaler()
        self._kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        # add meanshift so we don't have to tune the cluster amount
        self._meanshift = MeanShift(bin_seeding=True, cluster_all=True, bandwidth=25)
        self._nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')

    def fit(self, ratings_df, movies_df, y=None):
        """Fit the model on rating and movie data"""
        # Build & filter matrix
        user_movie = ratings_df.pivot_table(index='UserID', columns='MovieID', values='Rating')
        active_users = user_movie.count(axis=1) >= self.min_ratings
        active_movies = user_movie.count(axis=0) >= self.min_ratings
        df = user_movie.loc[active_users, active_movies].fillna(0)
        if df.empty:
            raise ValueError("No users/movies meet the min_ratings threshold.")
            
        # Precompute Values
        self._user_ids = [int(uid) for uid in df.index]
        self._movie_ids = [int(mid) for mid in df.columns]
        self._X = df.values
        self._movies_df = movies_df.copy()
        self._global_mean = float(np.mean(self._X[self._X > 0]))
        
        # Precompute lookup structures
        self._valid_user_ids = set(self._user_ids)
        self._valid_movie_ids = set(self._movie_ids)
        self._user_to_idx = {uid: i for i, uid in enumerate(self._user_ids)}
        self._movie_to_idx = {mid: i for i, mid in enumerate(self._movie_ids)}
        
        # Scale, Cluster, & Fit NN
        self._X_scaled = self._scaler.fit_transform(self._X)
        self._meanshift.fit(self._X_scaled)
        cluster_labels = self._meanshift.labels_
        self._user_to_cluster = dict(zip(self._user_ids, cluster_labels))
        self._n_clusters_found_ = len(np.unique(cluster_labels))
        print(f"found {self._n_clusters_found_} user taste clusters")
        self._nn.fit(self._X_scaled)
        #compute the biases just as a benchmark
        self._compute_biases(ratings_df, lambda_reg=15)
        
        self.is_fitted_ = True
        return self

    def predict(self, user_id, movie_id):
        """Predict rating for a user-movie pair."""
        if not getattr(self, 'is_fitted_', False):
            raise RuntimeError("Estimator not fitted. Call .fit() first.")
            
        # Cold-start fallback
        if user_id not in self._user_to_cluster or movie_id not in self._movie_ids:
            return self._global_mean
            
        full_user_idx = self._user_ids.index(user_id)
        full_movie_idx = self._movie_ids.index(movie_id)
        
        user_vec_scaled = self._scaler.transform(self._X[full_user_idx].reshape(1, -1))
        dists, idxs = self._nn.kneighbors(user_vec_scaled)
        similarities = 1 - dists[0]
        
        neighbor_ratings = self._X[idxs[0], full_movie_idx]
        valid = neighbor_ratings > 0
        
        if valid.sum() == 0:
            # Fallback to movie mean or global mean
            movie_ratings = self._X[:, full_movie_idx]
            valid_movie = movie_ratings > 0
            return np.mean(movie_ratings[valid_movie]) if valid_movie.sum() > 0 else self._global_mean
            
        return np.average(neighbor_ratings[valid], weights=similarities[valid])
    
    def predict_batch(self, user_ids, movie_ids):
        """Vectorized rating prediction for multiple user-movie pairs."""
        if not getattr(self, 'is_fitted_', False):
            raise RuntimeError("Estimator not fitted. Call .fit() first.")
            
        # Force inputs to standard Python ints to match dict keys
        user_ids = np.array([int(u) for u in user_ids], dtype=int)
        movie_ids = np.array([int(m) for m in movie_ids], dtype=int)
        predictions = np.full(len(user_ids), self._global_mean)
        
        # Vectorized validation using boolean indexing
        valid_mask = np.array([
            (u in self._valid_user_ids) and (m in self._valid_movie_ids)
            for u, m in zip(user_ids, movie_ids)
        ])
        
        if not np.any(valid_mask):
            return predictions
            
        v_users = user_ids[valid_mask]
        v_movies = movie_ids[valid_mask]
        
        # Precompute item means for fallback
        item_ratings_sum = np.sum(self._X, axis=0)
        item_ratings_count = np.sum(self._X > 0, axis=0)
        item_means = np.where(item_ratings_count > 0, 
                              item_ratings_sum / item_ratings_count, 
                              self._global_mean)
        
        # Group by unique users to minimize NN calls
        unique_users, inverse_indices = np.unique(v_users, return_inverse=True)
        predictions_valid = np.empty(len(v_users))
        
        # Batch scale & find neighbors
        unique_full_indices = [self._user_to_idx[uid] for uid in unique_users]
        X_unique_scaled = self._scaler.transform(self._X[unique_full_indices])
        dists_all, neighbors_all = self._nn.kneighbors(X_unique_scaled)
        similarities_all = 1 - dists_all
        
        # Compute predictions per unique user
        for u_idx, u_id in enumerate(unique_users):
            sims = similarities_all[u_idx]
            neighbor_indices = neighbors_all[u_idx]
            
            user_positions = np.where(inverse_indices == u_idx)[0]
            target_movie_ids = v_movies[user_positions]
            
            target_movie_indices = np.array([self._movie_to_idx[mid] for mid in target_movie_ids])
            
            neighbor_ratings = self._X[neighbor_indices][:, target_movie_indices]
            valid_mask_movies = neighbor_ratings > 0
            
            weights = sims[:, np.newaxis] * valid_mask_movies
            numerators = np.sum(weights * neighbor_ratings, axis=0)
            denominators = np.sum(weights, axis=0)
            
            preds = np.where(denominators > 0, 
                             numerators / denominators, 
                             item_means[target_movie_indices])
            
            predictions_valid[user_positions] = preds
            
        predictions[valid_mask] = predictions_valid
        return predictions
        
        
    def recommend(self, user_id, n_rec=5):
        """Return top-N recommended movies for a user."""
        if not getattr(self, 'is_fitted_', False):
            raise RuntimeError("Estimator not fitted. Call .fit() first.")
            
        if user_id not in self._user_to_cluster:
            return ["User not in training set or too few ratings"]
            
        full_user_idx = self._user_ids.index(user_id)
        user_vec_scaled = self._scaler.transform(self._X[full_user_idx].reshape(1, -1))
        cluster_id = self._user_to_cluster[user_id]
        
        # Filter to users in the same cluster (reduces noise)
        cluster_mask = np.array([self._user_to_cluster.get(uid) == cluster_id for uid in self._user_ids])
        if cluster_mask.sum() < 5:
            cluster_mask = np.ones(len(self._user_ids), dtype=bool)  # Fallback to all users
            
        X_cluster = self._X_scaled[cluster_mask]
        nn_cluster = NearestNeighbors(n_neighbors=min(self.n_neighbors, X_cluster.shape[0]), 
                                      metric='cosine', algorithm='brute')
        nn_cluster.fit(X_cluster)
        
        dists, idxs = nn_cluster.kneighbors(user_vec_scaled)
        similarities = 1 - dists[0]
        
        rec_scores = {}
        cluster_train_user_ids = [uid for uid, mask in zip(self._user_ids, cluster_mask) if mask]
        
        for i, sim in zip(idxs[0], similarities):
            su_id = cluster_train_user_ids[i]
            su_full_idx = self._user_ids.index(su_id)
            su_vec = self._X[su_full_idx]
            
            target_rated = np.where(self._X[full_user_idx] > 0)[0]
            similar_rated = np.where(su_vec > 0)[0]
            unseen_indices = np.setdiff1d(similar_rated, target_rated)
            
            for midx in unseen_indices:
                mid = self._movie_ids[midx]
                rec_scores[mid] = rec_scores.get(mid, 0) + (su_vec[midx] * sim)
                
        top_movies = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)[:n_rec]
        top_ids = [m[0] for m in top_movies]
        return self._movies_df[self._movies_df['MovieID'].isin(top_ids)][['MovieID', 'Title']].values.tolist()

class HybridUserClusterKNNRecommender(BaseEstimator, ClusterMixin):
    def __init__(self, n_neighbors=10, min_ratings=20, alpha=0.7):
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
        print(f"MeanShift found {self._n_clusters_found_} taste clusters.")
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
        """Hybrid prediction: blends CF + Content."""
        if not getattr(self, 'is_fitted_', False):
            raise RuntimeError("Estimator not fitted. Call .fit() first.")
            
        cf_preds = np.full(len(user_ids), self._global_mean)
        
        # CF prediction (reuse existing logic)
        user_ids_arr = np.array([int(u) for u in user_ids], dtype=int)
        movie_ids_arr = np.array([int(m) for m in movie_ids], dtype=int)
        valid_mask = np.array([(u in self._valid_user_ids) and (m in self._valid_movie_ids) 
                               for u, m in zip(user_ids_arr, movie_ids_arr)])
        
        if np.any(valid_mask):
            v_users = user_ids_arr[valid_mask]
            v_movies = movie_ids_arr[valid_mask]
            
            unique_users, inv_idx = np.unique(v_users, return_inverse=True)
            unique_full_idx = [self._user_to_idx[u] for u in unique_users]
            X_scaled_batch = self._scaler.transform(self._X[unique_full_idx])
            dists, neighs = self._nn.kneighbors(X_scaled_batch)
            sims = 1 - dists
            
            item_sum = np.sum(self._X, axis=0)
            item_cnt = np.sum(self._X > 0, axis=0)
            item_means = np.where(item_cnt > 0, item_sum / item_cnt, self._global_mean)
            
            preds_cf = np.full(len(v_users), self._global_mean)
            for u_idx, u_id in enumerate(unique_users):
                pos = np.where(inv_idx == u_idx)[0]
                m_idx = np.array([self._movie_to_idx[mid] for mid in v_movies[pos]])
                neigh_ratings = self._X[neighs[u_idx]][:, m_idx]
                valid = neigh_ratings > 0
                w = sims[u_idx][:, None] * valid
                num = np.sum(w * neigh_ratings, axis=0)
                den = np.sum(w, axis=0)
                preds_cf[pos] = np.where(den > 0, num / den, item_means[m_idx])
                
            cf_preds[valid_mask] = preds_cf

        # Content prediction
        content_preds = self._predict_content_batch(user_ids_arr, movie_ids_arr)
        
        # Adaptive blending: lean toward content if CF signal is weak
        alpha = alpha or self.alpha
        # If CF prediction equals global mean (cold/uncertain), shift weight to content
        cf_is_weak = (np.abs(cf_preds - self._global_mean) < 0.1)
        final_alpha = np.where(cf_is_weak, 0.3, alpha)
        
        return final_alpha * cf_preds + (1 - final_alpha) * content_preds
    
    def predict(self, user_id, movie_id, alpha=None):
        """Predict rating for a user-movie pair. 
        Returns float for scalar inputs, numpy array for list/array inputs."""
        preds = self.predict_batch(np.atleast_1d(user_id), np.atleast_1d(movie_id), alpha=alpha)
        # Return scalar if single input, else return full array
        return preds[0] if np.ndim(user_id) == 0 else preds

    def recommend(self, user_id, n_rec=5, alpha=None):
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

def cross_validate_recommender(model_class, ratings_df, movies_df, n_splits=5, **model_params):
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
            preds = model.predict_batch(valid_test['UserID'].values, valid_test['MovieID'].values)
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


def main():
    users, movies, ratings = load_data(dataset_folder)
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

    
    model = HybridUserClusterKNNRecommender(n_neighbors=num_neighbors, min_ratings=min_ratings)
    model.fit(train_ratings, movies)
    
    sample_user = 4200
    sample_movie = model._movie_ids[0]
    print(f"Predicted rating for User {sample_user} on Movie {sample_movie}: {model.predict(sample_user, sample_movie):.2f}")
    print(f"\nTop 5 recommendations for User {sample_user}:")
    for mid, title in model.recommend(sample_user, n_rec=5):
        print(f"- {title} (ID: {mid})")    

    param_grid = {
        'n_neighbors': [19, 20, 25],
        'min_ratings': [19, 20, 25],
        'alpha': [0.9]
    }
    
    best_mean_rmse = float('inf')
    best_params = None
    results = []
    
    for n_n in param_grid['n_neighbors']:
        for min_r in param_grid['min_ratings']:
            for a in param_grid['alpha']:
                print(f"\n🔍 Evaluating: n_neighbors={n_n}, min_ratings={min_r}, alpha={a}")
                try:
                    scores = cross_validate_recommender(
                        HybridUserClusterKNNRecommender,
                        ratings, movies,
                        n_splits=5,
                        n_neighbors=n_n, min_ratings=min_r, alpha=a
                    )
                    mean_r = np.mean(scores)
                    results.append({'n_neighbors': n_n, 'min_ratings': min_r, 'alpha': a, 'mean_rmse': mean_r})
                    if mean_r < best_mean_rmse:
                        best_mean_rmse = mean_r
                        best_params = {'n_neighbors': n_n, 'min_ratings': min_r, 'alpha': a}
                except Exception as e:
                    print(f"   Skipped: {e}")
    
    print(f"\n🏆 Best CV RMSE: {best_mean_rmse:.3f} | Params: {best_params}")

if __name__ == '__main__':
    main()