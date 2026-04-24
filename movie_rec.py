#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random

import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClusterMixin

import matplotlib.pyplot as plt

#global behavioral variables go here 
dataset_folder = 'dataset'
min_ratings = 20 #threshold for minimum ratings
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
        self._nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')

    def fit(self, ratings_df, movies_df, y=None):
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
        self._user_to_cluster = dict(zip(self._user_ids, self._kmeans.fit_predict(self._X_scaled)))
        self._nn.fit(self._X_scaled)
        
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

    
    model = UserClusterKNNRecommender(n_clusters=num_clusters, n_neighbors=num_neighbors, min_ratings=min_ratings)
    model.fit(train_ratings, movies)
    
    sample_user = 2486
    sample_movie = model._movie_ids[0]
    print(f"Predicted rating for User {sample_user} on Movie {sample_movie}: {model.predict(sample_user, sample_movie):.2f}")
    print(f"\nTop 5 recommendations for User {sample_user}:")
    for mid, title in model.recommend(sample_user, n_rec=5):
        print(f"- {title} (ID: {mid})")
        
    valid_test = test_ratings[
    (test_ratings['UserID'].isin(model._user_ids)) & 
    (test_ratings['MovieID'].isin(model._movie_ids))
    ]
    

    preds = model.predict_batch(valid_test['UserID'].values, valid_test['MovieID'].values)
    rmse = np.sqrt(mean_squared_error(valid_test['Rating'].values, preds))
    print(f"Vectorized RMSE: {rmse:.3f}")

if __name__ == '__main__':
    main()