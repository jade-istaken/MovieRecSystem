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
max_evals = 1000

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
        """Train the recommender on ratings and movie metadata."""
        # 1. Build user-movie matrix
        user_movie = ratings_df.pivot_table(index='UserID', columns='MovieID', values='Rating')
        
        # 2. Filter active users/movies to reduce noise
        active_users = user_movie.count(axis=1) >= self.min_ratings
        active_movies = user_movie.count(axis=0) >= self.min_ratings
        df = user_movie.loc[active_users, active_movies].fillna(0)
        
        if df.empty:
            raise ValueError("No users/movies meet the min_ratings threshold.")
            
        # 3. Store mappings & convert to numpy (bypasses pandas feature validation)
        self._user_ids = df.index.tolist()
        self._movie_ids = df.columns.tolist()
        self._X = df.values
        self._movies_df = movies_df.copy()
        
        # 4. Scale & Cluster users by taste
        self._X_scaled = self._scaler.fit_transform(self._X)
        cluster_labels = self._kmeans.fit_predict(self._X_scaled)
        self._user_to_cluster = dict(zip(self._user_ids, cluster_labels))
        
        # 5. Fit Nearest Neighbors on scaled data
        self._nn.fit(self._X_scaled)
        
        # Precompute global mean for cold-start fallbacks
        self._global_mean = np.mean(self._X[self._X > 0])
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
    train_users, test_users = train_test_split(ratings['UserID'].unique(), test_size=0.2, random_state=42)
    train_ratings = ratings[ratings['UserID'].isin(train_users)]
    
    model = UserClusterKNNRecommender(n_clusters=num_clusters, n_neighbors=num_neighbors, min_ratings=min_ratings)
    model.fit(train_ratings, movies)
    
    sample_user = train_users[0]
    sample_movie = model._movie_ids[0]
    print(f"Predicted rating for User {sample_user} on Movie {sample_movie}: {model.predict(sample_user, sample_movie):.2f}")
    print(f"\nTop 5 recommendations for User {sample_user}:")
    for mid, title in model.recommend(sample_user, n_rec=5):
        print(f"- {title} (ID: {mid})")

    
    test_ratings_subset = ratings[ratings['UserID'].isin(test_users)].head(500)  # sample for speed
    preds = [model.predict(row.UserID, row.MovieID) for _, row in test_ratings_subset.iterrows()]
    actuals = test_ratings_subset['Rating'].values
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    print(f"\nSample RMSE: {rmse:.3f}")

if __name__ == '__main__':
    main()