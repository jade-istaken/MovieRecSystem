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

import matplotlib.pyplot as plt

#global behavioral variables go here 
dataset_folder = 'dataset'
min_ratings = 20 #threshold for minimum ratings
num_neighbors = 10
num_clusters = 15
max_evals = 1000


def fit_scaler(x):
    scaler = StandardScaler()
    scaler = scaler.fit(x)
    return scaler

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

def basic_ratings_matrix(ratings):
    #this creates a matrix of users and movies, with ratings being the values inside them
    user_movie_matrix = ratings.pivot_table(index='UserID', columns='MovieID', values='Rating')

    #then prune the matrix to only movies with more than x reviews and users with more than x review
    active_users = user_movie_matrix.count(axis = 1) >= min_ratings
    active_movies = user_movie_matrix.count(axis = 0) >= min_ratings
    df = user_movie_matrix.loc[active_users, active_movies].fillna(0)
    user_ids = df.index.tolist()
    movie_ids = df.columns.tolist()
    return df, user_ids, movie_ids

def kmeans_cluster(X, n_clusters, scaler, user_ids):
    X_scaled = scaler.transform(X) # scale the data because ratings aren't normalized
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++')
    cluster_labels = kmeans.fit_predict(X_scaled)
    return dict(zip(user_ids, cluster_labels)) #this basically just returns a representation of user ratings

#kind of just a baseline nearest-neighbor predictor
def nearest_neighbor_cluster(X, n_neighbors, scaler):
    X_scaled = scaler.transform(X)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    return nn.fit(X_scaled)

def recommend_movies(user_id, user_to_cluster, user_ids, scaler, X, train_user_ids, X_train_scaled, movie_ids, movies, n_rec=5):
    if user_id not in user_to_cluster:
        return ["User not in training set or too few ratings"]

    full_idx = user_ids.index(user_id)
    user_vec_scaled = scaler.transform(X[full_idx].reshape(1, -1))
    cluster_id = user_to_cluster[user_id]

    # Filter to same cluster
    cluster_mask = np.array([user_to_cluster.get(uid) == cluster_id for uid in train_user_ids])
    if cluster_mask.sum() < 5:
        cluster_mask = np.ones(len(train_user_ids), dtype=bool)

    X_cluster = X_train_scaled[cluster_mask]
    nn_cluster = NearestNeighbors(n_neighbors=min(10, X_cluster.shape[0]), metric='cosine', algorithm='brute')
    nn_cluster.fit(X_cluster)

    dists, idxs = nn_cluster.kneighbors(user_vec_scaled)
    similarities = 1 - dists[0]

    rec_scores = {}
    cluster_train_user_ids = [uid for uid, mask in zip(train_user_ids, cluster_mask) if mask]

    for i, sim in zip(idxs[0], similarities):
        su_id = cluster_train_user_ids[i]
        su_full_idx = user_ids.index(su_id)
        su_vec = X[su_full_idx]

        target_rated = np.where(X[full_idx] > 0)[0]
        similar_rated = np.where(su_vec > 0)[0]
        unseen_indices = np.setdiff1d(similar_rated, target_rated)

        for midx in unseen_indices:
            mid = movie_ids[midx]
            rec_scores[mid] = rec_scores.get(mid, 0) + (su_vec[midx] * sim)

    top_movies = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)[:n_rec]
    top_ids = [m[0] for m in top_movies]
    return movies[movies['MovieID'].isin(top_ids)][['MovieID', 'Title']].values.tolist()

def predict_rating(user_id, movie_id, X_train, movie_ids, user_ids, scaler, X, nn):
    if movie_id not in movie_ids:
        return np.mean(X_train)
    try:
        full_user_idx = user_ids.index(user_id)
    except ValueError:
        return np.mean(X_train) #if the user or movie aren't in the data then just. get the average rating across all movies and users
    full_movie_idx = movie_ids.index(movie_id)
    user_vec_scaled = scaler.transform(X[full_user_idx].reshape(1, -1))

    dists, idxs = nn.kneighbors(user_vec_scaled)
    similarities = 1 - dists[0]

    #weighted average of neighbor's ratings
    neighbor_ratings = X_train[idxs[0], full_movie_idx]
    valid = neighbor_ratings > 0
    if valid.sum() == 0:
        return np.mean(X_train[:, full_movie_idx])
    return np.average(neighbor_ratings[valid], weights=similarities[valid])



def ratings_engine(users, movies, ratings):
    #first make the matrix
    user_movie_matrix, user_ids, movie_ids = basic_ratings_matrix(ratings)

    #strip only the values out
    X = user_movie_matrix.values

    #have to do a test/train split (speficy a random state so that it's reproducible
    #during development)
    train_idx, test_idx = train_test_split(np.arange(len(user_ids)), test_size=0.2, random_state=42)
    X_train, X_test = X[train_idx], X[test_idx]
    train_user_ids = [user_ids[i] for i in train_idx]
    test_user_ids = [user_ids[i] for i in test_idx]


    #make a scaler for everything
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    user_cluster_dict = kmeans_cluster(X_train, n_clusters=num_clusters, scaler=scaler, user_ids=train_user_ids)
    nn_cluster = nearest_neighbor_cluster(X_train, n_neighbors=num_neighbors, scaler=scaler)

    test_actuals = []
    test_preds = []

    # Predict test ratings for evaluation
    for i in range(X_test.shape[0]):
        u_id = test_user_ids[i]
        rated_movie_cols = np.where(X_test[i] > 0)[0]
        for j in rated_movie_cols:
            m_id = movie_ids[j]
            actual = X_test[i, j]
            pred = predict_rating(u_id, m_id, X_train, movie_ids, user_ids, scaler, X, nn_cluster)
            test_actuals.append(actual)
            test_preds.append(pred)
            if len(test_actuals) >= max_evals:
                break
        if len(test_actuals) >= max_evals:
            break

    if len(test_actuals) > 0:
        rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
        print(f"Sample RMSE: {rmse:.3f}")

    target_user = train_user_ids[random.randint(0, len(train_user_ids) - 1)]
    print(f"\nTop 5 recommendations for User {target_user}:")
    for mid, title in recommend_movies(target_user, n_rec=5, user_to_cluster=user_cluster_dict, user_ids=user_ids, scaler=scaler, X=X, train_user_ids=train_user_ids, X_train_scaled=X_train_scaled,movies=movies, movie_ids=movie_ids):
        print(f"- {title} (ID: {mid})")

def main():
    users, movies, ratings = load_data(dataset_folder)
    ratings_engine(users, movies, ratings)

    

if __name__ == '__main__':
    main()