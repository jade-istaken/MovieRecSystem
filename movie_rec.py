#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    user_movie_matrix = user_movie_matrix.loc[active_users, active_movies].fillna(0)
    return user_movie_matrix

def kmeans_cluster(X, n_clusters):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) # scale the data because ratings aren't normalized
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++')
    X['cluster'] = kmeans.fit_predict(X_scaled)
    return X['cluster'] #this basically just returns a representation of user ratings

def nearest_neighbor_cluster(X, n_neighbors):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    return nn.fit(X_scaled)

def ratings_engine(ratings):
    #first make the matrix
    user_movie_matrix = basic_ratings_matrix(ratings)

    #have to do a test/train split (speficy a random state so that it's reproducible
    #during development)
    train_users, test_users = train_test_split(user_movie_matrix.index, test_size=0.2, random_state=42)
    train_matrix = user_movie_matrix.loc[train_users]
    test_matrix = user_movie_matrix.loc[test_users]

    user_cluster = kmeans_cluster(train_matrix, n_clusters=num_clusters)
    nn = NearestNeighbors(n_neighbors=num_neighbors)



def main():
    users, movies, ratings = load_data(dataset_folder)
    ratings_engine(ratings)
    

    

if __name__ == '__main__':
    main()