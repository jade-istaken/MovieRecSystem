#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

#global behavioral variables go here 
dataset_folder = 'dataset'


def load_data(dataset_folder):
    user_path = os.path.join(dataset_folder, 'users.dat')
    movie_path = os.path.join(dataset_folder, 'movies.dat')
    rating_path = os.path.join(dataset_folder, 'ratings.dat')
    print(user_path)

    users = pd.read_csv(user_path, sep='::', engine='python', header=None,
                        names=['UserID','Gender','Age','Occupation','Zip'])
    movies = pd.read_csv(movie_path, sep='::', engine='python', header=None, encoding='latin-1',
                         names=['MovieID','Title','Genres'])
    ratings = pd.read_csv(rating_path, sep='::', engine='python', header=None,
                          names=['UserID','MovieID','Rating','Timestamp'])

    return users, movies, ratings

def basic_ratings_matrix(ratings):
    #this basic system just like. tries to predict how a user will rate a given movie
    
    
    #this just makes a matrix where the rows are users, columns are movies, and any given cell is that user's rating for the movie
    train_matrix = ratings.pivot_table(index="UserID", columns="MovieID", values="Rating")
      
    train_filled = train_matrix.fillna(0)
    sparse_train = csr_matrix(train_filled.values)
    
    # Center ratings by user mean to remove bias
    user_means = train_matrix.mean(axis=1)
    train_centered = train_matrix.sub(user_means, axis=0).fillna(0)
    
    #make an item matrix for seeing how similar movies are 
    item_matrix = cosine_similarity(sparse_train.T, dense_output=False)
    
    return train_matrix, item_matrix, train_centered, user_means 
    
def predict_cf(user_id, movie_id, train_matrix, item_matrix, train_centered, user_means):
    movie_ids = train_matrix.columns.tolist()
    user_ids = train_matrix.index.tolist()
    
    movie_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    m_idx = movie_to_idx[movie_id]
    u_idx = user_to_idx[user_id]
    
    #fallbacks
    if movie_id not in movie_to_idx:
        return user_means.mean()
    if user_id not in user_to_idx:
        return user_means.mean()
    
    user_vector = train_centered.iloc[u_idx].values
    item_similarities = item_matrix[m_idx].toarray().flatten()
    
    numerator = np.dot(item_similarities, user_vector)
    denominator = np.abs(item_similarities).sum()
    
    if denominator == 0:
        return user_means.iloc[u_idx]
    return user_means.iloc[u_idx] + (numerator/denominator)

def ratings_engine(ratings):
    #have to do a test/train split (speficy a random state so that it's reproducible
    #during development)
    train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)
    
    train_matrix, item_matrix, train_centered, user_means = basic_ratings_matrix(train_df)
    
    #this part makes it so that the test is only on user x movie pairs that exist and can, yknow, be tested    
    test_df = test_df[test_df['MovieID'].isin(train_matrix.columns)]
    test_df = test_df[test_df['UserID'].isin(train_matrix.index)]
    
    test_preds = [predict_cf(row['UserID'], row['MovieID'], train_matrix, item_matrix, train_centered, user_means) for _, row in test_df.iterrows()]
    rmse = np.sqrt(mean_squared_error(test_df['Rating'], test_preds))
    mae = mean_absolute_error(test_df['Rating'], test_preds)
    print(f"RMSE: {rmse:.3f} | MAE: {mae:.3f}")

def main():
    users, movies, ratings = load_data(dataset_folder)
    ratings_engine(ratings)
    
    
    

if __name__ == '__main__':
    main()