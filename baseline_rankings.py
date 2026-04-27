"""
baselines_ranking.py
Simple baseline recommenders for top-N ranking evaluation.
All classes implement .fit(ratings_df, movies_df) and .recommend(user_id, n_rec).
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict


class RandomRecommender:
    """Recommends random unseen movies. Establishes the performance floor."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._all_movie_ids: Optional[List[int]] = None
        self._user_rated: Dict[int, set] = {}
        self._rng = np.random.default_rng(random_state)
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> 'RandomRecommender':
        self._all_movie_ids = movies_df['MovieID'].tolist()
        # Precompute rated items per user for fast exclusion
        self._user_rated = {
            int(uid): set(row['MovieID'].astype(int).tolist())
            for uid, row in ratings_df.groupby('UserID')
        }
        return self
    
    def recommend(self, user_id: int, n_rec: int = 10) -> List[int]:
        # Get items user has already rated
        rated = self._user_rated.get(user_id, set())
        # Candidate pool: unseen movies
        candidates = [m for m in self._all_movie_ids if m not in rated]
        if not candidates:
            return []  # No unseen items
        # Sample randomly without replacement
        n = min(n_rec, len(candidates))
        return self._rng.choice(candidates, size=n, replace=False).tolist()


class PopularityRecommender:
    """Recommends most-popular (most-rated) unseen movies. Strong simple baseline."""
    
    def __init__(self, popularity_metric: str = 'count', min_ratings: int = 5):
        """
        Args:
            popularity_metric: 'count' (most rated) or 'mean_rating' (highest avg)
            min_ratings: Minimum ratings for a movie to be considered popular
        """
        self.popularity_metric = popularity_metric
        self.min_ratings = min_ratings
        self._movie_scores: Dict[int, float] = {}
        self._all_movie_ids: Optional[List[int]] = None
        self._user_rated: Dict[int, set] = {}
        self._sorted_candidates: Optional[List[int]] = None
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> 'PopularityRecommender':
        self._all_movie_ids = movies_df['MovieID'].tolist()
        
        # Precompute rated items per user
        self._user_rated = {
            int(uid): set(row['MovieID'].astype(int).tolist())
            for uid, row in ratings_df.groupby('UserID')
        }
        
        # Compute popularity scores
        if self.popularity_metric == 'count':
            scores = ratings_df.groupby('MovieID').size()
        elif self.popularity_metric == 'mean_rating':
            counts = ratings_df.groupby('MovieID').size()
            means = ratings_df.groupby('MovieID')['Rating'].mean()
            # Only consider movies with enough ratings
            scores = means[counts >= self.min_ratings]
        else:
            raise ValueError(f"Unknown popularity_metric: {self.popularity_metric}")
        
        # Filter by min_ratings and store
        self._movie_scores = {
            int(mid): float(score) 
            for mid, score in scores.items() 
            if scores[mid] >= self.min_ratings
        }
        
        # Pre-sort candidates by popularity (descending)
        self._sorted_candidates = sorted(
            self._movie_scores.keys(), 
            key=lambda m: self._movie_scores[m], 
            reverse=True
        )
        
        return self
    
    def recommend(self, user_id: int, n_rec: int = 10) -> List[int]:
        # Get items user has already rated
        rated = self._user_rated.get(user_id, set())
        # Filter popular candidates to unseen items
        unseen = [m for m in self._sorted_candidates if m not in rated]
        return unseen[:n_rec]


class UserMeanPopularityRecommender:
    """
    Hybrid baseline: predict ratings using user mean + item bias,
    then recommend highest-predicted unseen movies.
    Bridges rating-prediction and ranking evaluation.
    """
    
    def __init__(self, lambda_reg: float = 15.0):
        self.lambda_reg = lambda_reg
        self._global_mean: Optional[float] = None
        self._user_means: Dict[int, float] = {}
        self._item_biases: Dict[int, float] = {}
        self._all_movie_ids: Optional[List[int]] = None
        self._user_rated: Dict[int, set] = {}
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> 'UserMeanPopularityRecommender':
        self._all_movie_ids = movies_df['MovieID'].tolist()
        self._user_rated = {
            int(uid): set(row['MovieID'].astype(int).tolist())
            for uid, row in ratings_df.groupby('UserID')
        }
        
        # Global mean
        self._global_mean = float(ratings_df['Rating'].mean())
        
        # User means (leniency/severity)
        self._user_means = {
            int(uid): float(row['Rating'].mean())
            for uid, row in ratings_df.groupby('UserID')
        }
        
        # Item biases via regularized estimation
        item_stats = ratings_df.groupby('MovieID').agg(
            count=('Rating', 'size'),
            mean=('Rating', 'mean')
        )
        for mid, row in item_stats.iterrows():
            if row['count'] >= self.lambda_reg:
                # Shrink toward global mean for sparse items
                shrinkage = row['count'] / (row['count'] + self.lambda_reg)
                bias = (row['mean'] - self._global_mean) * shrinkage
                self._item_biases[int(mid)] = float(bias)
            else:
                self._item_biases[int(mid)] = 0.0
        
        return self
    
    def _predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating: user_mean + item_bias (clipped to [1,5])."""
        user_mean = self._user_means.get(user_id, self._global_mean)
        item_bias = self._item_biases.get(movie_id, 0.0)
        return float(np.clip(user_mean + item_bias, 1.0, 5.0))
    
    def recommend(self, user_id: int, n_rec: int = 10) -> List[int]:
        rated = self._user_rated.get(user_id, set())
        candidates = [m for m in self._all_movie_ids if m not in rated]
        
        if not candidates:
            return []
        
        # Score candidates by predicted rating
        scored = [(m, self._predict(user_id, m)) for m in candidates]
        # Sort by predicted rating (descending), break ties by popularity
        scored.sort(key=lambda x: (-x[1], self._item_biases.get(x[0], 0)))
        
        return [m for m, _ in scored[:n_rec]]