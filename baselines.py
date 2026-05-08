"""
baselines.py
Cross-validation utilities for baseline recommendation models.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class BaselineResult:
    """Container for baseline evaluation metrics."""
    model_name: str
    fold_rmse: List[float]
    mean_rmse: float
    std_rmse: float
    mean_coverage: float
    n_folds_evaluated: int


class GlobalMeanBaseline:
    """Predicts the global mean rating for all user-movie pairs."""
    def __init__(self):
        self.global_mean: Optional[float] = None
        self.is_fitted: bool = False
    
    def fit(self, ratings_df: pd.DataFrame) -> 'GlobalMeanBaseline':
        self.global_mean = float(ratings_df['Rating'].mean())
        self.is_fitted = True
        return self
    
    def predict(self, user_id: int, movie_id: int) -> float:
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .predict()")
        return self.global_mean
    
    def predict_batch(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .predict_batch()")
        return np.full(len(user_ids), self.global_mean)


class UserMeanBaseline:
    """Predicts each user's historical mean rating, falling back to global mean."""
    def __init__(self):
        self.user_means: Optional[Dict[int, float]] = None
        self.global_mean: Optional[float] = None
        self._user_ids: Optional[List[int]] = None
        self.is_fitted: bool = False
    
    def fit(self, ratings_df: pd.DataFrame) -> 'UserMeanBaseline':
        self.global_mean = float(ratings_df['Rating'].mean())
        user_means_series = ratings_df.groupby('UserID')['Rating'].mean()
        self.user_means = {int(uid): float(mean) for uid, mean in user_means_series.items()}
        self._user_ids = list(self.user_means.keys())
        self.is_fitted = True
        return self
    
    def predict(self, user_id: int, movie_id: int) -> float:
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .predict()")
        return float(self.user_means.get(user_id, self.global_mean))
    
    def predict_batch(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .predict_batch()")
        preds = np.array([self.user_means.get(int(u), self.global_mean) for u in user_ids])
        return np.clip(preds, 1.0, 5.0)


class BiasBaseline:
    """Bias model: r̂_ui = μ + b_u + b_i using regularized ALS."""
    def __init__(self, lambda_reg: float = 15.0, n_iterations: int = 5):
        self.lambda_reg = lambda_reg
        self.n_iterations = n_iterations
        self.global_mean: Optional[float] = None
        self.user_bias: Optional[np.ndarray] = None
        self.item_bias: Optional[np.ndarray] = None
        self._user_ids: Optional[List[int]] = None
        self._movie_ids: Optional[List[int]] = None
        self._user_to_idx: Optional[Dict[int, int]] = None
        self._movie_to_idx: Optional[Dict[int, int]] = None
        self.is_fitted: bool = False
    
    def fit(self, ratings_df: pd.DataFrame, min_ratings: int = 5) -> 'BiasBaseline':
        user_movie = ratings_df.pivot_table(index='UserID', columns='MovieID', values='Rating')
        active_users = user_movie.count(axis=1) >= min_ratings
        active_movies = user_movie.count(axis=0) >= min_ratings
        df = user_movie.loc[active_users, active_movies].fillna(0)
        if df.empty:
            raise ValueError("No users/movies meet min_ratings threshold")
        
        self._user_ids = [int(uid) for uid in df.index]
        self._movie_ids = [int(mid) for mid in df.columns]
        self._user_to_idx = {uid: i for i, uid in enumerate(self._user_ids)}
        self._movie_to_idx = {mid: i for i, mid in enumerate(self._movie_ids)}
        
        X = df.values
        self.global_mean = float(np.mean(X[X > 0]))
        n_users, n_movies = X.shape
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_movies)
        
        valid_ratings = ratings_df[
            (ratings_df['UserID'].isin(self._user_ids)) & 
            (ratings_df['MovieID'].isin(self._movie_ids))
        ].copy()
        if valid_ratings.empty:
            self.is_fitted = True
            return self
            
        valid_ratings['u_idx'] = valid_ratings['UserID'].map(self._user_to_idx)
        valid_ratings['i_idx'] = valid_ratings['MovieID'].map(self._movie_to_idx)
        valid_ratings = valid_ratings.dropna(subset=['u_idx', 'i_idx'])
        
        for _ in range(self.n_iterations):
            for u_idx in range(n_users):
                user_data = valid_ratings[valid_ratings['u_idx'] == u_idx]
                if len(user_data) == 0: continue
                residuals = user_data['Rating'].values - self.global_mean - self.item_bias[user_data['i_idx'].values]
                self.user_bias[u_idx] = residuals.sum() / (self.lambda_reg + len(user_data))
            
            for i_idx in range(n_movies):
                item_data = valid_ratings[valid_ratings['i_idx'] == i_idx]
                if len(item_data) == 0: continue
                residuals = item_data['Rating'].values - self.global_mean - self.user_bias[item_data['u_idx'].values]
                self.item_bias[i_idx] = residuals.sum() / (self.lambda_reg + len(item_data))
        
        self.is_fitted = True
        return self
    
    def predict(self, user_id: int, movie_id: int) -> float:
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .predict()")
        if user_id not in self._user_to_idx or movie_id not in self._movie_to_idx:
            return self.global_mean
        pred = self.global_mean + self.user_bias[self._user_to_idx[user_id]] + self.item_bias[self._movie_to_idx[movie_id]]
        return float(np.clip(pred, 1.0, 5.0))
    
    def predict_batch(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .predict_batch()")
        preds = np.full(len(user_ids), self.global_mean)
        for i, (u, m) in enumerate(zip(user_ids, movie_ids)):
            if u in self._user_to_idx and m in self._movie_to_idx:
                pred = self.global_mean + self.user_bias[self._user_to_idx[u]] + self.item_bias[self._movie_to_idx[m]]
                preds[i] = np.clip(pred, 1.0, 5.0)
        return preds


def evaluate_baseline_cv(baseline_model, ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
                         n_splits: int = 5, min_ratings: int = 5, random_state: int = 42,
                         verbose: bool = True) -> BaselineResult:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_rmses, fold_coverages = [], []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(ratings_df)):
        train_df, test_df = ratings_df.iloc[train_idx], ratings_df.iloc[test_idx]
        try:
            if isinstance(baseline_model, BiasBaseline):
                baseline_model.fit(train_df, min_ratings=min_ratings)
            else:
                baseline_model.fit(train_df)
            
            # Filter test to seen users for fair comparison with hybrid model
            if hasattr(baseline_model, '_user_ids') and baseline_model._user_ids is not None:
                valid_test = test_df[test_df['UserID'].isin(baseline_model._user_ids)]
            else:
                valid_test = test_df
            
            if len(valid_test) < 20:
                if verbose: print(f"  Fold {fold_idx+1}: Skipped")
                continue
                
            preds = baseline_model.predict_batch(valid_test['UserID'].values, valid_test['MovieID'].values)
            rmse = np.sqrt(mean_squared_error(valid_test['Rating'], preds))
            fold_rmses.append(rmse)
            fold_coverages.append(len(valid_test) / len(test_df) * 100)
            if verbose: print(f"  Fold {fold_idx+1}: RMSE={rmse:.3f}, Coverage={fold_coverages[-1]:.1f}%")
        except Exception as e:
            if verbose: print(f"  Fold {fold_idx+1}: Failed - {e}")
            
    return BaselineResult(
        model_name=baseline_model.__class__.__name__,
        fold_rmse=fold_rmses,
        mean_rmse=float(np.mean(fold_rmses)),
        std_rmse=float(np.std(fold_rmses)),
        mean_coverage=float(np.mean(fold_coverages)),
        n_folds_evaluated=len(fold_rmses)
    )


def compare_baselines_cv(ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
                         n_splits: int = 5, min_ratings_values: Optional[List[int]] = None,
                         random_state: int = 42, verbose: bool = True) -> pd.DataFrame:
    if min_ratings_values is None:
        min_ratings_values = [5, 10, 20, 40]
    
    results = []
    
    # 1. Global Mean
    if verbose: print("\nEvaluating GlobalMeanBaseline...")
    gm = GlobalMeanBaseline()
    r = evaluate_baseline_cv(gm, ratings_df, movies_df, n_splits=n_splits, verbose=verbose)
    results.append({'model': 'GlobalMean', 'min_ratings': None, 'mean_rmse': r.mean_rmse, 
                    'std_rmse': r.std_rmse, 'mean_coverage': r.mean_coverage, 'n_folds': r.n_folds_evaluated})
    
    # 2. User Mean
    if verbose: print("\nEvaluating UserMeanBaseline...")
    um = UserMeanBaseline()
    r = evaluate_baseline_cv(um, ratings_df, movies_df, n_splits=n_splits, verbose=verbose)
    results.append({'model': 'UserMean', 'min_ratings': None, 'mean_rmse': r.mean_rmse, 
                    'std_rmse': r.std_rmse, 'mean_coverage': r.mean_coverage, 'n_folds': r.n_folds_evaluated})
    
    # 3. Bias Baseline (across min_ratings)
    for min_r in min_ratings_values:
        if verbose: print(f"\nEvaluating BiasBaseline (min_ratings={min_r})...")
        bias = BiasBaseline(lambda_reg=15.0)
        r = evaluate_baseline_cv(bias, ratings_df, movies_df, n_splits=n_splits, min_ratings=min_r, verbose=verbose)
        results.append({'model': 'Bias', 'min_ratings': min_r, 'mean_rmse': r.mean_rmse, 
                        'std_rmse': r.std_rmse, 'mean_coverage': r.mean_coverage, 'n_folds': r.n_folds_evaluated})
        
    results_df = pd.DataFrame(results)
    if verbose: print(f"\nBaseline Comparison Summary:\n{results_df.to_string(index=False)}")
    return results_df


def plot_baseline_comparison(results_df: pd.DataFrame, hybrid_rmse: Optional[float] = None,
                             hybrid_coverage: Optional[float] = None,
                             output_path_png: Optional[str] = 'baseline_comparison.png',
                             output_path_pdf: Optional[str] = 'baseline_comparison.pdf',
                             figsize: tuple = (9, 5), dpi: int = 300, show_plot: bool = True) -> 'plt.Figure':
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.0)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    plot_data = results_df.copy()
    plot_data['label'] = plot_data.apply(
        lambda row: f"{row['model']}\n(min_ratings={row['min_ratings']})" 
        if pd.notna(row['min_ratings']) else row['model'], axis=1)
    
    colors = {'GlobalMean': '#E63946', 'UserMean': '#457B9D', 'Bias': '#2A9D8F'}
    bars = ax.barh(plot_data['label'], plot_data['mean_rmse'], xerr=plot_data['std_rmse'],
                   color=[colors.get(row['model'], '#A8DADC') for _, row in plot_data.iterrows()],
                   capsize=4, alpha=0.9)
    
    if hybrid_rmse is not None:
        ax.axvline(x=hybrid_rmse, color='#F4A261', linestyle='--', linewidth=2, 
                   label=f'Your Hybrid (RMSE={hybrid_rmse:.3f})')
        if hybrid_coverage is not None:
            ax.text(hybrid_rmse + 0.01, len(plot_data) - 0.5, f'Coverage: {hybrid_coverage:.1f}%', 
                    color='#F4A261', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('RMSE (Lower is Better)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Model Configuration', fontsize=11, fontweight='bold')
    ax.set_title('Baseline Comparison: 5-Fold Cross-Validation', fontsize=13, fontweight='bold', pad=15)
    ax.xaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    if hybrid_rmse is not None: ax.legend(loc='lower right', fontsize=9)
    
    for i, (_, row) in enumerate(plot_data.iterrows()):
        ax.text(row['mean_rmse'] + 0.01, i, f"{row['mean_coverage']:.1f}% cov", 
                va='center', fontsize=8, alpha=0.8)
    
    plt.tight_layout()
    if output_path_png: fig.savefig(output_path_png, bbox_inches='tight', dpi=dpi)
    if output_path_pdf: fig.savefig(output_path_pdf, bbox_inches='tight')
    if show_plot: plt.show()
    return fig


def generate_baseline_report(ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
                             hybrid_rmse: Optional[float] = None, hybrid_coverage: Optional[float] = None,
                             n_splits: int = 5, min_ratings_values: Optional[List[int]] = None,
                             output_plot: bool = True, verbose: bool = True) -> Dict[str, Union[pd.DataFrame, 'plt.Figure', str]]:
    if verbose: print("🚀 Starting baseline evaluation...")
    results_df = compare_baselines_cv(ratings_df, movies_df, n_splits=n_splits, min_ratings_values=min_ratings_values, verbose=verbose)
    
    output = {'results_df': results_df}
    if output_plot:
        output['figure'] = plot_baseline_comparison(results_df, hybrid_rmse=hybrid_rmse, 
                                                    hybrid_coverage=hybrid_coverage, show_plot=verbose)
    
    best_bias = results_df[results_df['model'] == 'Bias'].loc[results_df[results_df['model'] == 'Bias']['mean_rmse'].idxmin()] if (results_df['model'] == 'Bias').any() else None
    summary_lines = ["\nBaseline Evaluation Summary", "=" * 40, f"Cross-validation: {n_splits}-fold", f"Total ratings: {len(ratings_df):,}", "", "Results:"]
    for _, row in results_df.iterrows():
        m = row['model'] + (f" (min_ratings={int(row['min_ratings'])})" if pd.notna(row['min_ratings']) else "")
        summary_lines.append(f"  • {m:30s} → RMSE: {row['mean_rmse']:.3f} ± {row['std_rmse']:.3f}  (coverage: {row['mean_coverage']:.1f}%)")
    if hybrid_rmse is not None:
        summary_lines.append(f"\n  • Hybrid Model          → RMSE: {hybrid_rmse:.3f}  (coverage: {hybrid_coverage:.1f}%)")
        best_base_rmse = results_df['mean_rmse'].min()
        summary_lines.append(f"\nImprovement vs. best baseline: {(best_base_rmse - hybrid_rmse)/best_base_rmse*100:+.2f}%")
    output['summary'] = "\n".join(summary_lines)
    if verbose: print(output['summary'])
    return output

