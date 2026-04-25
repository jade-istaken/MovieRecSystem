"""
baselines.py
Cross-validation utilities for baseline recommendation models.
All code is function-wrapped; no top-level execution.
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
        """Compute global mean from training ratings."""
        self.global_mean = float(ratings_df['Rating'].mean())
        self.is_fitted = True
        return self
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """Return global mean for any input."""
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .predict()")
        return self.global_mean
    
    def predict_batch(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        """Vectorized prediction: returns array of global means."""
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .predict_batch()")
        return np.full(len(user_ids), self.global_mean)


class BiasBaseline:
    """
    Bias model: r̂_ui = μ + b_u + b_i
    Uses regularized alternating least squares for bias estimation.
    """
    
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
    
    def fit(self, ratings_df: pd.DataFrame, 
            min_ratings: int = 5) -> 'BiasBaseline':
        """
        Fit bias model on filtered ratings.
        
        Args:
            ratings_df: DataFrame with UserID, MovieID, Rating columns
            min_ratings: Minimum ratings for user/movie inclusion
        """
        # Build user-movie matrix
        user_movie = ratings_df.pivot_table(
            index='UserID', columns='MovieID', values='Rating'
        )
        
        # Filter active users/movies
        active_users = user_movie.count(axis=1) >= min_ratings
        active_movies = user_movie.count(axis=0) >= min_ratings
        df = user_movie.loc[active_users, active_movies].fillna(0)
        
        if df.empty:
            raise ValueError("No users/movies meet min_ratings threshold")
        
        # Store mappings with type safety
        self._user_ids = [int(uid) for uid in df.index]
        self._movie_ids = [int(mid) for mid in df.columns]
        self._user_to_idx = {uid: i for i, uid in enumerate(self._user_ids)}
        self._movie_to_idx = {mid: i for i, mid in enumerate(self._movie_ids)}
        
        X = df.values
        self.global_mean = float(np.mean(X[X > 0]))
        
        # Initialize biases
        n_users, n_movies = X.shape
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_movies)
        
        # Build index lookup for fast iteration
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
        
        # Alternating Least Squares
        for _ in range(self.n_iterations):
            # Update user biases
            for u_idx in range(n_users):
                user_data = valid_ratings[valid_ratings['u_idx'] == u_idx]
                if len(user_data) == 0:
                    continue
                residuals = (user_data['Rating'].values - self.global_mean - 
                           self.item_bias[user_data['i_idx'].values])
                self.user_bias[u_idx] = residuals.sum() / (self.lambda_reg + len(user_data))
            
            # Update item biases
            for i_idx in range(n_movies):
                item_data = valid_ratings[valid_ratings['i_idx'] == i_idx]
                if len(item_data) == 0:
                    continue
                residuals = (item_data['Rating'].values - self.global_mean - 
                           self.user_bias[item_data['u_idx'].values])
                self.item_bias[i_idx] = residuals.sum() / (self.lambda_reg + len(item_data))
        
        self.is_fitted = True
        return self
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating using μ + b_u + b_i."""
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .predict()")
        
        # Cold-start fallback
        if (user_id not in self._user_to_idx or 
            movie_id not in self._movie_to_idx):
            return self.global_mean
        
        u_idx = self._user_to_idx[user_id]
        m_idx = self._movie_to_idx[movie_id]
        
        pred = self.global_mean + self.user_bias[u_idx] + self.item_bias[m_idx]
        return float(np.clip(pred, 1.0, 5.0))
    
    def predict_batch(self, user_ids: np.ndarray, 
                      movie_ids: np.ndarray) -> np.ndarray:
        """Vectorized bias prediction."""
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .predict_batch()")
        
        preds = np.full(len(user_ids), self.global_mean)
        
        for i, (u, m) in enumerate(zip(user_ids, movie_ids)):
            if u in self._user_to_idx and m in self._movie_to_idx:
                u_idx = self._user_to_idx[u]
                m_idx = self._movie_to_idx[m]
                pred = self.global_mean + self.user_bias[u_idx] + self.item_bias[m_idx]
                preds[i] = np.clip(pred, 1.0, 5.0)
        
        return preds


def evaluate_baseline_cv(
    baseline_model: Union[GlobalMeanBaseline, BiasBaseline],
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    n_splits: int = 5,
    min_ratings: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> BaselineResult:
    """
    Run K-fold cross-validation on a baseline model.
    
    Args:
        baseline_model: Instantiated baseline class (GlobalMeanBaseline or BiasBaseline)
        ratings_df: Full ratings DataFrame
        movies_df: Movies DataFrame (for movie ID validation)
        n_splits: Number of CV folds
        min_ratings: Minimum ratings threshold for bias model filtering
        random_state: Random seed for fold splitting
        verbose: Print progress
    
    Returns:
        BaselineResult dataclass with aggregated metrics
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_rmses = []
    fold_coverages = []
    
    total_interactions = len(ratings_df)
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(ratings_df)):
        train_df = ratings_df.iloc[train_idx]
        test_df = ratings_df.iloc[test_idx]
        
        try:
            # Fit model (min_ratings only affects BiasBaseline)
            if isinstance(baseline_model, BiasBaseline):
                baseline_model.fit(train_df, min_ratings=min_ratings)
            else:
                baseline_model.fit(train_df)
            
            # Filter test to seen users/movies (for fair comparison with hybrid model)
            if hasattr(baseline_model, '_user_ids'):
                valid_test = test_df[
                    (test_df['UserID'].isin(baseline_model._user_ids)) & 
                    (test_df['MovieID'].isin(baseline_model._movie_ids))
                ]
            else:
                # GlobalMeanBaseline: all test pairs are valid
                valid_test = test_df
            
            if len(valid_test) < 20:
                if verbose:
                    print(f"  Fold {fold_idx+1}: Skipped (too few valid pairs)")
                continue
            
            # Predict and evaluate
            preds = baseline_model.predict_batch(
                valid_test['UserID'].values, 
                valid_test['MovieID'].values
            )
            rmse = np.sqrt(mean_squared_error(valid_test['Rating'], preds))
            
            coverage = len(valid_test) / len(test_df) * 100
            
            fold_rmses.append(rmse)
            fold_coverages.append(coverage)
            
            if verbose:
                print(f"  Fold {fold_idx+1}: RMSE={rmse:.3f}, Coverage={coverage:.1f}%")
                
        except Exception as e:
            if verbose:
                print(f"  Fold {fold_idx+1}: Failed - {e}")
            continue
    
    if not fold_rmses:
        raise RuntimeError("No folds produced valid results")
    
    return BaselineResult(
        model_name=baseline_model.__class__.__name__,
        fold_rmse=fold_rmses,
        mean_rmse=float(np.mean(fold_rmses)),
        std_rmse=float(np.std(fold_rmses)),
        mean_coverage=float(np.mean(fold_coverages)),
        n_folds_evaluated=len(fold_rmses)
    )


def compare_baselines_cv(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    n_splits: int = 5,
    min_ratings_values: Optional[List[int]] = None,
    random_state: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare Global Mean and Bias baselines across multiple min_ratings thresholds.
    
    Args:
        ratings_df: Full ratings DataFrame
        movies_df: Movies DataFrame
        n_splits: Number of CV folds
        min_ratings_values: List of min_ratings to test for BiasBaseline
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        DataFrame with one row per (model, min_ratings) combination
    """
    if min_ratings_values is None:
        min_ratings_values = [5, 10, 20, 50]
    
    results = []
    
    # 1. Global Mean Baseline (min_ratings doesn't apply)
    if verbose:
        print(f"\n🔍 Evaluating GlobalMeanBaseline (5-fold CV)...")
    
    gm_model = GlobalMeanBaseline()
    gm_result = evaluate_baseline_cv(
        gm_model, ratings_df, movies_df,
        n_splits=n_splits, random_state=random_state, verbose=verbose
    )
    
    results.append({
        'model': 'GlobalMean',
        'min_ratings': None,
        'mean_rmse': gm_result.mean_rmse,
        'std_rmse': gm_result.std_rmse,
        'mean_coverage': gm_result.mean_coverage,
        'n_folds': gm_result.n_folds_evaluated
    })
    
    # 2. Bias Baseline at each min_ratings threshold
    for min_r in min_ratings_values:
        if verbose:
            print(f"\n🔍 Evaluating BiasBaseline (min_ratings={min_r}, 5-fold CV)...")
        
        bias_model = BiasBaseline(lambda_reg=15.0)
        bias_result = evaluate_baseline_cv(
            bias_model, ratings_df, movies_df,
            n_splits=n_splits, min_ratings=min_r, 
            random_state=random_state, verbose=verbose
        )
        
        results.append({
            'model': 'Bias',
            'min_ratings': min_r,
            'mean_rmse': bias_result.mean_rmse,
            'std_rmse': bias_result.std_rmse,
            'mean_coverage': bias_result.mean_coverage,
            'n_folds': bias_result.n_folds_evaluated
        })
    
    results_df = pd.DataFrame(results)
    
    if verbose:
        print(f"\n📊 Baseline Comparison Summary:")
        print(results_df.to_string(index=False))
    
    return results_df


def plot_baseline_comparison(
    results_df: pd.DataFrame,
    hybrid_rmse: Optional[float] = None,
    hybrid_coverage: Optional[float] = None,
    output_path_png: Optional[str] = 'baseline_comparison.png',
    output_path_pdf: Optional[str] = 'baseline_comparison.pdf',
    figsize: tuple = (9, 5),
    dpi: int = 300,
    show_plot: bool = True
) -> 'plt.Figure':
    """
    Generate bar chart comparing baseline RMSE values.
    
    Args:
        results_df: DataFrame from compare_baselines_cv()
        hybrid_rmse: Optional RMSE of your hybrid model to overlay
        hybrid_coverage: Optional coverage of hybrid model
        output_path_png: Path to save PNG (None to skip)
        output_path_pdf: Path to save PDF (None to skip)
        figsize: Figure dimensions
        dpi: Resolution for raster output
        show_plot: Whether to call plt.show()
    
    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.0)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Prepare data for plotting
    plot_data = results_df.copy()
    plot_data['label'] = plot_data.apply(
        lambda row: f"{row['model']}\n(min_ratings={row['min_ratings']})" 
        if pd.notna(row['min_ratings']) else row['model'],
        axis=1
    )
    
    # Color scheme
    colors = {'GlobalMean': '#E63946', 'Bias': '#457B9D'}
    
    # Bar plot
    bars = ax.barh(
        plot_data['label'], 
        plot_data['mean_rmse'],
        xerr=plot_data['std_rmse'],
        color=[colors.get(row['model'], '#A8DADC') for _, row in plot_data.iterrows()],
        capsize=4,
        alpha=0.9
    )
    
    # Add hybrid model as reference line if provided
    if hybrid_rmse is not None:
        ax.axvline(x=hybrid_rmse, color='#2A9D8F', linestyle='--', 
                   linewidth=2, label=f'Your Hybrid (RMSE={hybrid_rmse:.3f})')
        if hybrid_coverage is not None:
            ax.text(hybrid_rmse + 0.01, len(plot_data) - 0.5, 
                   f'Coverage: {hybrid_coverage:.1f}%', 
                   color='#2A9D8F', fontsize=9, fontweight='bold')
    
    # Labels and title
    ax.set_xlabel('RMSE (Lower is Better)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Model Configuration', fontsize=11, fontweight='bold')
    ax.set_title('Baseline Comparison: 5-Fold Cross-Validation', 
                 fontsize=13, fontweight='bold', pad=15)
    
    # Format x-axis
    ax.xaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Legend
    if hybrid_rmse is not None:
        ax.legend(loc='lower right', fontsize=9)
    
    # Add coverage annotation to bars
    for i, (_, row) in enumerate(plot_data.iterrows()):
        ax.text(row['mean_rmse'] + 0.01, i, 
               f"{row['mean_coverage']:.1f}% cov", 
               va='center', fontsize=8, alpha=0.8)
    
    plt.tight_layout()
    
    # Save outputs
    if output_path_png:
        fig.savefig(output_path_png, bbox_inches='tight', dpi=dpi)
    if output_path_pdf:
        fig.savefig(output_path_pdf, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    return fig


def generate_baseline_report(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    hybrid_rmse: Optional[float] = None,
    hybrid_coverage: Optional[float] = None,
    n_splits: int = 5,
    min_ratings_values: Optional[List[int]] = None,
    output_plot: bool = True,
    verbose: bool = True
) -> Dict[str, Union[pd.DataFrame, 'plt.Figure', str]]:
    """
    High-level function: run baseline CV + generate comparison plot + return summary.
    
    Args:
        ratings_df: Full ratings DataFrame
        movies_df: Movies DataFrame  
        hybrid_rmse: Optional RMSE of your hybrid model for comparison
        hybrid_coverage: Optional coverage of hybrid model
        n_splits: Number of CV folds
        min_ratings_values: min_ratings values to test for BiasBaseline
        output_plot: Whether to generate and save comparison plot
        verbose: Print progress
    
    Returns:
        dict with keys:
            - 'results_df': DataFrame of all baseline metrics
            - 'figure': matplotlib Figure (if output_plot=True)
            - 'summary': Formatted text summary for report
    """
    if verbose:
        print("🚀 Starting baseline evaluation...")
    
    # Run comparison
    results_df = compare_baselines_cv(
        ratings_df, movies_df,
        n_splits=n_splits,
        min_ratings_values=min_ratings_values,
        verbose=verbose
    )
    
    output = {'results_df': results_df}
    
    # Generate plot
    if output_plot:
        fig = plot_baseline_comparison(
            results_df,
            hybrid_rmse=hybrid_rmse,
            hybrid_coverage=hybrid_coverage,
            show_plot=verbose
        )
        output['figure'] = fig
    
    # Generate text summary
    best_bias = results_df[results_df['model'] == 'Bias'].loc[
        results_df[results_df['model'] == 'Bias']['mean_rmse'].idxmin()
    ] if (results_df['model'] == 'Bias').any() else None
    
    summary_lines = [
        "\n📋 Baseline Evaluation Summary",
        "=" * 40,
        f"Cross-validation: {n_splits}-fold",
        f"Total ratings evaluated: {len(ratings_df):,}",
        "",
        "Results:",
    ]
    
    for _, row in results_df.iterrows():
        model_str = row['model']
        if pd.notna(row['min_ratings']):
            model_str += f" (min_ratings={int(row['min_ratings'])})"
        summary_lines.append(
            f"  • {model_str:25s} → RMSE: {row['mean_rmse']:.3f} ± {row['std_rmse']:.3f}  "
            f"(coverage: {row['mean_coverage']:.1f}%)"
        )
    
    if hybrid_rmse is not None:
        summary_lines.append(
            f"\n  • Your Hybrid Model      → RMSE: {hybrid_rmse:.3f}  "
            f"(coverage: {hybrid_coverage:.1f}%)"
        )
        
        # Compute improvement
        best_baseline_rmse = results_df['mean_rmse'].min()
        improvement = (best_baseline_rmse - hybrid_rmse) / best_baseline_rmse * 100
        summary_lines.append(
            f"\n🎯 Improvement vs. best baseline: {improvement:+.2f}%"
        )
    
    output['summary'] = "\n".join(summary_lines)
    
    if verbose:
        print(output['summary'])
    
    return output


# ============================================================================
# EXECUTION GUARD: Only run when executed as script
# ============================================================================
if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load data (adjust paths as needed)
    ratings = pd.read_csv('reviews.dat', sep='::', engine='python', header=None,
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    movies = pd.read_csv('movies.dat', sep='::', engine='python', header=None,
                         encoding='latin-1', names=['MovieID', 'Title', 'Genres'])
    
    # Optional: provide your hybrid model's metrics for comparison
    HYBRID_RMSE = 0.945      # Replace with your actual result
    HYBRID_COVERAGE = 82.4   # Replace with your actual coverage
    
    # Generate full report
    report = generate_baseline_report(
        ratings_df=ratings,
        movies_df=movies,
        hybrid_rmse=HYBRID_RMSE,
        hybrid_coverage=HYBRID_COVERAGE,
        n_splits=5,
        min_ratings_values=[5, 10, 20, 50],
        verbose=True
    )
    
    print(f"\n✅ Baseline evaluation complete.")
    print(f"   Results saved to: baseline_comparison.png/pdf")