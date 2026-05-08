"""
plotting.py
Modular visualization utilities for movie recommendation evaluation.
All code is function-wrapped; no top-level execution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import Optional, List, Dict, Union

# Set default styling (applied when functions are called)
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)
DEFAULT_DPI = 300
DEFAULT_FIGSIZE = (10, 6)


def evaluate_model_coverage(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    min_ratings: int,
    alpha: float = 0.7,
    n_neighbors: int = 10,
    test_size: float = 0.2,
    random_state: int = 42
) -> Optional[Dict[str, Union[float, int]]]:
    """
    Evaluate a single min_ratings configuration and return metrics.
    
    Returns None if too few valid test interactions.
    """
    # Lazy import to avoid circular dependencies
    from movie_rec import HybridUserClusterKNNRecommender
    
    train_df, test_df = train_test_split(
        ratings_df, test_size=test_size, random_state=random_state
    )
    
    model = HybridUserClusterKNNRecommender(
        n_neighbors=n_neighbors,
        min_ratings=min_ratings,
        alpha=alpha
    )
    model.fit(train_df, movies_df)
    
    # Filter test to seen users/movies
    valid_test = test_df[
        (test_df['UserID'].isin(model._user_ids)) & 
        (test_df['MovieID'].isin(model._movie_ids))
    ]
    
    if len(valid_test) < 50:
        return None
        
    # Predict & compute RMSE
    preds = model.predict_batch(
        valid_test['UserID'].values, 
        valid_test['MovieID'].values,
        alpha=alpha
    )
    rmse = np.sqrt(mean_squared_error(valid_test['Rating'], preds))
    
    # Coverage metrics
    total_users = ratings_df['UserID'].nunique()
    total_movies = movies_df['MovieID'].nunique()
    total_interactions = len(test_df)
    
    return {
        'min_ratings': min_ratings,
        'rmse': rmse,
        'user_coverage': len(model._user_ids) / total_users * 100,
        'movie_coverage': len(model._movie_ids) / total_movies * 100,
        'interaction_coverage': len(valid_test) / total_interactions * 100,
        'n_users': len(model._user_ids),
        'n_movies': len(model._movie_ids),
        'valid_test_size': len(valid_test)
    }


def sweep_min_ratings(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    min_ratings_values: List[int],
    alpha: float = 0.7,
    n_neighbors: int = 10,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Sweep multiple min_ratings values and return a DataFrame of results.
    """
    results = []
    
    for mr in min_ratings_values:
        if verbose:
            print(f"🔍 Evaluating min_ratings={mr}...")
            
        metrics = evaluate_model_coverage(
            ratings_df, movies_df, 
            min_ratings=mr, 
            alpha=alpha, 
            n_neighbors=n_neighbors
        )
        
        if metrics:
            results.append(metrics)
            if verbose:
                print(f"   ✓ RMSE={metrics['rmse']:.3f}, "
                      f"Coverage={metrics['interaction_coverage']:.1f}%")
        elif verbose:
            print(f"   ✗ Skipped (too few valid interactions)")
    
    return pd.DataFrame(results)


def plot_coverage_vs_rmse(
    results_df: pd.DataFrame,
    sweet_spot_threshold: float = 70.0,
    sweet_spot_rmse_max: float = 0.95,
    output_path_png: Optional[str] = 'coverage_vs_rmse_tradeoff.png',
    output_path_pdf: Optional[str] = 'coverage_vs_rmse_tradeoff.pdf',
    figsize: tuple = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
    show_plot: bool = True
) -> plt.Figure:
    """
    Generate dual-axis plot of RMSE vs. Coverage.
    
    Args:
        results_df: DataFrame from sweep_min_ratings()
        sweet_spot_threshold: Min coverage % to annotate as "sweet spot"
        sweet_spot_rmse_max: Max RMSE to qualify for sweet spot annotation
        output_path_png: Path to save PNG (None to skip)
        output_path_pdf: Path to save PDF (None to skip)
        figsize: Figure dimensions
        dpi: Resolution for raster output
        show_plot: Whether to call plt.show()
    
    Returns:
        matplotlib Figure object for further customization
    """
    if results_df.empty:
        raise ValueError("results_df is empty. Run sweep_min_ratings first.")
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Primary axis: RMSE (lower is better)
    color_rmse = '#2E86AB'
    ax1.set_xlabel('Minimum Ratings Threshold (min_ratings)', 
                   fontsize=11, fontweight='bold')
    ax1.set_ylabel('RMSE (Rating Prediction Error)', 
                   color=color_rmse, fontsize=11, fontweight='bold')
    
    line1 = ax1.plot(
        results_df['min_ratings'], 
        results_df['rmse'], 
        marker='o', 
        color=color_rmse, 
        linewidth=2.5, 
        markersize=8, 
        label='RMSE'
    )
    ax1.tick_params(axis='y', labelcolor=color_rmse)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Secondary axis: Coverage % (higher is better)
    color_cov = '#A23B72'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Interaction Coverage (%)', 
                   color=color_cov, fontsize=11, fontweight='bold')
    
    line2 = ax2.plot(
        results_df['min_ratings'], 
        results_df['interaction_coverage'], 
        marker='s', 
        color=color_cov, 
        linewidth=2.5, 
        markersize=8, 
        label='Coverage'
    )
    ax2.tick_params(axis='y', labelcolor=color_cov)
    
    # Annotate sweet spot
    sweet_mask = (
        (results_df['interaction_coverage'] >= sweet_spot_threshold) & 
        (results_df['rmse'] <= sweet_spot_rmse_max)
    )
    
    if sweet_mask.any():
        sweet = results_df[sweet_mask].iloc[0]
        ax1.annotate(
            f'Sweet Spot\n(min_ratings={sweet["min_ratings"]})',
            xy=(sweet['min_ratings'], sweet['rmse']),
            xytext=(sweet['min_ratings'] + 20, sweet['rmse'] + 0.02),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
            fontsize=9, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
        )
    
    # Title and legend
    plt.title(
        'Accuracy-Coverage Trade-off: min_ratings vs. RMSE', 
        fontsize=14, 
        fontweight='bold', 
        pad=20
    )
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10)
    
    # Save outputs
    plt.tight_layout()
    
    if output_path_png:
        fig.savefig(output_path_png, bbox_inches='tight', dpi=dpi)
    if output_path_pdf:
        fig.savefig(output_path_pdf, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    return fig


def generate_coverage_report(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    min_ratings_values: Optional[List[int]] = None,
    alpha: float = 0.7,
    n_neighbors: int = 10,
    output_plot: bool = True,
    output_table: bool = True,
    verbose: bool = True
) -> Dict[str, Union[pd.DataFrame, plt.Figure]]:
    """
    High-level convenience function: sweep + plot + return results.
    
    Returns:
        dict with keys:
            - 'results_df': DataFrame of all metrics
            - 'figure': matplotlib Figure (if output_plot=True)
            - 'sweet_spot': dict of best balanced config (if found)
    """
    if min_ratings_values is None:
        min_ratings_values = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
    
    # Run sweep
    results_df = sweep_min_ratings(
        ratings_df, movies_df, 
        min_ratings_values=min_ratings_values,
        alpha=alpha,
        n_neighbors=n_neighbors,
        verbose=verbose
    )
    
    output = {'results_df': results_df}
    
    # Generate plot
    if output_plot and not results_df.empty:
        fig = plot_coverage_vs_rmse(results_df, show_plot=verbose)
        output['figure'] = fig
    
    # Print table
    if output_table and not results_df.empty and verbose:
        print("\n📋 Summary Table for Report:")
        print(results_df[['min_ratings', 'rmse', 'interaction_coverage']].to_string(index=False))
    
    # Identify sweet spot
    if not results_df.empty:
        sweet_mask = (
            (results_df['interaction_coverage'] >= 70) & 
            (results_df['rmse'] <= 0.95)
        )
        if sweet_mask.any():
            output['sweet_spot'] = results_df[sweet_mask].iloc[0].to_dict()
    
    return output


if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path to import movie_rec
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load your data (adjust paths as needed)
    ratings = pd.read_csv('reviews.dat', sep='::', engine='python', header=None,
                          names=['UserID','MovieID','Rating','Timestamp'])
    movies = pd.read_csv('movies.dat', sep='::', engine='python', header=None, 
                         encoding='latin-1', names=['MovieID','Title','Genres'])
    
    # Generate full report
    report = generate_coverage_report(
        ratings_df=ratings,
        movies_df=movies,
        alpha=0.7,
        n_neighbors=10,
        verbose=True
    )
    
    # Print sweet spot if found
    if 'sweet_spot' in report:
        ss = report['sweet_spot']
        print(f"\n Recommended Configuration:")
        print(f"   min_ratings = {ss['min_ratings']}")
        print(f"   Expected RMSE = {ss['rmse']:.3f}")
        print(f"   Interaction Coverage = {ss['interaction_coverage']:.1f}%")