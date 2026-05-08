"""
grid_search_viz.py
Grid search utilities with heatmap visualization for recommendation models.
All code is function-wrapped; no top-level execution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

# Set professional styling defaults
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.0)
DEFAULT_DPI = 300
DEFAULT_FIGSIZE = (10, 7)


@dataclass
class GridSearchResult:
    """Container for a single hyperparameter combination result."""
    min_ratings: int
    n_neighbors: int
    mean_rmse: float
    std_rmse: float
    mean_coverage: float
    n_folds_evaluated: int
    error: Optional[str] = None


def evaluate_single_config(
    model_class,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    min_ratings: int,
    n_neighbors: int,
    alpha: float = 0.7,
    n_splits: int = 5,
    random_state: int = 42
) -> GridSearchResult:
    """
    Evaluate a single (min_ratings, n_neighbors) configuration via K-fold CV.
    
    Returns GridSearchResult with aggregated metrics.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_rmses = []
    fold_coverages = []
    
    total_interactions = len(ratings_df)
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(ratings_df)):
        train_df = ratings_df.iloc[train_idx]
        test_df = ratings_df.iloc[test_idx]
        
        try:
            # Instantiate and fit model
            model = model_class(
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
                continue
                
            # Predict and evaluate
            preds = model.predict_batch(
                valid_test['UserID'].values,
                valid_test['MovieID'].values,
                alpha=alpha
            )
            rmse = np.sqrt(mean_squared_error(valid_test['Rating'], preds))
            coverage = len(valid_test) / len(test_df) * 100
            
            fold_rmses.append(rmse)
            fold_coverages.append(coverage)
            
        except Exception as e:
            # Return result with error flag
            return GridSearchResult(
                min_ratings=min_ratings,
                n_neighbors=n_neighbors,
                mean_rmse=np.nan,
                std_rmse=np.nan,
                mean_coverage=np.nan,
                n_folds_evaluated=0,
                error=str(e)
            )
    
    if not fold_rmses:
        return GridSearchResult(
            min_ratings=min_ratings,
            n_neighbors=n_neighbors,
            mean_rmse=np.nan,
            std_rmse=np.nan,
            mean_coverage=np.nan,
            n_folds_evaluated=0,
            error="No valid folds produced results"
        )
    
    return GridSearchResult(
        min_ratings=min_ratings,
        n_neighbors=n_neighbors,
        mean_rmse=float(np.mean(fold_rmses)),
        std_rmse=float(np.std(fold_rmses)),
        mean_coverage=float(np.mean(fold_coverages)),
        n_folds_evaluated=len(fold_rmses)
    )


def run_grid_search_cv(
    model_class,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    min_ratings_values: List[int],
    n_neighbors_values: List[int],
    alpha: float = 0.7,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run full grid search over min_ratings × n_neighbors with cross-validation.
    
    Args:
        model_class: Recommender class with .fit() and .predict_batch()
        ratings_df: Full ratings DataFrame
        movies_df: Movies DataFrame
        min_ratings_values: List of min_ratings to test
        n_neighbors_values: List of n_neighbors to test
        alpha: Hybrid blending weight (passed to model)
        n_splits: Number of CV folds
        random_state: Random seed for reproducibility
        verbose: Print progress
    
    Returns:
        DataFrame with one row per (min_ratings, n_neighbors) combination
    """
    results = []
    total = len(min_ratings_values) * len(n_neighbors_values)
    
    for i, min_r in enumerate(min_ratings_values):
        for j, n_n in enumerate(n_neighbors_values):
            if verbose:
                print(f"[{i*len(n_neighbors_values)+j+1}/{total}] "
                      f"min_ratings={min_r:3d}, n_neighbors={n_n:2d}...", 
                      end=" ", flush=True)
            
            result = evaluate_single_config(
                model_class, ratings_df, movies_df,
                min_ratings=min_r,
                n_neighbors=n_n,
                alpha=alpha,
                n_splits=n_splits,
                random_state=random_state
            )
            results.append(result)
            
            if verbose:
                if pd.isna(result.mean_rmse):
                    print(f"✗ Failed: {result.error}")
                else:
                    print(f"✓ RMSE={result.mean_rmse:.3f}±{result.std_rmse:.3f}, "
                          f"Coverage={result.mean_coverage:.1f}%")
    
    # Convert to DataFrame and pivot for heatmap
    results_df = pd.DataFrame([
        {
            'min_ratings': r.min_ratings,
            'n_neighbors': r.n_neighbors,
            'mean_rmse': r.mean_rmse,
            'std_rmse': r.std_rmse,
            'mean_coverage': r.mean_coverage,
            'n_folds': r.n_folds_evaluated,
            'error': r.error
        }
        for r in results
    ])
    
    return results_df


def plot_grid_search_heatmap(
    results_df: pd.DataFrame,
    metric: str = 'mean_rmse',
    annotate_values: bool = True,
    highlight_best: bool = False,
    coverage_overlay: bool = False,
    min_ratings_log_scale: bool = False,
    cmap: str = 'viridis_r',  # Reversed: lower RMSE = darker
    output_path_png: Optional[str] = 'grid_search_heatmap.png',
    output_path_pdf: Optional[str] = 'grid_search_heatmap.pdf',
    figsize: tuple = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
    show_plot: bool = True
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Generate heatmap visualization of grid search results.
    
    Args:
        results_df: DataFrame from run_grid_search_cv()
        metric: Column to visualize ('mean_rmse', 'mean_coverage', etc.)
        annotate_values: Whether to print values in cells
        highlight_best: Draw border around best configuration
        coverage_overlay: Add coverage as text annotation alongside RMSE
        min_ratings_log_scale: Use log scale for y-axis (helps with wide ranges)
        cmap: Matplotlib colormap name
        output_path_png: Path to save PNG (None to skip)
        output_path_pdf: Path to save PDF (None to skip)
        figsize: Figure dimensions
        dpi: Resolution for raster output
        show_plot: Whether to call plt.show()
    
    Returns:
        Tuple of (matplotlib Figure, pivot DataFrame used for plotting)
    """
    if results_df.empty:
        raise ValueError("results_df is empty. Run run_grid_search_cv first.")
    
    # Create pivot table for heatmap
    pivot_df = results_df.pivot_table(
        index='min_ratings',
        columns='n_neighbors',
        values=metric,
        aggfunc='first'  # Each combo appears once
    )
    
    # Create coverage pivot for optional overlay
    if coverage_overlay and 'mean_coverage' in results_df.columns:
        coverage_pivot = results_df.pivot_table(
            index='min_ratings',
            columns='n_neighbors',
            values='mean_coverage',
            aggfunc='first'
        )
    else:
        coverage_pivot = None
    
    # Set up figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Handle log scale for y-axis
    if min_ratings_log_scale:
        # Reindex with sorted unique values for proper ordering
        y_labels = sorted(pivot_df.index.unique())
        y_pos = np.arange(len(y_labels))
        y_mapping = {val: pos for pos, val in enumerate(y_labels)}
        
        # Create mapped index
        pivot_mapped = pivot_df.copy()
        pivot_mapped.index = pivot_mapped.index.map(y_mapping)
        pivot_mapped = pivot_mapped.sort_index()
        
        # Plot with mapped indices
        heatmap = sns.heatmap(
            pivot_mapped,
            annot=annotate_values,
            fmt='.3f' if metric == 'mean_rmse' else '.1f',
            cmap=cmap,
            cbar_kws={'label': metric.replace('_', ' ').title()},
            ax=ax,
            xticklabels=pivot_df.columns,
            yticklabels=[y_labels[i] for i in pivot_mapped.index]
        )
        ax.set_ylabel('Minimum Ratings (min_ratings) [log scale]')
    else:
        heatmap = sns.heatmap(
            pivot_df,
            annot=annotate_values,
            fmt='.3f' if metric == 'mean_rmse' else '.1f',
            cmap=cmap,
            cbar_kws={'label': metric.replace('_', ' ').title()},
            ax=ax
        )
        ax.set_ylabel('Minimum Ratings (min_ratings)')
    
    # Axis labels
    ax.set_xlabel('Number of Neighbors (n_neighbors)')
    
    # Title
    metric_name = metric.replace('_', ' ').title()
    plt.title(f'Grid Search Results: {metric_name}', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add coverage overlay if requested
    if coverage_overlay and coverage_pivot is not None:
        # Re-align coverage pivot to match heatmap structure
        for i, (min_r, row) in enumerate(pivot_df.iterrows()):
            for j, n_n in enumerate(pivot_df.columns):
                if pd.notna(row[n_n]) and pd.notna(coverage_pivot.loc[min_r, n_n]):
                    rmse_val = row[n_n]
                    cov_val = coverage_pivot.loc[min_r, n_n]
                    # Add coverage as smaller text below RMSE
                    ax.text(j + 0.5, i + 0.75, f'{cov_val:.0f}% cov', 
                           ha='center', va='top', fontsize=7, color='white', 
                           fontweight='bold', alpha=0.9)
    
    # Highlight best configuration
    if highlight_best and metric == 'mean_rmse':
        # Find best (lowest RMSE) valid result
        valid_results = results_df[results_df['mean_rmse'].notna()]
        if not valid_results.empty:
            best_idx = valid_results['mean_rmse'].idxmin()
            best = valid_results.loc[best_idx]
            
            # Find cell position
            if min_ratings_log_scale:
                y_pos = sorted(pivot_df.index.unique()).index(best['min_ratings'])
            else:
                y_pos = list(pivot_df.index).index(best['min_ratings'])
            x_pos = list(pivot_df.columns).index(best['n_neighbors'])
            
            # Draw rectangle around best cell
            rect = plt.Rectangle(
                (x_pos, y_pos), 1, 1,
                fill=False, edgecolor='gold', linewidth=3,
                label=f'Best: RMSE={best["mean_rmse"]:.3f}'
            )
            ax.add_patch(rect)
            
            # Add legend for best marker
            if not ax.get_legend():
                ax.plot([], [], color='gold', linewidth=3, label='Best Configuration')
                ax.legend(loc='upper right', fontsize=9)
    
    # Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Tight layout
    plt.tight_layout()
    
    # Save outputs
    if output_path_png:
        fig.savefig(output_path_png, bbox_inches='tight', dpi=dpi)
    if output_path_pdf:
        fig.savefig(output_path_pdf, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    return fig, pivot_df


def generate_grid_search_report(
    model_class,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    min_ratings_values: Optional[List[int]] = None,
    n_neighbors_values: Optional[List[int]] = None,
    alpha: float = 0.7,
    n_splits: int = 5,
    output_plot: bool = True,
    coverage_overlay: bool = True,
    verbose: bool = True
) -> Dict[str, Union[pd.DataFrame, plt.Figure, str]]:
    """
    High-level convenience function: run grid search + generate heatmap + return summary.
    
    Args:
        model_class: Recommender class to tune
        ratings_df: Full ratings DataFrame
        movies_df: Movies DataFrame
        min_ratings_values: List of min_ratings to test (default: [5,10,20,50,100])
        n_neighbors_values: List of n_neighbors to test (default: [5,10,20,30,50])
        alpha: Hybrid blending weight
        n_splits: Number of CV folds
        output_plot: Whether to generate and save heatmap
        coverage_overlay: Add coverage annotations to heatmap cells
        verbose: Print progress
    
    Returns:
        dict with keys:
            - 'results_df': Full DataFrame of all combinations
            - 'pivot_df': Pivoted DataFrame used for heatmap
            - 'figure': matplotlib Figure (if output_plot=True)
            - 'best_config': Dict of best hyperparameters
            - 'summary': Formatted text summary for report
    """
    # Default parameter grids
    if min_ratings_values is None:
        min_ratings_values = [5, 10, 20, 50, 100]
    if n_neighbors_values is None:
        n_neighbors_values = [5, 10, 20, 30, 50]
    
    if verbose:
        print(f"Starting grid search: {len(min_ratings_values)} × {len(n_neighbors_values)} = "
              f"{len(min_ratings_values)*len(n_neighbors_values)} configurations")
        print(f"   Model: {model_class.__name__}, α={alpha}, {n_splits}-fold CV\n")
    
    # Run grid search
    results_df = run_grid_search_cv(
        model_class, ratings_df, movies_df,
        min_ratings_values=min_ratings_values,
        n_neighbors_values=n_neighbors_values,
        alpha=alpha,
        n_splits=n_splits,
        verbose=verbose
    )
    
    # Find best configuration
    valid_results = results_df[results_df['mean_rmse'].notna()]
    best_config = None
    if not valid_results.empty:
        best_idx = valid_results['mean_rmse'].idxmin()
        best = valid_results.loc[best_idx]
        best_config = {
            'min_ratings': int(best['min_ratings']),
            'n_neighbors': int(best['n_neighbors']),
            'mean_rmse': float(best['mean_rmse']),
            'std_rmse': float(best['std_rmse']),
            'mean_coverage': float(best['mean_coverage'])
        }
    
    output = {
        'results_df': results_df,
        'best_config': best_config
    }
    
    # Generate heatmap
    if output_plot and not results_df.empty:
        if verbose:
            print(f"\nGenerating heatmap visualization...")
        
        fig, pivot_df = plot_grid_search_heatmap(
            results_df,
            metric='mean_rmse',
            annotate_values=True,
            highlight_best=True,
            coverage_overlay=coverage_overlay,
            min_ratings_log_scale=max(min_ratings_values) / min(min_ratings_values) > 10,
            show_plot=verbose
        )
        output['figure'] = fig
        output['pivot_df'] = pivot_df
    
    # Generate text summary
    summary_lines = [
        "\nGrid Search Summary",
        "=" * 50,
        f"Parameter Grid: min_ratings × n_neighbors",
        f"Configurations evaluated: {len(results_df)}",
        f"Valid results: {len(valid_results)}",
        "",
    ]
    
    if best_config:
        summary_lines.extend([
            "Best Configuration:",
            f"   min_ratings  = {best_config['min_ratings']}",
            f"   n_neighbors  = {best_config['n_neighbors']}",
            f"   Mean RMSE    = {best_config['mean_rmse']:.3f} ± {best_config['std_rmse']:.3f}",
            f"   Coverage     = {best_config['mean_coverage']:.1f}%",
            "",
            "Top 5 Configurations by RMSE:",
        ])
        
        top_5 = valid_results.nsmallest(5, 'mean_rmse')
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            summary_lines.append(
                f"   {i}. min_ratings={int(row['min_ratings']):3d}, "
                f"n_neighbors={int(row['n_neighbors']):2d} → "
                f"RMSE={row['mean_rmse']:.3f}±{row['std_rmse']:.3f} "
                f"(cov: {row['mean_coverage']:.1f}%)"
            )
    else:
        summary_lines.append("No valid configurations produced results.")
    
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
    
    # Import your model class
    from movie_rec import HybridUserClusterKNNRecommender
    
    # Load data (adjust paths as needed)
    ratings = pd.read_csv('reviews.dat', sep='::', engine='python', header=None,
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    movies = pd.read_csv('movies.dat', sep='::', engine='python', header=None,
                         encoding='latin-1', names=['MovieID', 'Title', 'Genres'])
    
    # Generate full grid search report
    report = generate_grid_search_report(
        model_class=HybridUserClusterKNNRecommender,
        ratings_df=ratings,
        movies_df=movies,
        min_ratings_values=[5, 10, 20, 50, 100],
        n_neighbors_values=[5, 10, 20, 30, 50],
        alpha=0.7,
        n_splits=3,  # Use 3 folds for faster testing; increase to 5 for final report
        coverage_overlay=True,
        verbose=True
    )
    
    # Print best config for easy copying
    if report['best_config']:
        bc = report['best_config']
        print(f"\nGrid search complete. Recommended config:")
        print(f"   HybridUserClusterKNNRecommender(")
        print(f"       min_ratings={bc['min_ratings']},")
        print(f"       n_neighbors={bc['n_neighbors']},")
        print(f"       alpha=0.7")
        print(f"   )  # Expected CV RMSE: {bc['mean_rmse']:.3f}")