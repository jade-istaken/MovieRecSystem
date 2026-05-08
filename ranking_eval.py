"""
ranking_eval.py
Evaluation utilities for top-N recommendation ranking.
All code is function-wrapped; no top-level execution.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

DEFAULT_DPI = 300
DEFAULT_FIGSIZE = (10, 6)

# Core Metrics

def define_relevance(rating: float, threshold: float = 4.0, 
                     relative: bool = False, user_mean: Optional[float] = None) -> bool:
    """Determine if a rating counts as 'relevant' for recommendation evaluation."""
    if relative:
        if user_mean is None:
            raise ValueError("user_mean required when relative=True")
        return rating >= (user_mean + threshold)
    return rating >= threshold


def compute_dcg(relevances: np.ndarray, k: Optional[int] = None) -> float:
    """Discounted Cumulative Gain @ K."""
    if k is not None:
        relevances = relevances[:k]
    if len(relevances) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(relevances / discounts))


def compute_ndcg(relevances: np.ndarray, k: Optional[int] = None) -> float:
    """Normalized DCG @ K. Returns NaN if no relevant items exist."""
    dcg = compute_dcg(relevances, k)
    ideal_relevances = np.sort(relevances)[::-1]
    idcg = compute_dcg(ideal_relevances, k)
    if idcg == 0:
        return np.nan
    
    # Sanity check: NDCG must be in [0, 1]
    ndcg = dcg / idcg
    if ndcg < 0 or ndcg > 1:
        warnings.warn(f"NDCG out of bounds: {ndcg}. Clipping to [0,1].")
        return float(np.clip(ndcg, 0, 1))
    
    return float(ndcg)
    return float(dcg / idcg)


def compute_ap(relevances: np.ndarray, k: Optional[int] = None) -> float:
    """Average Precision @ K. Returns NaN if no relevant items exist."""
    if k is not None:
        relevances = relevances[:k]
    if not np.any(relevances):
        return np.nan
    precisions = []
    relevant_count = 0
    for i, rel in enumerate(relevances):
        if rel:
            relevant_count += 1
            precisions.append(relevant_count / (i + 1))
    return float(np.mean(precisions)) if precisions else np.nan


def evaluate_user_recommendations(
    user_id: int,
    train_ratings: pd.DataFrame,
    test_relevant_items: List[int],
    recommended_items: List[int],
    k: int = 10
) -> Dict[str, Union[int, float]]:
    """Compute ranking metrics for a single user's top-K recommendations."""
    # Filter to top-K
    recs_k = recommended_items[:k]
    if not recs_k or not test_relevant_items:
        return {'hit': 0, 'precision': np.nan, 'recall': np.nan, 
                'ndcg': np.nan, 'ap': np.nan, 'n_relevant': len(test_relevant_items)}
    
    # Binary relevance vector
    test_relevant_set = set(int(m) for m in test_relevant_items)  # Ensure int type
    relevances = np.array([1 if int(m) in test_relevant_set else 0 for m in recs_k])
    total_relevant = len(set(test_relevant_items))
    
    return {
        'user_id': user_id,
        'hit': int(np.any(relevances)),  # Always 0 or 1
        'precision': float(np.mean(relevances)),
        'recall': float(np.sum(relevances) / total_relevant) if total_relevant > 0 else 0.0,
        'ndcg': compute_ndcg(relevances),
        'ap': compute_ap(relevances),
        'n_relevant': total_relevant
    }


#Holdout Split

def create_holdout_split_strict(
    ratings_df: pd.DataFrame,
    test_ratio: float = 0.2,
    min_test_items: int = 1,
    relevance_threshold: float = 4.0,
    relative_relevance: bool = False,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[int, List[int]], Dict[int, float]]:
    """
    Create user-wise hold-out split with GUARANTEED train/test separation.
    Test-relevant items are COMPLETELY excluded from training data.
    """
    np.random.seed(random_state)
    user_means = ratings_df.groupby('UserID')['Rating'].mean().to_dict() if relative_relevance else {}
    
    # Identify relevant items per user
    user_relevant = defaultdict(list)
    for _, row in ratings_df.iterrows():
        uid, mid, rating = int(row['UserID']), int(row['MovieID']), float(row['Rating'])
        umean = user_means.get(uid) if relative_relevance else None
        if define_relevance(rating, relevance_threshold, relative_relevance, umean):
            user_relevant[uid].append(mid)
    
    # Split: test items excluded from training
    train_rows = []
    test_relevant = {}
    test_items_set = set()
    
    for uid, relevant_items in user_relevant.items():
        if len(relevant_items) < min_test_items + 1:
            train_rows.extend([(uid, mid) for mid in relevant_items])
            continue
        
        n_test = max(min_test_items, int(len(relevant_items) * test_ratio))
        test_items = np.random.choice(relevant_items, size=n_test, replace=False).tolist()
        train_items = [m for m in relevant_items if m not in test_items]
        
        test_relevant[uid] = test_items
        test_items_set.update(test_items)
        train_rows.extend([(uid, mid) for mid in train_items])
        
        # Add non-relevant items ONLY if not in test set
        user_all = ratings_df[ratings_df['UserID'] == uid]['MovieID'].unique()
        for mid in user_all:
            if mid not in relevant_items and mid not in test_items_set:
                train_rows.append((uid, mid))
    
    # Build training DataFrame via fast merge
    train_df = ratings_df.merge(
        pd.DataFrame(train_rows, columns=['UserID', 'MovieID']),
        on=['UserID', 'MovieID'],
        how='inner'
    ).copy()
    
    return train_df, test_relevant, user_means


#CV Eval

def evaluate_ranking_cv(
    model, ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
    k_values: List[int] = [5, 10, 20], n_folds: int = 5,
    test_ratio: float = 0.2, relevance_threshold: float = 4.0,
    relative_relevance: bool = False, min_train_ratings: int = 5,
    random_state: int = 42, verbose: bool = True,
    cold_threshold: int = 20, longtail_percentile: float = 50.0
) -> pd.DataFrame:
    """
    Cross-validated ranking evaluation with long-tail coverage and cold/warm user analysis.
    Backward-compatible: existing calls work unchanged.
    """
    from sklearn.model_selection import KFold
    import numpy as np
    import pandas as pd
    from collections import defaultdict
    
    np.random.seed(random_state)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    results = []
    
    user_counts = ratings_df.groupby('UserID').size()
    eligible_users = user_counts[user_counts >= min_train_ratings + 2].index.tolist()
    
    for fold_idx, (train_user_idx, test_user_idx) in enumerate(kf.split(eligible_users)):
        train_users = [eligible_users[i] for i in train_user_idx]
        test_users = [eligible_users[i] for i in test_user_idx]
        
        if verbose:
            print(f"\nFold {fold_idx+1}/{n_folds}: {len(train_users)} train, {len(test_users)} test users")
        
        train_df = ratings_df[ratings_df['UserID'].isin(train_users)].copy()
        test_ratings_subset = ratings_df[ratings_df['UserID'].isin(test_users)].copy()
        
        train_df_holdout, test_relevant, _ = create_holdout_split_strict(
            test_ratings_subset, test_ratio=test_ratio, min_test_items=1,
            relevance_threshold=relevance_threshold, relative_relevance=relative_relevance,
            random_state=random_state + fold_idx
        )
        
        full_train = pd.concat([train_df, train_df_holdout], ignore_index=True)
        
        try:
            model.fit(full_train, movies_df)
        except Exception as e:
            if verbose: print(f"  Fit failed: {e}")
            continue
        
        # Precompute fold-level stats
        pop_counts = full_train.groupby('MovieID').size()
        lt_threshold = np.percentile(pop_counts.values, longtail_percentile) if len(pop_counts) > 0 else 0
        longtail_movies = set(pop_counts[pop_counts <= lt_threshold].index)
        user_activity = full_train.groupby('UserID').size().to_dict()
        
        n_eligible = sum(1 for uid in test_users if uid in test_relevant and len(test_relevant[uid]) > 0)
        n_served = 0
        fold_user_metrics = []
        
        for uid in test_users:
            if uid not in test_relevant or not test_relevant[uid]:
                continue
                
            try:
                recs_raw = model.recommend(uid, n_rec=max(k_values) * 2)
                recommended_ids = [int(r[0]) if isinstance(r, (list, tuple)) else int(r) for r in recs_raw]
            except Exception:
                continue
                
            user_train_items = set(full_train[full_train['UserID'] == uid]['MovieID'].values)
            unseen_recs = [m for m in recommended_ids if m not in user_train_items][:max(k_values)]
            
            if verbose and uid in test_users[:5]:  # Sample first 5 users
                print(f"  User {uid}: longtail_movies={len(longtail_movies)}/{len(pop_counts)} "
                      f"({len(longtail_movies)/len(pop_counts)*100:.1f}%)")
                print(f"  Recommendations: {unseen_recs[:10]}")
                lt_recs = [m for m in unseen_recs if m in longtail_movies]
                print(f"  Long-tail recs: {len(lt_recs)}/{len(unseen_recs)} = {len(lt_recs)/len(unseen_recs)*100:.1f}%")
            
            if unseen_recs:
                n_served += 1
                
            activity = user_activity.get(uid, 0)
            group = 'cold' if activity < cold_threshold else 'warm'
            lt_count = sum(1 for m in unseen_recs if m in longtail_movies)
            lt_ratio = lt_count / len(unseen_recs) if unseen_recs else 0.0
            
            for k in k_values:
                metrics = evaluate_user_recommendations(
                    user_id=uid, train_ratings=full_train,
                    test_relevant_items=test_relevant[uid],
                    recommended_items=unseen_recs, k=k
                )
                metrics.update({
                    'k': k, 'longtail_ratio': lt_ratio, 
                    'activity_group': group, 'n_training_ratings': activity
                })
                fold_user_metrics.append(metrics)
        
        if not fold_user_metrics:
            continue
            
        metrics_df = pd.DataFrame(fold_user_metrics)
        
        for k in k_values:
            k_data = metrics_df[metrics_df['k'] == k]
            row = {'fold': fold_idx + 1, 'k': k}
            
            # Standard metrics (backward compatible)
            for prefix in ['hit', 'precision', 'recall', 'ndcg', 'ap']:
                vals = k_data[prefix].values
                has_relevant = k_data['n_relevant'].values > 0
                vals_filled = np.where(has_relevant & np.isnan(vals), 0.0, vals)
                row[f'{prefix}'] = float(np.nanmean(vals_filled))
                row[f'{prefix}_std'] = float(np.nanstd(vals_filled))
                row[f'{prefix}_valid_pct'] = float((has_relevant).mean() * 100)
                
            row['n_users_evaluated'] = len(k_data)
            row['coverage'] = n_served / n_eligible if n_eligible > 0 else 0.0
            
            # New analysis columns
            row['longtail_coverage'] = float(k_data['longtail_ratio'].mean())
            
            for group_name in ['cold', 'warm']:
                g_data = k_data[k_data['activity_group'] == group_name]
                if len(g_data) == 0:
                    row[f'hit_{group_name}'] = np.nan
                    row[f'ndcg_{group_name}'] = np.nan
                    continue
                    
                row[f'hit_{group_name}'] = float(g_data['hit'].mean())
                g_ndcg = g_data['ndcg'].values
                g_has_rel = g_data['n_relevant'].values > 0
                g_ndcg_filled = np.where(g_has_rel & np.isnan(g_ndcg), 0.0, g_ndcg)
                row[f'ndcg_{group_name}'] = float(np.nanmean(g_ndcg_filled))
                
            results.append(row)
            
    return pd.DataFrame(results)


#Visualization

def plot_ranking_metrics(
    results_df: pd.DataFrame,
    metrics: List[str] = ['ndcg', 'precision', 'recall'],
    output_path_png: Optional[str] = 'ranking_metrics.png',
    output_path_pdf: Optional[str] = 'ranking_metrics.pdf',
    figsize: tuple = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
    show_plot: bool = True
) -> 'plt.Figure':
    """Line plot of ranking metrics vs. K with error bars."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)
    
    # Aggregate across folds
    agg_df = results_df.groupby('k')[[m for m in metrics]].agg(['mean', 'std']).reset_index()
    agg_df.columns = ['k'] + [f'{m}_{stat}' for m in metrics for stat in ['mean', 'std']]
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
    
    for i, metric in enumerate(metrics):
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        if mean_col not in agg_df.columns: continue
        ax.errorbar(agg_df['k'], agg_df[mean_col], yerr=agg_df[std_col],
                    label=metric.upper(), marker='o', color=colors[i], linewidth=2, capsize=4)
    
    ax.set_xlabel('K (Number of Recommendations)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=11, fontweight='bold')
    ax.set_title('Top-N Recommendation Performance', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xticks(sorted(agg_df['k'].unique()))
    plt.tight_layout()
    
    if output_path_png: fig.savefig(output_path_png, bbox_inches='tight', dpi=dpi)
    if output_path_pdf: fig.savefig(output_path_pdf, bbox_inches='tight')
    if show_plot: plt.show()
    return fig


def compare_ranking_models(
    models: Dict[str, any],
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    k_values: List[int] = [5, 10, 20],
    n_folds: int = 3,
    test_ratio: float = 0.2,
    relevance_threshold: float = 4.0,
    random_state: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """Compare multiple recommender models on ranking metrics."""
    all_results = []
    for name, model in models.items():
        if verbose: print(f"\n Evaluating model: {name}")
        results = evaluate_ranking_cv(
            model, ratings_df, movies_df, k_values=k_values, n_folds=n_folds,
            test_ratio=test_ratio, relevance_threshold=relevance_threshold,
            random_state=random_state, verbose=verbose
        )
        if not results.empty:
            results['model'] = name
            all_results.append(results)
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'ndcg',
    output_path_png: Optional[str] = 'model_comparison.png',
    output_path_pdf: Optional[str] = 'model_comparison.pdf',
    figsize: tuple = (10, 6),
    dpi: int = DEFAULT_DPI,
    show_plot: bool = True
) -> 'plt.Figure':
    """Grouped bar chart comparing models on a single ranking metric."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)
    agg = comparison_df.groupby(['model', 'k'])[metric].agg(['mean', 'std']).reset_index()
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    models = agg['model'].unique()
    k_values = sorted(agg['k'].unique())
    x = np.arange(len(k_values))
    width = 0.8 / len(models)
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        m_data = agg[agg['model'] == model]
        means = [m_data[m_data['k'] == k]['mean'].values[0] if not m_data[m_data['k'] == k].empty else np.nan for k in k_values]
        stds = [m_data[m_data['k'] == k]['std'].values[0] if not m_data[m_data['k'] == k].empty else 0 for k in k_values]
        ax.bar(x + i*width - (len(models)-1)*width/2, means, width, yerr=stds,
               label=model, color=colors[i], capsize=4, alpha=0.9)
    
    ax.set_xlabel('K', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'{metric.upper()}', fontsize=11, fontweight='bold')
    ax.set_title(f'Model Comparison: {metric.upper()}@K', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(k_values)
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    
    if output_path_png: fig.savefig(output_path_png, bbox_inches='tight', dpi=dpi)
    if output_path_pdf: fig.savefig(output_path_pdf, bbox_inches='tight')
    if show_plot: plt.show()
    return fig

def debug_holdout_integrity(ratings_df, test_relevant, train_df, n_users=5):
    """Verify test-relevant items are NOT in training data."""
    print(f"\n Debugging holdout integrity (sample {n_users} users)...")
    
    issues_found = 0
    for uid in list(test_relevant.keys())[:n_users]:
        if not test_relevant[uid]:
            continue
            
        # Get training items for this user
        train_items = set(train_df[train_df['UserID'] == uid]['MovieID'].astype(int).values)
        test_items = set(int(m) for m in test_relevant[uid])
        
        # Check for leakage
        leaked = test_items & train_items
        if leaked:
            print(f" User {uid}: {len(leaked)} test items found in training: {list(leaked)[:3]}")
            issues_found += len(leaked)
        
        # Check relevance definition
        user_ratings = ratings_df[ratings_df['UserID'] == uid]
        relevant_by_threshold = user_ratings[user_ratings['Rating'] >= 4.0]['MovieID'].astype(int).tolist()
        missing_from_test = [m for m in relevant_by_threshold if m not in test_items and m not in train_items]
        
        if missing_from_test:
            print(f" User {uid}: {len(missing_from_test)} relevant items not in train OR test (dropped?)")
    
    if issues_found == 0:
        print("No leakage detected in sample")
    else:
        print(f"Found {issues_found} leaked items total in sample")
    
    return issues_found == 0

def print_analysis_summary(results_df, k=10, cold_threshold=20):
    """Print a clean table of ranking metrics + long-tail + cold/warm split."""
    if results_df.empty:
        print("No results to display.")
        return
        
    agg = results_df[results_df['k'] == k].mean()
    
    print(f"\nRanking Analysis Summary (K={k}, Cold < {cold_threshold} ratings)")
    print("-" * 60)
    print(f"Overall Hit Rate@{k}:      {agg['hit']:.3f}")
    print(f"Overall NDCG@{k}:          {agg['ndcg']:.3f}")
    print(f"Long-Tail Coverage:        {agg['longtail_coverage']*100:.1f}%")
    print(f"Model Coverage:            {agg['coverage']*100:.1f}%")
    print("-" * 60)
    print(f"Cold Users Hit Rate:       {agg.get('hit_cold', 0):.3f} (NDCG: {agg.get('ndcg_cold', 0):.3f})")
    print(f"Warm Users Hit Rate:       {agg.get('hit_warm', 0):.3f} (NDCG: {agg.get('ndcg_warm', 0):.3f})")
    
    # Ratio interpretation
    if pd.notna(agg.get('hit_cold')) and pd.notna(agg.get('hit_warm')) and agg['hit_warm'] > 0:
        ratio = agg['hit_cold'] / agg['hit_warm']
        print(f"\nCold/Warm Performance Ratio: {ratio:.2f}x")
        if ratio > 0.7: print("   Model handles sparse users well")
        elif ratio > 0.4: print("   Moderate cold-start degradation (expected)")
        else: print("   Severe cold-start drop (needs fallback)")
        
        
def print_ranking_comparison_summary(
    comparison_df, k=10, cold_threshold=20, longtail_percentile=50.0, round_decimals=3
):
    """Print a clean, aligned comparison table for ranking metrics."""
    import pandas as pd
    import numpy as np
    
    if comparison_df.empty:
        print("No results to display.")
        return
        
    k_data = comparison_df[comparison_df['k'] == k].copy()
    if k_data.empty:
        print(f"No results found for K={k}. Available K: {sorted(comparison_df['k'].unique())}")
        return
        
    # Aggregate across folds
    agg_dict = {
        'ndcg': ['mean', 'std'], 'hit': ['mean', 'std'], 'precision': ['mean', 'std'],
        'coverage': ['mean', 'std'], 'longtail_coverage': ['mean', 'std'],
        'hit_cold': ['mean', 'std'], 'ndcg_cold': ['mean', 'std'],
        'hit_warm': ['mean', 'std'], 'ndcg_warm': ['mean', 'std'],
        'n_users_evaluated': 'mean'
    }
    summary = k_data.groupby('model').agg(agg_dict)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.round(round_decimals)
    
    print(f"\nRanking Comparison Summary (K={k})")
    print(f"   Cold users: < {cold_threshold} training ratings | Long-tail: bottom {longtail_percentile:.0f}% by popularity")
    print("=" * 140)
    
    headers = ["Model", "NDCG@K", "Hit@K", "Prec@K", "Coverage", "Long-Tail%",
               "Cold Hit", "Cold NDCG", "Warm Hit", "Warm NDCG", "N Users"]
    print(f"{headers[0]:<24} " + " ".join(f"{h:>12}" for h in headers[1:]))
    print("-" * 140)
    
    # Safe formatter
    def fmt(mean_col, std_col, row, is_pct=False):
        m, s = row[mean_col], row[std_col]
        if pd.isna(m): return "     -      "
        val_m = m * 100 if is_pct else m
        val_s = s * 100 if is_pct else s
        return f"{val_m:.{round_decimals}f}±{val_s:.{round_decimals}f}"
        
    for model_name, row in summary.iterrows():
        ndcg = fmt('ndcg_mean', 'ndcg_std', row)
        hit  = fmt('hit_mean', 'hit_std', row)
        prec = fmt('precision_mean', 'precision_std', row)
        cov  = fmt('coverage_mean', 'coverage_std', row, is_pct=True)
        lt   = fmt('longtail_coverage_mean', 'longtail_coverage_std', row, is_pct=True)
        ch   = fmt('hit_cold_mean', 'hit_cold_std', row)
        cn   = fmt('ndcg_cold_mean', 'ndcg_cold_std', row)
        wh   = fmt('hit_warm_mean', 'hit_warm_std', row)
        wn   = fmt('ndcg_warm_mean', 'ndcg_warm_std', row)
        ne   = f"{int(row['n_users_evaluated_mean']):>6}"
        
        print(f"{str(model_name):<24} {ndcg:>12} {hit:>12} {prec:>12} {cov:>12} {lt:>12} {ch:>12} {cn:>12} {wh:>12} {wn:>12} {ne}")
        
    print("=" * 140)
    
    # Interpretation guide
    print("\nQuick Interpretation Guide:")
    print("   • NDCG/Hit/Prec: Higher = better ranking quality")
    print("   • Coverage: % of eligible users who received recommendations")
    print("   • Long-Tail%: % of recommendations from niche/less-popular movies")
    print("   • Cold/Warm: Performance split by user activity level")
    print("   • ± values: Standard deviation across CV folds")
    
    # Highlight best performers
    print("\nBest per Metric (ignoring std):")
    for metric, label in [('ndcg_mean', 'NDCG'), ('hit_mean', 'Hit Rate'), 
                          ('longtail_coverage_mean', 'Long-Tail'), ('coverage_mean', 'Coverage')]:
        if metric in summary.columns:
            valid = summary[summary[metric].notna()]
            if not valid.empty:
                best_model = valid[metric].idxmax()
                best_val = valid.loc[best_model, metric]
                suffix = '%' if 'coverage' in metric or 'longtail' in metric else ''
                print(f"   • {label}: {best_model} ({best_val*100 if suffix else best_val:.{round_decimals}f}{suffix})")
# ============================================================================
# SANITY & DEBUG UTILITIES
# ============================================================================

def sanity_check_recommendations(model, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, n_users: int = 10, verbose: bool = True) -> bool:
    """Check if recommendations ever include training items."""
    user_counts = ratings_df.groupby('UserID').size()
    sample_users = user_counts[user_counts >= 20].sample(min(n_users, len(user_counts)), random_state=42).index.tolist()
    model.fit(ratings_df, movies_df)
    
    leaks_found = 0
    for uid in sample_users:
        train_items = set(ratings_df[ratings_df['UserID'] == uid]['MovieID'].values)
        recs = model.recommend(uid, n_rec=20)
        rec_ids = [int(r[0]) if isinstance(r, (list, tuple)) else int(r) for r in recs]
        leaked = [m for m in rec_ids if m in train_items]
        if leaked:
            leaks_found += len(leaked)
    if leaks_found == 0 and verbose: print("No training items found in recommendations")
    return leaks_found == 0
