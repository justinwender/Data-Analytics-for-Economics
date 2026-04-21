"""clustering_utils.py - Reusable K-Means + PCA clustering pipeline.

Course: ECON 5200, Lab 22.
"""
from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def run_kmeans_pipeline(
    df: pd.DataFrame,
    features: list[str],
    k: int,
    random_state: int = 42,
) -> dict:
    """End-to-end K-Means pipeline.

    1. Pull the requested columns out of ``df``.
    2. Standardize with ``StandardScaler`` (K-Means is distance-based, so
       every feature must be on a comparable scale).
    3. Fit K-Means with ``n_clusters=k`` and the supplied random state.
    4. Return the labels, the fitted scaler, the fitted model, the scaled
       feature matrix, and the silhouette score against ``X_scaled``.

    Args:
        df: DataFrame containing ``features`` as columns.
        features: Names of feature columns to cluster on.
        k: Number of clusters.
        random_state: Seed for ``KMeans`` initialization.

    Returns:
        Dictionary with keys ``labels``, ``scaler``, ``model``, ``X_scaled``,
        ``silhouette``.

    Raises:
        KeyError: If any ``features`` are missing from ``df``.
        ValueError: If ``k < 2``.
    """
    if k < 2:
        raise ValueError("k must be at least 2 for silhouette score.")
    missing = set(features) - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features].values)

    model = KMeans(n_clusters=k, init="k-means++", n_init="auto", random_state=random_state)
    labels = model.fit_predict(X_scaled)
    score = float(silhouette_score(X_scaled, labels))

    return {
        "labels": labels,
        "scaler": scaler,
        "model": model,
        "X_scaled": X_scaled,
        "silhouette": score,
    }


def evaluate_k_range(
    X: np.ndarray,
    k_range: Iterable[int],
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute WCSS (elbow) and silhouette scores across a range of K values."""
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, init="k-means++", n_init="auto", random_state=random_state)
        labels = model.fit_predict(X)
        rows.append({
            "k": k,
            "wcss": float(model.inertia_),
            "silhouette": float(silhouette_score(X, labels)) if k >= 2 else np.nan,
        })
    return pd.DataFrame(rows).set_index("k")


def plot_pca_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    title: str = "K-Means clusters in PCA space",
) -> plt.Figure:
    """Plot a 2D PCA scatter of ``X`` colored by ``labels`` with loading arrows."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=30, alpha=0.75)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(title)

    # Loading arrows (scaled so they fit the plot)
    scale = 0.9 * max(np.abs(X_pca).max(), 1.0)
    for i, name in enumerate(feature_names):
        vx, vy = pca.components_[0, i], pca.components_[1, i]
        ax.arrow(0, 0, vx * scale, vy * scale, color="black", alpha=0.5,
                 length_includes_head=True, head_width=scale * 0.02)
        ax.text(vx * scale * 1.05, vy * scale * 1.05, name, fontsize=8, color="black", alpha=0.8)

    ax.legend(*scatter.legend_elements(), title="Cluster", loc="best", fontsize=8)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 400
    features = [f"x{i}" for i in range(5)]
    # Three blobs in 5D for a self-test
    centers = rng.normal(scale=3, size=(3, 5))
    X_sim = np.vstack([rng.normal(loc=c, scale=1.0, size=(n // 3, 5)) for c in centers])
    df_sim = pd.DataFrame(X_sim, columns=features)

    out = run_kmeans_pipeline(df_sim, features, k=3)
    print("silhouette @ k=3:", out["silhouette"])

    grid = evaluate_k_range(out["X_scaled"], range(2, 8))
    print(grid)

    fig = plot_pca_clusters(out["X_scaled"], out["labels"], features)
    plt.close(fig)
    print("plot_pca_clusters: PASS")
