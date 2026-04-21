# Lab 22 — Clustering Economies: K-Means, PCA, UMAP

## Objective
Diagnose and repair a broken K-Means pipeline on World Bank development indicators, extend the corrected workflow to a synthetic customer-segmentation problem, compare PCA and UMAP for projection, and ship the workflow as a reusable module.

## Methodology
- Pulled ten World Bank WDI indicators via `wbgapi` covering income, health, education, inequality, environment, trade, labor, and urbanization. Filtered to countries with at least seven non-null values and median-imputed the rest.
- Part 1 (diagnose): identified four planted bugs in the starter pipeline: clustering on raw un-scaled features (letting GDP per capita dominate distances), calling `KMeans(k=4)` instead of `n_clusters=4`, running PCA before `StandardScaler`, and omitting `random_state`.
- Part 2 (fix): standardized the feature matrix, fit `KMeans(n_clusters=4, random_state=42)`, projected with PCA after scaling, and verified the checkpoints (standardized mean ~ 0, std ~ 1; PC1 variance in the 35 to 50 percent range; silhouette in the 0.15 to 0.40 range; balanced cluster sizes).
- Part 3 (extend): generated 2,000 synthetic customers from four latent segments with `make_blobs`, clustered with K-Means, and compared PCA and UMAP projections side-by-side. Used the Hungarian algorithm (`linear_sum_assignment`) to remap cluster labels and computed accuracy against the ground-truth segmentation.
- Part 4 (module): packaged `run_kmeans_pipeline`, `evaluate_k_range`, and `plot_pca_clusters` into `src/clustering_utils.py` with type hints, docstrings, and a `__main__` smoke test.
- Challenge: fit `AgglomerativeClustering(linkage='ward')` on the WDI data, plotted the dendrogram, cross-tabulated labels against K-Means, and compared silhouette scores.

## Key Findings
- The corrected pipeline produces four interpretable country clusters: a high-income OECD block, a middle-income emerging-market block, a lower-middle-income block, and a low-income fragile-state block. PC1 loads on income and health indicators, PC2 on trade openness and urbanization.
- K-Means recovers the ground-truth customer segmentation with roughly 90 to 95 percent accuracy after Hungarian matching. UMAP separates the segments into cleanly disjoint clouds even in 2D, while PCA leaves the two middle segments partially overlapped.
- Elbow and silhouette plots on the WDI data both favor K = 3 or K = 4, consistent with the usual "development tiers" story.
- Agglomerative clustering with Ward linkage agrees with K-Means on roughly 80 percent of country assignments; the disagreements are concentrated in middle-income countries near cluster boundaries.
- UMAP's `n_neighbors=15, min_dist=0.1` defaults preserve local neighborhood structure that PCA flattens; for customer segmentation, UMAP is the right default for visual confirmation of cluster separability, while PCA remains the right default for interpretable loadings.

## Reproducing
1. `pip install -r requirements.txt`.
2. Run `lab-ch22-diagnostic-1.ipynb` top to bottom.

## Stack
Python 3.13, pandas, numpy, scikit-learn, wbgapi, umap-learn, scipy, matplotlib, seaborn.
