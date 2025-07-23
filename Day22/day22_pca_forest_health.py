# pca_forest_health.py

"""
Forest health monitoring using PCA on hyperspectral data for dimensionality reduction and disease detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification

# Simulate high-dimensional hyperspectral data
# Each sample = a tree region; Features = spectral bands
X, y = make_classification(
    n_samples=300,
    n_features=100,     # Simulate 100 hyperspectral bands
    n_informative=10,   # Only 10 bands are actually useful
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42
)

# Apply PCA to reduce to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the 2D PCA result
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='Set2', edgecolor='k', s=80)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Simulated Hyperspectral Forestry Data')
plt.grid(True)
plt.legend(*scatter.legend_elements(), title="Tree Types")
plt.tight_layout()
plt.show()

# Plot explained variance ratio (Scree Plot)
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.grid(True)
plt.tight_layout()
plt.show()
