import numpy as np
from matplotlib import pyplot as plt


def plot_raw_data(X, labels):
    plt.scatter(X[:, 0], X[:, 1])
    for i, label in enumerate(labels):
        plt.text(X[i, 0] - 0.1, X[i, 1] + 0.1, label, fontsize=10, ha='right', va='bottom')

    plt.show()


def plot_clusters(X, labels, cluster_labels, cluster_centers=None):
    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.get_cmap('tab10', len(unique_clusters))

    for i, cluster in enumerate(unique_clusters):
        cluster_points = X[cluster_labels == cluster]
        plt.scatter(
            cluster_points[:, 0], cluster_points[:, 1],
            label=f'Cluster {cluster}', color=colors(i), alpha=0.6
        )

    if cluster_centers is not None:
        plt.scatter(
            cluster_centers[:, 0], cluster_centers[:, 1],
            c=[colors(i) for i in unique_clusters], marker='D', s=50, label='Centroids'
        )

    for i, label in enumerate(labels):
        plt.text(X[i, 0] - 0.1, X[i, 1] + 0.1, label, fontsize=10, ha='right', va='bottom')

    plt.legend()
    plt.show()
