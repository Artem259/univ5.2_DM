import numpy as np
import scipy
from matplotlib import pyplot as plt


def plot_raw_data(X, labels):
    plt.scatter(X[:, 0], X[:, 1])
    for i, label in enumerate(labels):
        plt.text(X[i, 0] - 0.1, X[i, 1] + 0.1, label, fontsize=10, ha='right', va='bottom')

    plt.show()


def plot_clusters(
    X,
    cluster_labels,
    labels=None,
    special_indices=None,
    cluster_centers=None,
    text_shift=0.1,
    fontsize=10,
    figsize=(6.4, 4.8),
    title=None,
    xlabel=None,
    ylabel=None,
):
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

    if labels is not None and special_indices is None:
        for i, label in enumerate(labels):
            plt.text(
                X[i, 0] - text_shift, X[i, 1] + text_shift, label,
                fontsize=fontsize, ha='right', va='bottom'
            )

    if special_indices is not None:
        plt.scatter(
            X[special_indices, 0], X[special_indices, 1],
            c='orange', marker='s', s=50, label='Special'
        )
        for i in special_indices:
            plt.text(
                X[i, 0] - text_shift, X[i, 1] + text_shift, labels[i],
                fontsize=fontsize, ha='right', va='bottom'
            )

    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def plot_dendrogram(children, distances, labels, title, figsize=(6.4, 4.8)):
    linkage_matrix = np.column_stack([children, distances, np.zeros(len(children))]).astype(float)

    plt.figure(figsize=figsize)
    scipy.cluster.hierarchy.dendrogram(linkage_matrix, labels=labels, count_sort='descending')
    plt.title(title)
    plt.ylabel("Distance")
    plt.show()
