import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load MNIST dataset
def load_mnist_images_raw(filepath):
    with open(filepath, 'rb') as f:
        # Read 16-byte header
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # Read pixel data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
        return images

# K-means clustering implementation
def kmeans(X, k, max_iters=100, seed=0):
    rng = np.random.default_rng(seed)

    # Init: k distinct random training images as starting centroids
    centroids = X[rng.choice(len(X), size=k, replace=False)].copy()
    labels = None
    x_sq = (X ** 2).sum(axis=1, keepdims=True)  # ||x||^2, computed once

    for it in range(max_iters):
        # --- Assignment step (vectorized squared Euclidean distance) ---
        c_sq = (centroids ** 2).sum(axis=1)                  # (k,)
        dists = x_sq - 2 * X @ centroids.T + c_sq            # (60000, k)
        new_labels = dists.argmin(axis=1)

        # Converged: no point changed cluster
        if labels is not None and np.array_equal(new_labels, labels):
            print(f"k={k}: converged after {it} iterations")
            break
        labels = new_labels

        # --- Update step: each centroid becomes the mean of its points ---
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                centroids[j] = members.mean(axis=0)
            else:  # empty cluster -> reseed with a random point
                centroids[j] = X[rng.integers(len(X))]

    return centroids, labels


def show_centroids(centroids, title):
    k = len(centroids)
    cols = min(k, 10)
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.7), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, c in enumerate(centroids):
        axes.ravel()[i].imshow(c.reshape(28, 28), cmap="gray")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X_train = load_mnist_images_raw("mnist_data/train-images.idx3-ubyte")
    print(X_train.shape)  # (60000, 784)

    for k in (2, 5, 20, 30):
        centroids, labels = kmeans(X_train, k)
        show_centroids(centroids, f"K-means with k={k}")
