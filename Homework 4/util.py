from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def plot_clusters(Y, mus, covs, labels, K):
    cmap = plt.cm.get_cmap('hsv', K)
    fig, ax = plt.subplots(figsize=(4, 4))
    colors = ['red', 'blue', 'green']
    for j in range(np.unique(labels).shape[0]):
        Y_j = Y[(labels == (j+1))]
        ax.scatter(Y_j[:, 0], Y_j[:, 1], color=colors[j])
    for i in range(K):
        plot_cov_ellipse(cov=covs[i],
                         pos=mus[i],
                         nstd=2,
                         ax=ax,
                         color=cmap(i),
                         alpha=0.1)

    plt.show()