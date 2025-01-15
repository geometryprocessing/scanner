import matplotlib.pyplot as plt
import numpy as np

def compute_confusion_matrix(pattern, filter=True):
    n = pattern.shape[0]
    mtx = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        mtx[i, :] = np.linalg.norm(pattern - pattern[i, :], axis=1)

    m = np.max(mtx)
    m = 1 - mtx / m
    per = 95

    if filter:
        ii = np.arange(pattern.shape[0])
        i, j = np.meshgrid(ii, ii)
        idx = np.abs(i - j) < 0.05 * pattern.shape[0]
        idx |= m < 0.5
        # m[idx] = 0

        avg, pr, mx = np.average(m[~idx]), np.percentile(m[~idx], per), np.max(m[~idx])
        # print("Average:", avg, "Max %d %%:" % per, pr)
    else:
        avg, pr, mx = np.average(m), np.percentile(m, per), np.max(m)
        # print("Average:", np.average(m), "Max: %d %%" % per, np.percentile(m, per))

    print("Average:", avg, "Max:", mx, "Max %d %%:" % per, pr)

    return m, (avg, pr, mx)

def plot_confusion_matrix(matrix,
                          cmap: str='viridis',
                          figsize: tuple=(16,12),
                          filename: str = None):
    plt.figure(figsize)

    plt.imshow(matrix, vmin=0, vmax=1, cmap=cmap)
    plt.axis('off')
    if filename:
        plt.savefig(filename, transparent=True, bbox_inches='tight')
    plt.show()