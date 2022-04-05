import numpy as np
import scipy


def affinity_matrix_to_laplacian(aff, normalize=False):
    N = aff.shape[0]
    if normalize:
        aa = np.sum(aff, axis=1)
        D = scipy.sparse.spdiags(aa, 0, N, N)
        aa = np.sqrt(1 / aa)
        Dl2 = scipy.sparse.spdiags(aa, 0, N, N)
        lap = np.matmul(np.matmul(Dl2, (D - aff)), Dl2)
    else:
        lap = scipy.sparse.spdiags(np.sum(aff, axis=1).reshape([-1]), 0, N, N) - aff
        # lap = scipy.sparse.diags(np.sum(aff, axis=1).reshape(-1)) - aff
        
    return lap
