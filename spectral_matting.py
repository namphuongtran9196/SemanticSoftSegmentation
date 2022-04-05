import numpy as np
from numpy.lib.function_base import cov
import numpy.matlib
import skimage
import scipy.ndimage
from scipy.misc import imread
import matplotlib.pyplot as plt

## Checked
def local_RGB_normal_distributions(image, window_radius=1, epsilon=1e-8):
    h, w, c = image.shape
    N = h * w
    window_size = 2 * window_radius + 1

    mean_image = scipy.ndimage.generic_filter(image, np.mean, size=(window_size, window_size, 1), mode='nearest')

    covar_mat = np.zeros((3, 3, N))
    for r in range(c):
        for s in range(r, c):
            temp = scipy.ndimage.generic_filter(image[:, :, r] * image[:, :, s], np.mean, size=(window_size, window_size), mode='nearest') - mean_image[:, :, r] * mean_image[:, :, s]
            covar_mat[r, s, :] = temp.T.reshape((-1))

    for i in range(c):
        covar_mat[i, i, :] += epsilon

    for r in range(1, c):
        for s in range(r):
            covar_mat[r, s, :] = covar_mat[s, r, :]
            
    return mean_image, covar_mat


## Checked
def matting_affinity(image, in_map=None, window_radius=1, epsilon=1e-7):
    
    window_size = 2 * window_radius + 1
    neigh_size = window_size ** 2
    h, w, c = image.shape
    if in_map is None:
        in_map = np.full((h, w), True)
    N = h * w
    epsilon = epsilon / neigh_size

    mean_image, covar_mat = local_RGB_normal_distributions(image, window_radius, epsilon)
    
    # Determine pixels and their local neighbors
    indices = np.array(range(h * w)).reshape(w, h).transpose((1, 0))
    neigh_ind = skimage.util.view_as_windows(indices, window_size)
    s1, s2, s3, s4 = neigh_ind.shape
    neigh_ind = neigh_ind.transpose((1, 0, 3, 2)).reshape(s1 * s2, s3 * s4)
    
    in_map = in_map[window_radius:-window_radius, window_radius:-window_radius]
    # neigh_ind = neigh_ind[in_map].reshape(-1, neigh_size)
    neigh_ind = neigh_ind[in_map.transpose(1, 0).reshape(-1), :]    
    in_ind = neigh_ind[:, neigh_size // 2]    
    pix_cnt = in_ind.shape[0]
    
    # Prepare in & out data
    # image = image.reshape(-1, 3)
    image = image.transpose((1, 0, 2)).reshape(-1, 3)
    
    # mean_image = mean_image.reshape(-1, 3)
    mean_image = mean_image.transpose((1, 0, 2)).reshape(-1, 3)
  
    flow_rows = np.zeros((neigh_size, neigh_size, pix_cnt))
    flow_cols = np.zeros((neigh_size, neigh_size, pix_cnt))
    flows = np.zeros((neigh_size, neigh_size, pix_cnt))

    # Compute matting affinity
    for i in range(pix_cnt):
        neighs = neigh_ind[i:i+1, :]    
        shifted_win_colors = image[neighs[0],:] - np.matlib.repmat(mean_image[in_ind[i:i+1]], neighs.shape[1], 1)
        #shifted_win_colors = image[neighs] - np.matlib.repmat(mean_image[in_ind[i]], neighs.shape[0], 1)
        
        flows[:, :, i] = shifted_win_colors @ np.linalg.solve(covar_mat[:, :, in_ind[i]], shifted_win_colors.T)
    
        neighs = np.matlib.repmat(neighs, neighs.shape[1], 1)        
        flow_rows[:, :, i] = neighs
        flow_cols[:, :, i] = neighs.T
    
    flows = (flows + 1) / neigh_size
    
    W = scipy.sparse.coo_matrix((flows.transpose(2, 1, 0).reshape(-1), (flow_rows.transpose(2, 1, 0).reshape(-1), flow_cols.transpose(2, 1, 0).reshape(-1))), shape=[N, N])
    W = (W + W.T) / 2
            
    return W


if __name__ == '__main__':
    indices = np.array(range(6 * 6)).reshape(6, 6)
    print(indices)
    neigh_ind = skimage.util.view_as_windows(indices, 3)
    neigh_ind = neigh_ind.reshape(16, 9)
    print(neigh_ind)
    # in_map = in_map[window_radius:-window_radius, window_radius:-window_radius]
    # neigh_ind = neigh_ind[in_map].reshape(-1, neigh_size)
    # in_ind = neigh_ind[:, neigh_size // 2]
    # pix_cnt = in_ind.shape[0]
