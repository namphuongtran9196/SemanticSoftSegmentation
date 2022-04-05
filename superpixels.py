from numpy.core.fromnumeric import shape
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import numpy as np
from skimage import measure
from scipy.special import erf
import scipy


def find_dif_pairs(m1, m2, n):
    m_min = np.minimum(m1, m2)
    m_max = np.maximum(m1, m2)
    x = np.where(m1 - m2 == 0, 0, m_min*n + m_max)
    return x
    
def find_neigh(X):
    X = X.astype('int')
    n = np.max(X) + 1
    x1 = find_dif_pairs(X[:, :-1], X[:, 1:], n)
    x2 = find_dif_pairs(X[:-1, :], X[1:, :], n)
    x3 = find_dif_pairs(X[:-1, :-1], X[1:, 1:], n)
    x4 = find_dif_pairs(X[1:, :-1], X[:-1, 1:], n)

    pair_idxs = np.concatenate([x1.reshape(-1), x2.reshape(-1), x3.reshape(-1), x4.reshape(-1)])
    pair_idxs = np.unique(pair_idxs)
    pair_idxs = pair_idxs[pair_idxs != 0]
    c1 = pair_idxs // n
    c2 = pair_idxs % n
    res = np.concatenate([c1.reshape(-1, 1), c2.reshape(-1, 1)], axis=1)
    return res

def label2idx(label_arr):
    return [np.where(label_arr.T.ravel() == i)[0]
            for i in range(1, np.max(label_arr) + 1)]
    
def sigmoid_aff(feat1, feat2, steepness, center):
    aff = np.abs(feat1 - feat2)
    aff = 1 - np.sqrt(aff @ aff.T)
    aff = erf(steepness * (aff - center))
    return aff

def sigmoid_aff_pos(feat1, feat2, steepness, center):
    aff = np.abs(feat1 - feat2)
    aff = 1 - np.sqrt(aff @ aff.T)
    aff = (erf(steepness * (aff - center)) + 1) / 2
    return aff

class Superpixels():
    # TODO: implements adjacent_regions_graph
    def __init__(self, image, spcnt=2500, L=None, N=None, neigh=None):
        
        if L is not None:
            self.labels = L
            self.spcount = N[0][0]
        else:
            self.labels = slic(image, n_segments=spcnt, compactness=1e-20, sigma=1, start_label=1)
            self.spcount = np.unique(self.labels).shape[0]
        
        # g = adjacent_regions_graph(L)

        self.neigh = find_neigh(self.labels)
        s = measure.regionprops(self.labels)
        cent = np.asarray([list(p.centroid) for p in s])
        self.centroid = np.round(cent).astype(np.int)
        
        h, w, _ = image.shape
        py, px = self.centroid[:,0], self.centroid[:, 1]
        sub2ind = px*h + py
        sub2ind = sub2ind.reshape(-1,1)
        self.centroid = np.concatenate((self.centroid, sub2ind), axis=1)
        
        tmp = self.compute_region_means(image)
    
    
    def compute_region_means(self, image):
        h, w, c = image.shape
        image = image.transpose((1, 0, 2)).reshape(-1, c)
        
        reg_means = np.zeros((self.spcount, c))
        idx = label2idx(self.labels)
        
        for i in range(len(idx)):
            reg_means[i, :] = np.mean(image[idx[i], :], axis=0)

        return reg_means
    
    def neighbor_affinities(self, features, erf_steepness=20, erf_center=0.85):
        h, w, _ = features.shape
        N = h * w
        sp_means = self.compute_region_means(features)

        affs = np.zeros((self.neigh.shape[0], 1))
        inds1 = affs.copy()
        inds2 = affs.copy()
        for i in range(self.neigh.shape[0]):
            ind1 = self.neigh[i, 0] - 1
            ind2 = self.neigh[i, 1] - 1
                    
            affs[i, :] = sigmoid_aff(sp_means[ind1, :], sp_means[ind2, :], erf_steepness, erf_center)        
            inds1[i, :] = self.centroid[ind1, 2]
            inds2[i, :] = self.centroid[ind2, 2]

        W = scipy.sparse.coo_matrix((affs.reshape(-1), (inds1.reshape(-1), inds2.reshape(-1))), shape=[N, N])
        W = W.T + W
        return W
    
    def nearby_affinities(self, image, erf_steepness=50, erf_center=0.95, prox_thresh=0.2):
        h, w, _ = image.shape
        N = h * w
        sp_means = self.compute_region_means(image)

        combination_cnt = self.spcount
        combination_cnt = combination_cnt * (combination_cnt - 1) // 2
        affs = np.zeros((combination_cnt, 1))
        inds1 = affs.copy()
        inds2 = affs.copy()
        cnt = 0
        cents = self.centroid[:, :2].astype('float')
    
        cents[:, 0] = (cents[:, 0] + 1.) / h
        cents[:, 1] = (cents[:, 1] + 1.) / w

        for i in range(self.spcount):
            for j in range(i+1, self.spcount):
                try:
                    centdist = cents[i, :2] - cents[j, :2]         
                    centdist = np.sqrt(centdist @ centdist.T)
                    if centdist > prox_thresh:
                        affs[cnt] = 0
                    else:
                        affs[cnt] = sigmoid_aff_pos(sp_means[i, :], sp_means[j, :], erf_steepness, erf_center)
                            
                    inds1[cnt] = self.centroid[i, 2]                
                    inds2[cnt] = self.centroid[j, 2]
                    cnt += 1
                except:
                    pass
        
        W = scipy.sparse.coo_matrix((affs.reshape(-1), (inds1.reshape(-1), inds2.reshape(-1))), shape=[N, N])
        W = W.T + W
        return W         
        
    
        
        
    
        
        