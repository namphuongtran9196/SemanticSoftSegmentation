from image_gradient import image_gradient
from preprocess_features import preprocess_features
from spectral_matting import matting_affinity
from affinity_matrix_to_laplacian import affinity_matrix_to_laplacian
from soft_segments_from_eigs import soft_segments_from_eigs
from group_segments import group_segments
from image_gradient import image_gradient
from sparsify_segments import sparsify_segments
from superpixels import Superpixels

import scipy
from scipy.misc import imread
import numpy as np


def semantic_soft_segmentation(image, features, laplacian=None, init_segments=None, eigen_vectors=None, eigen_values=None, init_soft_segments=None, grouped_segments=None):
    
    print('Semantic Soft Segmentation')
    
    if features.shape[2] > 3:
        features = preprocess_features(features, image)
    else:
        features = features.astype('float')

    superpixels = Superpixels(image)
    h, w, _ = image.shape
    
    
    print('     Computing affinities...')
    if laplacian is None:
        affinities = list()
        affinities.append(matting_affinity(image))

        
        affinities.append(superpixels.neighbor_affinities(features))
        affinities.append(superpixels.nearby_affinities(image))
        laplacian = affinity_matrix_to_laplacian(affinities[0]
                                                + 0.01 * affinities[1]
                                                + 0.01 * affinities[2]
                                                )    
    
    print('     Computing eigenvectors...')
    if eigen_values is None:
        eig_cnt = 100
        eigen_values, eigen_vectors = scipy.sparse.linalg.eigs(laplacian, k=eig_cnt, sigma=0, which='LM')
        eigen_values, eigen_vectors = np.diag(eigen_values.astype('float')), eigen_vectors.astype('float')


    # eigen_values = eigen_values[idx][:eig_cnt]
    # eigen_vectors = eigen_vectors[:, idx][:eig_cnt]
    
    
    if init_soft_segments is None:
        print('     Initial optimization...')    
        initial_segm_cnt = 40
        sparsity_param = 0.8
        iter_cnt = 40
        init_soft_segments = soft_segments_from_eigs(eigen_vectors, laplacian, h, w, 
                                                    eig_vals=eigen_values, features=features, 
                                                    comp_cnt=initial_segm_cnt, max_iter=iter_cnt, 
                                                    sparsity_param=sparsity_param, initial_segments=init_segments)
        
        
        
    
    print('     Final optimization...')
    if grouped_segments is None:
        grouped_segments = group_segments(init_soft_segments, features)
        
    img_grad, _, _, _ = image_gradient(image, False, 6)        
    soft_segments = sparsify_segments(grouped_segments, laplacian, img_grad)
    return soft_segments

    # return (
    #     soft_segments,
    #     init_soft_segments,
    #     laplacian,
    #     affinities,
    #     features,
    #     # superpixels,
    #     eigen_vectors,
    #     eigen_values
    # )
