import numpy as np
from imguidedfilter import imguidedfilter
from scipy.io import loadmat
from scipy.sparse.linalg import eigs
import cv2

def feature_PCA(features, dim):
    features = features.astype('float')
    
    h, w, d = features.shape
    features = np.reshape(features, [h*w, d])
    featmean = np.mean(features, axis=0, keepdims=True)
    features = features - featmean
    covar = np.matmul(features.T, features)
    
    # eigen_values, eigen_vectors = np.linalg.eig(covar)
    # idx = eigen_values.argsort()
    # eigen_vectors = eigen_vectors[:, idx[:-dim-1:-1]]
    
    _, eigen_vectors = eigs(covar, k=3, which='LR')
    eigen_vectors = eigen_vectors.astype('float')
 
    pcafeat = np.matmul(features, eigen_vectors)
    pcafeat = pcafeat.reshape([h, w, dim])
    return pcafeat

# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def preprocess_features(features, image=None):
    features = features.astype('double')
    
    features[features > 5] = 5
    features[features < -5] = -5
    
    if image is not None:
        fd = features.shape[2]
        #fd = features.shape[3]
        maxfd = fd - fd % 3
        for i in range(0, maxfd, 3):
            #  features(:, :, i : i+2) = imguidedfilter(features(:, :, i : i+2), image, 'NeighborhoodSize', 10);
            features[:, :, i : i + 3] = imguidedfilter(features[:, :, i : i + 3], image, (10, 10), 0.01)
        for i in range(maxfd, fd):
            # features(:, :, i) = imguidedfilter(features(:, :, i), image, 'NeighborhoodSize', 10);
            features[:, :, i] = imguidedfilter(features[:, :, i], image, (10, 10), 0.01)
    
    simp = feature_PCA(features, 3)

    for i in range(0, 3):
        # simp(:,:,i) = simp(:,:,i) - min(min(simp(:,:,i)));
        simp[:, :, i] = simp[:, :, i] - simp[:, :, i].min()
        # simp(:,:,i) = simp(:,:,i) / max(max(simp(:,:,i)));
        simp[:, :, i] = simp[:, :, i] / simp[:, :, i].max()

    return simp


if __name__ == '__main__':
    image = cv2.cvtColor(cv2.imread('docia_resized.png'), cv2.COLOR_BGR2RGB) / 255.
    mat = loadmat('docia_resized.mat')
    features = mat['embedmap']

    features = preprocess_features(features, image)

    print(image.shape, features.shape)
    print(features[0, 0, 1])
