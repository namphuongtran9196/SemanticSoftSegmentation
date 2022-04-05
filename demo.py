from scipy.misc import imread
from semantic_soft_segmentation import semantic_soft_segmentation
from visualize_soft_segments import visualize_soft_segments
import matplotlib.pyplot as plt
import scipy.io
import cv2
import argparse
import sys
import numpy as np



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str,
                        help='Path to image', default='images/docia_resized.png')
    parser.add_argument('--features_path', type=str,
                        help='Path to folder image', default='mat/features_pca.mat')
    args = parser.parse_args(sys.argv[1:])
        
    image = cv2.cvtColor(cv2.imread(args.image_path), cv2.COLOR_BGR2RGB) / 255.
    print(image.shape)

    mat = scipy.io.loadmat(args.features_path)
    try:
        features = mat['features']
    except:
        features = mat['embedmap']
    
    # tmp = scipy.io.loadmat('mat/Laplacian.mat')
    # Laplacian = tmp['Laplacian']
    
    # tmp = scipy.io.loadmat('mat/initialSegments.mat')
    # init_segments = tmp['initialSegments']
    # init_segments = init_segments - 1
    
    
    # tmp = scipy.io.loadmat('mat/eigens.mat')
    # eigenvalues = tmp['eigenvalues']
    # eigenvectors = tmp['eigenvectors']
    
    # tmp = scipy.io.loadmat('mat/initSoftSegments.mat')
    # init_soft_segments = tmp['initSoftSegments']
    
    # tmp = scipy.io.loadmat('mat/groupedSegments.mat')
    # grouped_segments = tmp['groupedSegments']

    # sss = semantic_soft_segmentation(image, features, laplacian=Laplacian, eigen_vectors=eigenvectors, eigen_values=eigenvalues, init_segments=init_segments, init_soft_segments=init_soft_segments, grouped_segments=grouped_segments)
    sss = semantic_soft_segmentation(image, features)
    np.save('sss_04.npy', sss)
    vis_img = visualize_soft_segments(sss)
    plt.imshow(vis_img)
    plt.show()