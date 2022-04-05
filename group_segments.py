import numpy as np
from sklearn.cluster import KMeans


def group_segments(segments, features, segm_cnt=5):

    h, w, cnt = segments.shape
    comp_features = np.zeros((cnt, features.shape[2]))
    
    for i in range(cnt):
        cc = segments[:,:,i:i+1] * features
        cc = np.sum(cc.reshape(-1, 3), axis=0) / np.sum(segments[:,:,i])
        comp_features[i,:] = cc
        
    ids = KMeans(n_clusters=segm_cnt).fit(comp_features).labels_
    grouped_segments = np.zeros((h, w, segm_cnt))
    for i in range(segm_cnt):
        grouped_segments[:,:,i] = np.sum(segments[:,:,ids==i], axis=2)
    
    return grouped_segments