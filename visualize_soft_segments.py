import numpy as np
from numpy import core

def dertermine_alpha_order(alphas):
    alphas = np.sum(alphas, axis=0)
    alphas = np.sum(alphas, axis=1)
    old_s = alphas.shape[2]
    alphas = alphas.reshape(old_s, 1)
    order = np.argsort(alphas)[::-1]
    return order

def create_palette():
    colors = np.array(
            [[0,117,220],    [255,255,128],    [43,206,72],      [153,0,0],       [128,128,128],        [240,163,255],
            [153,63,0],     [76,0,92],        [0,92,49],        [255,204,153],    [148,255,181],        [143,124,0],    
            [157,204,0],    [194,0,136],      [0,51,128],       [255,164,5],      [255,168,187],        [66,102,0], 
            [255,0,16],    [94,241,242],     [0,153,143],      [224,255,102],    [116,10,255],         [255,255,0],
            [255,80,5],     [25,25,25]])
    
    colors = colors.reshape(26, 1, 3) / 255.
    colors = np.vstack([colors, colors])
    return colors

def visualize_soft_segments(soft_segments, do_ordering=False):
    if do_ordering:
        # Order layers w.r.t. sum(alpha(:)) -- makes visualizations more consistent across images
        order = dertermine_alpha_order(soft_segments)
        soft_segments = soft_segments[:, :, order]
    
    # A hard-coded set of 'distinct' colors
    colors = create_palette()
    
    # One solid color per segment, mixed w.r.t. alpha values
    h, w, n_c = soft_segments.shape
    vis = np.tile(soft_segments[:, :, 0:1], [1, 1, 3]) * np.tile(colors[0, 0, :], [h, w, 1])
    for i in range(1, n_c):
        vis = vis + np.tile(soft_segments[:, :, i:i+1], [1, 1, 3]) * np.tile(colors[i, 0, :], [h, w, 1])
    
    return vis

if __name__ == '__main__':
    import scipy.io
    import matplotlib.pyplot as plt
    
    tmp = scipy.io.loadmat('mat/sss.mat')
    sss = tmp['sss']
    # print(sss.shape)
    vis_img = visualize_soft_segments(sss)
    plt.imshow(vis_img)
    plt.show()
    
    
        



    