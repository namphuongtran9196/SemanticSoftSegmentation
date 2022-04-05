from math import sqrt
import numpy as np
from scipy.signal import convolve2d


def image_gradient(image, out_color=True, filt_size=6):
    if filt_size <= 3:
        filt_size = 1
        dk = np.array([0.425287, -0.0000, -0.425287])
        kk = np.array([0.229879, 0.540242, 0.229879])
    elif filt_size <= 5:
        filt_size = 4
        dk = np.array([0.0032, 0.0350, 0.1190, 0.1458, -0.0000, -0.1458, -0.1190, -0.0350, -0.0032])
        kk = np.array([0.0009, 0.0151, 0.0890, 0.2349, 0.3201, 0.2349, 0.0890, 0.0151, 0.0009])
    else:
        filt_size = 6
        dk = np.array([0.0001, 0.0019, 0.0142, 0.0509, 0.0963, 0.0878, 0.0000, -0.0878, -0.0963, -0.0509, -0.0142, -0.0019, -0.0001])
        kk = np.array([0.0000, 0.0007, 0.0071, 0.0374, 0.1126, 0.2119, 0.2605, 0.2119, 0.1126, 0.0374, 0.0071, 0.0007, 0.0000])
    
    # Repeat-pad image
    left_pad = image[:, 0:1, :]
    right_pad = image[:, -2:-1, :]
    
    image = np.concatenate([np.tile(left_pad, [1, 13, 1]), image, np.tile(right_pad, [1, 13, 1])], axis=1)
    
    up_pad = image[0:1, :, :]
    down_pad = image[-2:-1, :, :]
    
    image = np.concatenate([np.tile(up_pad, [13, 1, 1]), image, np.tile(down_pad, [13, 1, 1])])
    
    # Compute gradients
    y_gradients = np.zeros(image.shape)
    x_gradients = np.zeros(image.shape)
    
    for i in range(image.shape[2]):
        y_gradients[:, :, i] = conv2(dk, kk, image[:, :, i], mode='same')
        x_gradients[:, :, i] = conv2(kk, dk, image[:, :, i], mode='same')

    # Remove padding
    y_gradients = y_gradients[13:-13, 13:-13, :]
    x_gradients = x_gradients[13:-13, 13:-13, :]
    
    # Compute pixel-wise L2 norm if no color option is selected
    if not out_color:
        y_gradients = np.sqrt(np.sum(y_gradients * y_gradients, axis=2))
        x_gradients = np.sqrt(np.sum(x_gradients * x_gradients, axis=2))
    
    gradient_magnitude = np.sqrt(y_gradients * y_gradients + x_gradients * x_gradients)
    gradient_orientation = np.arctan2(x_gradients, y_gradients)
    
    return gradient_magnitude, gradient_orientation, x_gradients, y_gradients
    

def conv2(u, v, img, mode='same'):
    conv1_img = convolve2d(img, u.reshape(-1, 1), mode=mode)
    conv2_img = convolve2d(conv1_img, v.reshape(1, -1), mode=mode)
    return conv2_img
    