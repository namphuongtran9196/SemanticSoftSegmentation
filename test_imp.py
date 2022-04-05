import numpy as np
import matplotlib.pyplot as plt
import cv2

sss = np.load('npy/sss.npy')
sss = np.where(sss < 0.5, 0., 1.)
n_classes = sss.shape[-1]

for i in range(n_classes):
    cur_mask = sss[:, :, i]
    print(cur_mask[:10, :10])
    cv2.imshow(str(i), cur_mask)
    cv2.waitKey()
    cv2.imwrite(f'imp/{i}.png', (cur_mask*255).astype('uint8'))

