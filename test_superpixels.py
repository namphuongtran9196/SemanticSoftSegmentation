import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.io

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

from superpixels import Superpixels

mat = scipy.io.loadmat('mat/features_pca.mat')
features = mat['embedmap']
print(features.shape)

tmp = scipy.io.loadmat('mat/superpixels.mat')
L, N = tmp['L'], tmp['N']

image = cv2.cvtColor(cv2.imread('images/docia_resized.png'), cv2.COLOR_BGR2RGB) / 255.
superpixels = Superpixels(image)
slic_alg = mark_boundaries(image, superpixels.labels)
slic0_alg = mark_boundaries(image, L)
img_list = []
img_list.append(slic_alg)
img_list.append(slic0_alg)

names = ["SLIC", "SLIC0"]

fig = plt.figure(figsize=(20, 20))
columns = 2
rows = 1
for i in range(1, columns*rows +1):
    img = img_list[i-1]
    ax = fig.add_subplot(rows, columns, i) 
    ax.title.set_text(names[i-1])
    plt.axis("off")
    plt.imshow(img)

plt.show()
