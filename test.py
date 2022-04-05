import numpy as np
import matplotlib.pyplot as plt
import cv2

num = '04'
img_path = f'images/sss_{num}.png'
img_py_path = f'results/sss_{num}_python.png'
img_mat_path = f'results/sss_{num}_matlab.png'
img_paper_path = f'results/sss_{num}_paper.png'

img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
img_py = cv2.cvtColor(cv2.imread(img_py_path), cv2.COLOR_BGR2RGB)
img_mat = cv2.cvtColor(cv2.imread(img_mat_path), cv2.COLOR_BGR2RGB)
img_paper = cv2.cvtColor(cv2.imread(img_paper_path), cv2.COLOR_BGR2RGB)

img = cv2.resize(img, (210, 320))
img_py = cv2.resize(img_py, (210, 320))
img_mat = cv2.resize(img_mat, (210, 320))
img_paper = cv2.resize(img_paper, (210, 320))

img_list = []
img_list.append(img)
img_list.append(img_paper)
img_list.append(img_mat)
img_list.append(img_py)

options = ['RAW', 'PAPER', 'MATLAB', 'PYTHON']


fig = plt.figure(figsize=(20, 20))
columns = 4
rows = 1
for i in range(1, columns*rows +1):
    img = img_list[i-1]
    ax = fig.add_subplot(rows, columns, i) 
    ax.title.set_text(options[i-1])
    plt.axis("off")
    plt.imshow(img)

plt.savefig(f'results/sss_{num}_report.png')
plt.show()
