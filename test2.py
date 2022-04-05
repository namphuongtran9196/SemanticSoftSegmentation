import numpy as np
import matplotlib.pyplot as plt
import cv2

img_path = 'images/sss_04.png'
res_path = 'results/final/sss_04.png'

img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
res = cv2.cvtColor(cv2.imread(res_path), cv2.COLOR_BGR2RGB)

img_list = []
img_list.append(img)
img_list.append(res)

options = ['RAW', 'RESULT']

fig = plt.figure(figsize=(20, 20))
columns = 2
rows = 1
for i in range(1, columns*rows +1):
    img = img_list[i-1]
    ax = fig.add_subplot(rows, columns, i) 
    ax.title.set_text(options[i-1])
    plt.axis("off")
    plt.imshow(img)

plt.savefig(f'results/final_vis/sss_04_final.png')
plt.show()