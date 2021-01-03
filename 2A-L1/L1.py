import numpy as np
import cv2


#for plotting
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

image = cv2.imread("bwt1.jpg")

#printing type
print(type(image))

#printing size
print(len(image),"x",len(image[0]))


# #plotting
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.scatter(np.arange(len(image)),np.arange(len(image[0])),image.flatten())
# plt.show()