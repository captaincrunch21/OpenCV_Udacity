import numpy as np
import cv2


#for plotting
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def showImage(WindowName,ImageMat):
    cv2.imshow(WindowName,ImageMat)
    cv2.waitKey(0)
    cv2.destroyWindow(WindowName)


def applyFilter(Filter,Image):
    filter = Filter
    iShape = Image.shape
    Filtered_Image = np.zeros((iShape[0]-1,iShape[1]-1))
    for i in range(1,iShape[0]-1):
        for j in range(1,iShape[1]-1):
            Image_mat = Image[i-1:i+2,j-1:j+2]
            mul = np.multiply(Image_mat,filter)
            value = np.sum(mul)
            Filtered_Image[i-1,j-1] = value

    print(Filtered_Image/np.sum(filter))





#Correlation Filtering
#Make Random image 10x10
r_image = np.array([
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,100,100,100,100,0,0,0],
    [0,0,0,100,100,100,100,0,0,0],
    [0,0,0,0,100,100,100,0,0,0],
    [0,0,0,0,100,100,100,0,0,0],
    [0,0,0,0,0,100,100,0,0,0],
    [0,0,0,0,0,100,100,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0]]
    )

avg_filter = np.array([
        [1,1,1],
        [1,1,1],
        [1,1,1]
        ])

gauss_filter = np.array([
        [1,2,1],
        [2,4,2],
        [1,2,1]
        ])

print(r_image)
# showImage("rand",r_image/100)



print("#############################################################")
applyFilter(avg_filter,r_image)

print("#############################################################")
applyFilter(gauss_filter,r_image)
