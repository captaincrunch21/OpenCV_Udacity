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


def addByAverage(Image1,Image2):
    NewImage = np.add(Image1/2 , Image2/2)
    print(NewImage)
    showImage("Averaged Image",NewImage/255)


def readGrayScaleImage(ImageName):
    image = cv2.imread(ImageName,0)
    print(ImageName)
    print(type(image))
    print(color_image.shape)
    print("###########")
    return image


def readImage(ImageName,color_channel=0):
    image = cv2.imread(ImageName)
    print(ImageName)
    print(type(image))
    print(color_image.shape)
    print("###########")
    if color_image.shape[2] > color_channel :
        return image[:,:,color_channel]
    else:
        return image

def addNoiseToImage(Image):
    noiseMat = np.random.random_sample(Image.shape)
    noisedIMage = addByAverage(Image,noiseMat)
    print(noiseMat)
    showImage(noisedIMage)

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

##Numpy array slicing from rows 101 to 103 and columns 201 to 203
sliced_array = image[101:103,201:203]
print(sliced_array)
#cv2.imshow("Image_Window",sliced_array)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#Color Images
color_image = cv2.imread("fruitBowl.jpg")
print(len(color_image),"x",len(color_image[0]),"x",len(color_image[0][0]))
print(color_image.shape)

# showImage("RedChannel",color_image[:,:,0])
# showImage("GreenChannel",color_image[:,:,1])
# showImage("BlueChannel",color_image[:,:,2])

#adding two Images
aero1 = readGrayScaleImage("aero1.jpg")
aero2 = readGrayScaleImage("building.jpg")
#showImage("aero1",aero1)
#showImage("aero2",aero2)

#addByAverage(aero1,aero2[:480,:640])
addNoiseToImage(aero2)


