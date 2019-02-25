import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('somatic_cell.jpg',0)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(img)
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.figure(figsize=(15,5))
images = [blur, 0, th3]
titles = ['Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
plt.subplot(1,3,1),plt.imshow(images[0],'gray')
plt.title(titles[0]), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.hist(images[0].ravel(),256)
plt.title(titles[1]), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(images[2],'gray')
plt.title(titles[2]), plt.xticks([]), plt.yticks([])

im = cv2.imread('somatic_cell.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
img = cv2.drawContours(img, contours, -1, (255,255,255), 3)
plt.figure(figsize=(10,10))
plt.subplot(1,2,1),plt.title('Original Image'),plt.imshow(im)#,'red')
plt.subplot(1,2,2),plt.title('OpenCV.findContours'),plt.imshow(img,'gray')#,'red')

print('number of cells detceted: ',len(contours))
