import matplotlib.pyplot as plt

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import cv2 as cv

from rembg import remove





cimg = cv.imread('test_img4.jpg',1)
image_rgb = remove(cimg)
#cv.imwrite(output_path, output)




# Load picture, convert to grayscale and detect edges
#image_rgb = cv. imread("test_img4.jpg", 1)
cv.imshow("Image", image_rgb)
cv.waitKey(0)
#image_gray = color.rgb2gray(image_rgb)
image_gray = cv.cvtColor(image_rgb, cv.COLOR_RGB2GRAY)
(th, threshed) = cv.threshold(image_gray, 127, 255,	cv.THRESH_BINARY_INV | cv.THRESH_OTSU) 
# cv.imshow("Threshold", threshed)
# cv.waitKey(0)
edges = cv.Canny(threshed,th, th/2)

cv.imshow("Image", edges)
cv.waitKey(0)

#th, threshed = cv.threshold(edges, 100, 255, cv.THRESH_BINARY)
(cnts, hiers) = cv.findContours(edges, 1,1)
print("CNTS" + str(cnts))
for cnt in cnts:
    if(len(cnt)>=5):
        res = cimg.copy()
        print(cnt)
        ((centx,centy), (width,height), angle)= cv.fitEllipse(cnt)
        print(type(centx))
        cv.ellipse(res, (int(centx),int(centy)), (int(width),int(height)), angle, 0,  360,  color = (0,255,0),thickness = 2)
        cv.imshow("RES",res)
        cv.waitKey(0)
print("ESCO")





# edges = cv.medianBlur(edges, 5)
# print(type(edges))
# cv.imshow("Edges", edges)
# cv.waitKey(0)
# result = edges.copy()
# edges = cv.Canny(edges,0.1,5)
# cv.imshow("Edges", edges)
# cv.waitKey(0)

# ((centx,centy), (width,height), angle) = cv.fitEllipse(edges)
# cv.ellipse(result, (int(centx),int(centy)), (int(width/2),int(height/2)), angle, 0, 360, (0,0,255), 1)


# cv.imshow("Res", result)
# cv.waitKey(0)