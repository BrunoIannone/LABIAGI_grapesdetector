import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as im
import glob
from rembg import remove
import math

def edge_contour_search_algorithm(img_gray,image):
    ret, thresh = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY)
# visualize the binary image
    cv.imshow('Binary image', thresh)
    cv.waitKey(0)
    cv.imwrite('image_thres1.jpg', thresh)
    cv.destroyAllWindows()
    contours, hierarchy = cv.findContours(image=thresh, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    #return contours

    
    # draw contours on the original image
    image_copy = image.copy()
    cv.drawContours(image=image_copy, contours=contours, contourIdx=-1,
                 color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    #print(contours)
    
    for cnt in contours:
        if(len(cnt)>=5):
            res = image.copy()
            #print(cnt)
            ellipse= cv.fitEllipse(cnt)
            #print(ellipse[1])
            #print("MAJOR AXE: " + str(ellipse[1][0]))
            #print("Minoe AXE: " + str(ellipse[1][1]))
            a = ellipse[1][0]
            b = ellipse[1][1]
            ellipse_perimeter = 2*math.pi* math.sqrt((math.pow(a,2)+math.pow(b,2)/2))
            print("IL PERIMETRO DELL'ELLISSE È: "+ str(ellipse_perimeter))
            cv.ellipse(res, ellipse, color = (0,0,255),thickness = 2)
            cv.imshow("RES",res)
            cv.waitKey(0)

input = cv.imread('euro.png', 1)
cimg = remove(input)
cv.imshow("img", cimg)
cv.waitKey(0)
img = cv.cvtColor(cimg, cv.COLOR_RGB2GRAY)
img = cv.bilateralFilter(img, 5, 20, 20)
cv.imshow("bilateral", img)
cv.waitKey(0)
(T, threshInv) = cv.threshold(img, 0, 255,	cv.THRESH_BINARY_INV | cv.THRESH_OTSU) #OTSU

img = cv.Canny(img, 1, 100)
cv.imshow("img", img)
cv.waitKey(0)
#edge_contour_search_algorithm(img,cimg)
ccimg = input.copy()

circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, dp=2, minDist=50,param1= T, param2=30, minRadius=0, maxRadius=0)
        # # cv.imshow("detected circles", img)
        # # cv.waitKey(0)
        
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
                 # draw the outer circle
        cv.circle(ccimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # draw the center of the circle
        cv.circle(ccimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        print("Raggio: " + str(i[2]) + "\nCirconferenza: " + str(2*math.pi*i[2]))
        print("Il diametro di 1 euro in cm è 23,25 mm")
        cv.imshow("detected circles", ccimg)
        cv.waitKey(0)
        break ##primo cerchio
        