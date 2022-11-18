import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as im
import glob
from rembg import remove
import math
#import frst as fr
from utils_git import *
def main():
    
    # for i in range(len(argv)):
    for image in glob.glob('/home/bruno/Desktop/LABIAGI_grapesdetector/test/*.jpg'):
        # for image in glob.glob('test_img.png'):
        input = cv.imread('dw.jpeg', 1)
        #cv.imshow("input1",input)
        #cv.waitKey(0)

        
        
        
        input = cv.imread(image, 1)

        cimg = remove(input)

        #cv.imshow("img", cimg)

        #cv.waitKey(0)

        img = cv.cvtColor(cimg, cv.COLOR_RGB2GRAY)
        img = cv.GaussianBlur(img, (7, 7), 0)
        # cv.imshow("bilateral", img)
        #cv.waitKey(0)

        img = cv.Canny(img, 50, 100)
        img = cv.dilate(img, None, iterations=1)
        img = cv.erode(img, None, iterations=1)

    
        
        contour_img = edge_contour_search_algorithm(img,cimg) 
        
    


if __name__ == "__main__":
    # main(sys.argv[1:])
    main()
