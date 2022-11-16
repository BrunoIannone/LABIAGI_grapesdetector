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

        input = cv.imread(image, 1)

        cimg = remove(input)

        cv.imshow("img", cimg)

        cv.waitKey(0)

        img = cv.cvtColor(cimg, cv.COLOR_RGB2GRAY)
        img = cv.GaussianBlur(img, (7, 7), 0)
       # cv.imshow("bilateral", img)
        #cv.waitKey(0)

        img = cv.Canny(img, 50, 100)
        img = cv.dilate(img, None, iterations=1)
        img = cv.erode(img, None, iterations=1)

        # Performs fast radial symmetry transform
        # img: input image, grayscale
        # radii: integer value for radius size in pixels (n in the original paper); also used to size gaussian kernel
        # alpha: Strictness of symmetry transform (higher=more strict; 2 is good place to start)
        # beta: gradient threshold parameter, float in [0,1]
        # stdFactor: Standard deviation factor for gaussian kernel
        # mode: BRIGHT, DARK, or BOTH

        # def frst(img, radii, alpha, beta, stdFactor, mode='BOTH'):
        # S = frst(img, 1, 2, 0.1, 0.1, 'BOTH')
        # cv.imshow("S", S)
        # cv.waitKey(0)

    # for i in range(0,200):
    #     print(i)
    #     test  = cv.Canny(img,0,i)
    #     cv.imshow("Canned", test)
    #     cv.waitKey(0)
        
        contour_img = edge_contour_search_algorithm(img,cimg) ## forse S al posto di img?
        
        # dst = cv.cornerHarris(img,2,3,0.04)
        # #result is dilated for marking the corners, not important
        # dst = cv.dilate(dst,None)
        # ccimg = input.copy()
        # ccimg[dst>0.01*dst.max()]=[0,0,255]
        # cv.imshow("Edge", ccimg)
        # cv.waitKey(0)


if __name__ == "__main__":
    # main(sys.argv[1:])
    main()
