import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as im
import glob
from rembg import remove
import math
import src.berrysizeestimator as est
#import frst as fr
from utils_git import *


def main():

    input_image = cv.imread("dw.jpeg", 1)
    result_image = input_image.copy()
    # cv.imshow("input", input_image)
    # cv.waitKey(0)

    estimator = est.BerrySizeEstimator(input_image)
    ref = estimator.DetectReferiment()
    print(ref)
    with open('/home/bruno/Desktop/LABIAGI_grapesdetector/yolov5/runs/detect/exp14/labels/dw.txt') as f:
        line = f.readline()
    
        while(line != ''):       
        #print("Ricomincio")
        #cropped = cv.imread(image, 1)

        # cv.imshow("image", cropped)
        # cv.waitKey(0)
        
            line = line.strip().split(" ")
            #print(line)
            tl, tr, bl, br = estimator.NormalizedToImgCoordinatesForBbox(
                float(line[1]), float(line[2]), float(line[3]), float(line[4]))
            # test = estimator.BbVerticesDrawer(input_image,tl,tr,bl,br)
            cropped = input_image[tl[1]:bl[1],tl[0]:br[0],: ].copy()
            #processed_im = estimator.Preprocessing(cropped)
            processed_im = estimator.TestPreprocess(cropped)
            # cv.imshow("prepro", processed_im)
            # cv.waitKey(0)
            
            # cv.imshow("input", input_image[tl[1]:bl[1],tl[0]:br[0],: ])
            
            # cv.waitKey(0)
            contour_img = edge_contour_search_algorithm(
                processed_im, result_image[tl[1]:bl[1],tl[0]:br[0],: ], (2*ref)/4.0)
            # cv.imshow("input", input_image)
            # cv.waitKey(0)
            #print("finisco")
            line = f.readline()

           
        # cv.imshow("input", result_image)
        # cv.waitKey(0)
        cv.imwrite('/home/bruno/Desktop/LABIAGI_grapesdetector/res.jpg',result_image)
if __name__ == "__main__":
    # main(sys.argv[1:])
    main()
