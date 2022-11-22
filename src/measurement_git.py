import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as im
import glob
#from rembg import remove
import math
import berrysizeestimator as est

from utils_git import *


def main():

    input_image = cv.imread("dw.jpeg", 1)
    hough_image = input_image.copy()
    ellipse_image = input_image.copy()
    # cv.imshow("input", input_image)
    # cv.waitKey(0)

    #costruttore dello stimatore
    
    estimator = est.BerrySizeEstimator(input_image)
    #Cerca il riferimento nell'immagine, 
    # in questo caso uso la Hough transform 
    # perché sto usando un oggetto circolare
    
    ref = estimator.DetectReferiment() 
    #apro il file con le coordinate della bbox
    with open('/home/bruno/Desktop/LABIAGI_grapesdetector/yolov5/runs/detect/exp14/labels/dw.txt') as f:
        line = f.readline()
    
        while(line != ''):       
        
        
            line = line.strip().split(" ")
            
            #converto le coordinate
            tl, tr, bl, br = estimator.NormalizedToImgCoordinatesForBbox(
                float(line[1]), float(line[2]), float(line[3]), float(line[4]))
            
            #recupero la porzione di immagine selezionata dalla bbox
            cropped = input_image[tl[1]:bl[1],tl[0]:br[0],: ].copy()
            
            #Per fini di test, applico lo stimatore con entrambi i test
            estimator.DetectorHough(cropped, hough_image[tl[1]:bl[1],tl[0]:br[0],: ],(2*ref)/4.0)
            estimator.DetectorEllipses(cropped, ellipse_image[tl[1]:bl[1],tl[0]:br[0],: ],(2*ref)/4.0)
            
            line = f.readline()

    cv.imwrite('/home/bruno/Desktop/LABIAGI_grapesdetector/resHough.jpg',hough_image)
    cv.imwrite('/home/bruno/Desktop/LABIAGI_grapesdetector/resEllipse.jpg',ellipse_image)
main

if __name__ == "__main__":
    # main(sys.argv[1:])
    main()
