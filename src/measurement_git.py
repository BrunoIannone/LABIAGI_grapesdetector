import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as im
import glob
#from rembg import remove
import math
import berrysizeestimator as est
import os
from utils_git import *
ROOT_DIR = os.path.realpath(os.path.dirname(__file__))

def main():
    input_image = cv.imread(os.path.realpath(os.path.join(ROOT_DIR, '../dw.jpeg')), 1)
    hough_image = input_image.copy()
    ellipse_image = input_image.copy()
    # cv.imshow("input", input_image)
    # cv.waitKey(0)

    #costruttore dello stimatore
    
    estimator = est.BerrySizeEstimator(input_image)
    #Cerca il riferimento nell'immagine, 
    # in questo caso uso la Hough transform 
    # perché sto usando un oggetto circolare
    
    ref = estimator.detect_referiment() 
    #apro il file con le coordinate della bbox
    index_line =1
    accumulator = 0
    counter = 0
    min_error = 99
    max_error = 0
    pixel_per_metric = (2*ref)/4.0
    with open(os.path.realpath(os.path.join(ROOT_DIR, '../dw_circle.txt'))) as f:
        line = f.readline()
    
        while(line != ''):       
            print("Acino N°: " +  str(index_line))
            print(line)

        
            line = line.strip().split(" ")
            if(int(line[5]) == 0):
                # tl, tr, bl, br = estimator.NormalizedToImgCoordinatesForBbox(
                # float(line[1]), float(line[2]), float(line[3]), float(line[4]))
                # temp = input_image[tl[1]:bl[1],tl[0]:br[0],: ]
                # cv.imshow("circle",temp)
                # cv.waitKey(0)

                print("0 found")
                line = f.readline()
                index_line +=1

                continue


            #converto le coordinate
            tl, tr, bl, br = estimator.normalized_to_img_coordinates_for_bbox(
                float(line[1]), float(line[2]), float(line[3]), float(line[4]))
            
            #recupero la porzione di immagine selezionata dalla bbox
            cropped = input_image[tl[1]:bl[1],tl[0]:br[0],: ].copy()
            # cv.imshow("circle",cropped)
            # cv.waitKey(0)
            #Per fini di test, applico lo stimatore con entrambi i test
            
            estimated_diameter = estimator.detector_hough(cropped, hough_image[tl[1]:bl[1],tl[0]:br[0],: ],pixel_per_metric)
            if(estimated_diameter == -1):
                # tl, tr, bl, br = estimator.NormalizedToImgCoordinatesForBbox(
                # float(line[1]), float(line[2]), float(line[3]), float(line[4]))
                # temp = input_image[tl[1]:bl[1],tl[0]:br[0],: ]
                # cv.imshow("circle",temp)
                # cv.waitKey(0)
                line = f.readline()
                index_line+=1
                print("No valid circle")
                continue
            error = (float(line[6])/pixel_per_metric) - estimated_diameter
            if(error > max_error):
                max_error = error
            if(error < min_error):
                min_error = error
                if(min_error<0):
                    print("Negativo")
            
            accumulator += error
            counter +=1
            print("Diametro calcolato: " + str(estimated_diameter)+" cm")
            print("Diametro effettivo : "+ str(float(line[6])/pixel_per_metric) + " cm")
            
            
            
            
            #estimator.DetectorEllipses(cropped, ellipse_image[tl[1]:bl[1],tl[0]:br[0],: ],(2*ref)/4.0)
            
            line = f.readline()
            index_line+=1
            # cv.imshow("circle",hough_image[tl[1]:bl[1],tl[0]:br[0]])
            # cv.waitKey(0)

    print("Errore minimo: " + str(min_error) + " cm")
    print("Errore massimo: " + str(max_error)+" cm")
    print("Errore medio: " + str(accumulator/counter)+ " cm")
    
    cv.imwrite(os.path.realpath(os.path.join(ROOT_DIR, '../resHough.jpg')),hough_image)
    cv.imwrite(os.path.realpath(os.path.join(ROOT_DIR, '../resEllipse.jpg')),ellipse_image)


if __name__ == "__main__":
    # main(sys.argv[1:])
    main()
