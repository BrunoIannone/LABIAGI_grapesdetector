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
    min_error_ellipse = 99
    max_error_ellipse = 0
    accumulator_ellipse = 0

    # with open(os.path.realpath(os.path.join(ROOT_DIR, '../dw_circle.txt'))) as f:
    #     line = f.readline()
    
    #     while(line != ''):       
    #         #print("Acino N°: " +  str(index_line))
    #         print(line)

        
    #         line = line.strip().split(" ")
    #         if(int(line[5]) == 0):
               

    #             #print("0 found")
    #             line = f.readline()
    #             index_line +=1

    #             continue


    #         #converto le coordinate
    #         tl, tr, bl, br = estimator.normalized_to_img_coordinates_for_bbox(
    #             float(line[1]), float(line[2]), float(line[3]), float(line[4]))
            
    #         #recupero la porzione di immagine selezionata dalla bbox
    #         cropped = input_image[tl[1]:bl[1],tl[0]:br[0],: ].copy()
            
    #         #Per fini di test, applico lo stimatore con entrambi i test
            
    #         estimated_diameter = estimator.detector_hough(cropped, hough_image[tl[1]:bl[1],tl[0]:br[0],: ],pixel_per_metric)
    #         if(estimated_diameter == -1):
    #             line = f.readline()
    #             index_line+=1
    #             #print("No valid circle")
    #             continue
    #         error = (float(line[6])/pixel_per_metric) - estimated_diameter
    #         if(error > max_error):
    #             max_error = error
    #         if(error < min_error):
    #             min_error = error
    #             if(min_error<0):
    #                 print("Negativo")
            
    #         accumulator += error
    #         counter +=1
    #         # print("Diametro calcolato: " + str(estimated_diameter)+" cm")
    #         # print("Diametro effettivo : "+ str(float(line[6])/pixel_per_metric) + " cm")
            
            
            
            
            
            
    #         line = f.readline()
    #         index_line+=1
            
    # print("Errore minimo: " + str(min_error) + " cm")
    # print("Errore massimo: " + str(max_error)+" cm")
    # print("Errore medio: " + str(accumulator/counter)+ " cm")
    # cv.imwrite(os.path.realpath(os.path.join(ROOT_DIR, '../resHough.jpg')),hough_image)

    index_line =1
    accumulator_dimA = 0
    accumulator_dimB = 0
    counter = 0
    min_error_ellipse = 99
    max_error_ellipse = 0
    accumulator_ellipse = 0
    with open(os.path.realpath(os.path.join(ROOT_DIR, '../dw_ellipse.txt'))) as f:
        line = f.readline()
    
        while(line != ''):       
            #print("Acino N°: " +  str(index_line))
            print(line)

        
            line = line.strip().split(" ")
            if(int(line[5]) == 0):
               

                #print("0 found")
                line = f.readline()
                index_line +=1

                continue


            #converto le coordinate
            tl, tr, bl, br = estimator.normalized_to_img_coordinates_for_bbox(
                float(line[1]), float(line[2]), float(line[3]), float(line[4]))
            
            #recupero la porzione di immagine selezionata dalla bbox
            cropped = input_image[tl[1]:bl[1],tl[0]:br[0],: ].copy()
            estimated_axes = estimator.DetectorEllipses(cropped, ellipse_image[tl[1]:bl[1],tl[0]:br[0],: ],(2*ref)/4.0)
            
            print("DimA calcolata: " + str(estimated_axes[0])+" cm")
            print("DimA effettiva : "+ str(float(line[6])/pixel_per_metric) + " cm")
            print("DiB calcolata: " + str(estimated_axes[1])+" cm")
            print("DiamB effettiva : "+ str(float(line[7])/pixel_per_metric) + " cm")

            dimA_error =  (float(line[6])/pixel_per_metric)-estimated_axes[0]
            dimB_error = (float(line[7])/pixel_per_metric)-estimated_axes[1]
            accumulator_dimA += dimA_error
            accumulator_dimB += dimB_error
            counter+=1

            line = f.readline()


    print("Errore medio dimA: " + str(accumulator_dimA/counter)+ " cm")
    print("Errore medio dimB: " + str(accumulator_dimB/counter)+ " cm")
   
    cv.imwrite(os.path.realpath(os.path.join(ROOT_DIR, '../resEllipse.jpg')),ellipse_image)


if __name__ == "__main__":
    # main(sys.argv[1:])
    main()
