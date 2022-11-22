import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as im
import glob
from rembg import remove
#def main(argv):
def main():
    #for i in range(len(argv)):
    for image in glob.glob('/home/bruno/Desktop/LABIAGI_grapesdetector/data/*.jpg'):
    #for image in glob.glob('test_img.png'):
      

        input =  cv.imread(image,1)
        # cv.imshow("img",input)
        # cv.waitKey(0)
        cimg = remove(input)
        #cv.imwrite(output_path, output)
        #print(image)
        #cimg = cv.imread(image, 1)
        cv.imshow("img",cimg)
        cv.waitKey(0)

        #cimg = cv.imread('download.png', 1)
        img = cv.cvtColor(cimg, cv.COLOR_RGB2GRAY)
        img = cv.medianBlur(img, 5)
        (T, threshInv) = cv.threshold(img, 0, 255,	cv.THRESH_BINARY_INV | cv.THRESH_OTSU) #OTSU
        cv.imshow("Threshold", threshInv)
        cv.waitKey(0)
        print(T)
        # for i in range(0,200):
        #     print(i)
        #     data  = cv.Canny(img,0,i)
        #     cv.imshow("detected circles", data)
        #     cv.waitKey(0)

        # print(type(data))
        # print(data)
        
        
        
        #circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, dp=1, minDist=50, param1=40, param2=20, minRadius=0, maxRadius=0)<---- BUONO
        
        for i in range(1):
            ccimg = input.copy()
            print(i)
            circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, dp=2, minDist=50,param1= T, param2=30, minRadius=0, maxRadius=26)
        # # cv.imshow("detected circles", img)
        # # cv.waitKey(0)
        
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                 # draw the outer circle
                    cv.circle(ccimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # draw the center of the circle
                    cv.circle(ccimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        
                cv.imshow("detected circles", ccimg)
                cv.waitKey(0)
                cv.destroyAllWindows()
        
    print("Exiting...")
    return 0


if __name__ == "__main__":
    #main(sys.argv[1:])
    main()