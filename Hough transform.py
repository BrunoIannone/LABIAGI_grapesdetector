import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as im


def main(argv):
    for i in range(len(argv)):
        print(argv[i])
        cimg = cv.imread(argv[i], 1)

        # cimg = cv.imread('circles-257002-edited.png', 1)
        img = cv.cvtColor(cimg, cv.COLOR_RGB2GRAY)

        img = cv.medianBlur(img, 5)
        # mThres_Gray = 0.0
        # CannyAccThresh = cv.threshold(img,mThres_Gray,0,255, cv.THRESH_OTSU)

        # CannyThresh = 0.1 * CannyAccThresh
        circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, dp=1, minDist=100,
                                  param1=30, param2=50, minRadius=0, maxRadius=0)

        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # draw the outer circle
            cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv.imshow("detected circles", cimg)
        cv.waitKey(0)

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
