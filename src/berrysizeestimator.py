from abc import ABC, abstractmethod
import cv2 as cv
from rembg import remove
import numpy as np
from imutils import perspective
from imutils import contours
import imutils


class BerrySizeEstimatorAbstractClass(ABC):
    """
        Implementa i metodi astratti per misurare gli acini

    """

    def __init__(self, input_image):
        self.input_image = input_image

    @abstractmethod
    def Estimate(self):
        pass

    @abstractmethod
    def Preprocessing(self):
        pass

    @abstractmethod
    def BackGroundRemover(self, to_remove):
        pass

    @abstractmethod
    def NormalizedToImgCoordinatesForBbox(self):
        pass

    def DetectReferiment(self):
        pass


class BerrySizeEstimator(BerrySizeEstimatorAbstractClass):

    def __init__(self, input_image):
        super().__init__(input_image)

    def Estimate(self):
        return super().estimate()

    def Preprocessing(self,image):
        (T, threshInv) = cv.threshold(image, 0, 255,	cv.THRESH_BINARY_INV or cv.THRESH_OTSU) #OTSU
        
        no_bkgrnd_img = self.BackGroundRemover(image)
        preprocessed_img = cv.cvtColor(no_bkgrnd_img, cv.COLOR_RGB2GRAY)
        preprocessed_img = cv.GaussianBlur(preprocessed_img, (7, 7), 0)
        preprocessed_img = cv.Canny(preprocessed_img, T/2, T,L2gradient=True)
        # preprocessed_img = cv.dilate(preprocessed_img, None, iterations=1)
        # preprocessed_img = cv.erode(preprocessed_img, None, iterations=1)
        return preprocessed_img

    # classe test?
    def NormalizedToImgCoordinatesForBbox(self, x_center, y_center, box_width, box_height):
        input_width = self.input_image.shape[1]
        #print(input_width)
        input_height = self.input_image.shape[0]
        #print(input_height)
        # print("X_CENTER: " + str(x_center))
        # print("Y_CENTER: " + str(y_center))
        # print("BOX WIDTH: " + str(box_width))
        # print("BOX_HEIGHT: " + str(box_height))
        
        x1 = (x_center-(box_width/2))*input_width
        x2 = (x_center+(box_width/2))*input_width
        y1 = (y_center-(box_height/2))*input_height
        y2 = (y_center+(box_height/2))*input_height
        width = box_width*input_width
        height = box_height*input_height
        tr = (int(x1+width), int(y1))
        bl = (int(x2-width), int(y2))
        tl = (int(x1), int(y1))
        #print(tl)
        br = (int(x2), int(y2))
        #print(tl)
        # print(br)
        #print(tr)
        #print(bl)
        return (tl, tr, bl, br)

    def BbVerticesDrawer(self, image, tl, tr, bl, br):   #classe test?
        dummy_input_img = image.copy()
        cv.circle(dummy_input_img, tl, 5, (0, 0, 255), -1)
        cv.circle(dummy_input_img, br, 5, (0, 0, 255), -1)
        cv.circle(dummy_input_img, tr, 5, (0, 0, 255), -1)
        cv.circle(dummy_input_img, bl, 5, (0, 0, 255), -1)

        # cv.circle(dummy_input_img, (0,0), 50, (0, 0, 255), -1)
        return dummy_input_img

    def BackGroundRemover(self, to_remove):
        return remove(to_remove)

    def DetectReferiment(self):
        img = cv.cvtColor(self.input_image, cv.COLOR_RGB2GRAY)
        img = cv.medianBlur(img, 5)
        (T, threshInv) = cv.threshold(img, 0, 255,
                                      cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # OTSU
        # cv.imshow("thres", threshInv)
        # cv.waitKey(0)
        circles = cv.HoughCircles(
            img, cv.HOUGH_GRADIENT, dp=2, minDist=50, param1=T, param2=100, minRadius=0, maxRadius=0)

        ccimg = self.input_image.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            #print (circles[0][0])
            cv.circle(ccimg, (circles[0][0][0], circles[0][0][1]), circles[0][0][2], (0, 255, 0), 2)
            # draw the center of the circle
            cv.circle(ccimg, (circles[0][0][0], circles[0][0][1]), 2, (0, 0, 255), 3)
            # cv.imshow("ref", ccimg)
            # cv.waitKey(0)

            return circles[0][0][2]
            
