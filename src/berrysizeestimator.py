from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np
from imutils import perspective
from imutils import contours
from math import pi
from math import sqrt
import imutils
from scipy.spatial import distance as dist
import math

class BerrySizeEstimatorAbstractClass(ABC):
    """ 
        Abstract class for berry size estimator

    """

    def __init__(self, input_image):
        """
            Abstract class constructor

        Args:
            input_image (image): Input image
        """
        self.input_image = input_image

    # @abstractmethod
    # def back_ground_remover(self, to_remove):
    #     pass

    @abstractmethod
    def detect_referiment(self):
        """
            Referiment detect method
        """
        pass
    

class BerrySizeEstimator(BerrySizeEstimatorAbstractClass):
    #TODO: in questo caso l'estimator è unico, ma si potrebbe dividere per ellissi e cerchi

    def __init__(self, input_image):
        """
            Class constructor

        Args:
            input_image (image): Input image
        """
        super().__init__(input_image)

    def normalized_to_img_coordinates_for_bbox(self, x_center, y_center, box_width, box_height):
        """Method that converts YOLO normalized coordinates  in array indexes

        Args:
            x_center (float): The bbox X normalized center coordinate
            y_center (float): The bbox Y normalized center coordinate
            box_width (float): The bbox normalized width 
            box_height (float): The bbox normalized height

        Returns:
           tuple of int: (top lef, top right, bottom left, bottom right) bbox indices
        """
        input_width = self.input_image.shape[1]

        input_height = self.input_image.shape[0]

        x1 = (x_center-(box_width/2))*input_width
        x2 = (x_center+(box_width/2))*input_width
        y1 = (y_center-(box_height/2))*input_height
        y2 = (y_center+(box_height/2))*input_height
        width = box_width*input_width
        height = box_height*input_height
        tr = (int(x1+width), int(y1))
        bl = (int(x2-width), int(y2))
        tl = (int(x1), int(y1))
        br = (int(x2), int(y2))

        return (tl, tr, bl, br)

    def bb_vertices_drawer(self, image, tl, tr, bl, br):
        """ Method that draws bbox vertices

        Args:
            image (image): input image
            tl (int): top left index
            tr (int): top right index
            bl (int): bottom left index
            br (int): bottom right index

        Returns:
            image: image with drawn bbox vertices
        """
        dest_img = image.copy()
        cv.circle(dest_img, tl, 5, (0, 0, 255), -1)
        cv.circle(dest_img, br, 5, (0, 0, 255), -1)
        cv.circle(dest_img, tr, 5, (0, 0, 255), -1)
        cv.circle(dest_img, bl, 5, (0, 0, 255), -1)

        return dest_img

    # def back_ground_remover(se_gf, to_remove):
    #     return remove(to_remove)

    def detect_referiment(self):
        """Method that detects referiment diameter measure in pixel

        Returns:
            float: detected referiment in pixel
        """
        img = cv.cvtColor(self.input_image, cv.COLOR_RGB2GRAY)
        img = cv.medianBlur(img, 5)
        (T, threshed_img) = cv.threshold(img, 127, 255,
                                      cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # OTSU
        
        circles = cv.HoughCircles(
            threshed_img, cv.HOUGH_GRADIENT, dp=2, minDist=50, param1=T, param2=100, minRadius=0, maxRadius=0)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            return 2*circles[0][0][2]

    def thresholded_canny(self, image):
        """Method for Canny edge detection with Otsu thresholding

        Args:
            image (image): input image

        Returns:
            image: Canned image  (edges)
        """
        T, threshed_img = cv.threshold(
            image, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        preprocessed_img = cv.Canny(threshed_img, T/2, T, L2gradient=True)
        return preprocessed_img

    def preprocessing(self, image):
        #preprocessed_img = self.back_ground_remover(image)
        preprocessed_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        preprocessed_img = cv.GaussianBlur(preprocessed_img, (5, 5), 0)
        
        return preprocessed_img

    def draw_hough_results(self,res,circle,pixels_per_metric):
        """Method that draws detected circles on the input image

        Args:
            res (image): image where circles are drawn
            circle (array): Circle array [x_center,y_center,radius]
            pixels_per_metric (float): Pixels per metric ratio
        """
        
        # draw the outer circle
        cv.circle(res, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv.circle(res, (circle[0], circle[1]), 2, (0, 0, 255), 3)
        diameter_cm = 2*circle[2] /pixels_per_metric
        
        ## UNCOMMENT TO DRAW PERIMETER RESULTS ON THE IMAGE ## 

        # cv.putText(res, "{:.1f}cm".format(diameter_cm*pi),
        # (i[0],i[1]), cv.FONT_HERSHEY_SIMPLEX,
        # 0.40, (255, 255, 255), 2)
        #print("La circonferenza è: " + str(diameter_cm *pi))
        
    def detector_hough(self, image, dest,pixels_per_metric):
        """Method that detect grapes with Hough circle transform

        Args:
            image (image): Input image
            dest (image): Destination image
            pixels_per_metric (float): Pixels per metric ratio

        Returns:
            int: Return the first detected diameter in cm or -1 if no circles are detected
        """
        prepeocessed_img = self.preprocessing(image)
        T, threshinv = cv.threshold(
            prepeocessed_img, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        if (T == 0):
            print("Threshold not valid")
            return

        circles = cv.HoughCircles(threshinv, cv.HOUGH_GRADIENT, dp=2,
                                  minDist=50, param1=T, param2=30, minRadius=0, maxRadius=0)
        
        if (circles is not None):
            circles = np.uint16(np.around(circles))
            self.draw_hough_results(dest,circles[0][0],pixels_per_metric)
            return (circles[0][0][2] *2)/pixels_per_metric
        else:
            return -1

    def midpoint(self,ptA, ptB):
        """ Method that calculates the midpoint between two points A and B

        Args:
            ptA (tuple of int): Point A
            ptB (tuple of int): Point B

        Returns:
            tuple of int: Calculated midpoint
        """
        
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


    
    def draw_results(self,image,max_midpoint_tltr,max_midpoint_blbr,max_midpoint_tlbl,max_midpoint_trbr,maxDimA,maxDimB,max_box):
        """method that draws detected ellipses on the input image

        Args:
            image (image): Input image
            max_midpoint_tltr (tuple of int): Top-left top-right midpoint
            max_midpoint_blbr (tuple of int): Bottom-left and bottom-right midpoint 
            max_midpoint_tlbl (tuple of int): Top-left bottom-left midpoint
            max_midpoint_trbr (tuple of int): Top-right bottom-right midpoint
            maxDimA (float): Ellipse A axes
            maxDimB (float): Ellipse B axis
            max_box (box): Ellipse box

        Returns:
            image: Image with drawn ellipses
        """

        ## UNCOMMENT FOR DRAW CONTOURS AND TEXT (text not well disposed) ##

        #cv.drawContours(image, [max_box.astype("int")], -1, (0, 255, 0), 2)
        
        # for (x, y) in max_box:
            #cv.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
    
        cv.circle(image, (int(max_midpoint_tltr[0]), int(
            max_midpoint_tltr[1])), 5, (255, 0, 0), -1)
        cv.circle(image, (int(max_midpoint_blbr[0]), int(
            max_midpoint_blbr[1])), 5, (255, 0, 0), -1)
        cv.circle(image, (int(max_midpoint_tlbl[0]), int(
            max_midpoint_tlbl[1])), 5, (255, 0, 0), -1)
        cv.circle(image, (int(max_midpoint_trbr[0]), int(
            max_midpoint_trbr[1])), 5, (255, 0, 0), -1)
        cv.line(image, (int(max_midpoint_tltr[0]), int(max_midpoint_tltr[1])), (int(max_midpoint_blbr[0]), int(max_midpoint_blbr[1])),
                (255, 0, 0), 2)
        cv.line(image, (int(max_midpoint_tlbl[0]), int(max_midpoint_tlbl[1])), (int(max_midpoint_trbr[0]), int(max_midpoint_trbr[1])),
                (0, 0, 255), 2)
        # cv.putText(image, "{:.1f}cm".format(maxDimA),
        #         (int(max_midpoint_tltr[0]), int(
        #             max_midpoint_tltr[1])), cv.FONT_HERSHEY_SIMPLEX,
        #         0.40, (255, 255, 255), 2)
        # cv.putText(image, "{:.1f}cm".format(maxDimB),
        #         (int(max_midpoint_trbr[0]), int(
        #             max_midpoint_trbr[1])), cv.FONT_HERSHEY_SIMPLEX,
        #         0.40, (255, 255, 255), 2)
        #print(max_box)
        #cv.drawContours(image, [max_box.astype("int")], -1, (0, 255, 0), 2)
        # cv.imshow("res", image)
        # cv.waitKey(0)

        return image
    
    def DetectorEllipses(self,cropped, image, pixels_per_metric):
        """Method that detect grapes with ellipses

        Args:
            cropped (image): Cropped original image (for calculation)
            image (image): destination image (for drawing)
            pixels_per_metric (float): Pixels per metric ratio

        Returns:
            tuple of int: Returns the two biggest axes (madxDimA,maxDimB) or nothing if there's no valid contour
        """
        img_gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
        img_gray = self.thresholded_canny(img_gray)
        cnts = cv.findContours(
            image=img_gray, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)

        if (cnts == ((), None)):
            print("No valid contour found")
            return
        cnts = imutils.grab_contours(cnts)

        maxDimA = 0
        maxDimB = 0
        
        max_midpoint_tltr = 0
        max_midpoint_blbr = 0
        max_midpoint_tlbl = 0
        max_midpoint_trbr = 0
        max_box = 0
        

        for c in cnts:

            # if the contour is not sufficiently large, ignore it
            if cv.contourArea(c) < 1 or len(c) < 5:
                continue
            
            # compute the rotated bounding box of the contour

            ellipse = cv.fitEllipse(c)
           
            box = cv.BoxPoints(ellipse) if imutils.is_cv2(
            ) else cv.boxPoints(ellipse)
            box = np.array(box, dtype="int")
            
            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            
            box = perspective.order_points(box)

            (tl, tr, br, bl) = box

            (tltrX, tltrY) = self.midpoint(tl, tr)
            (blbrX, blbrY) = self.midpoint(bl, br)
            
            # compute the midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-righ and bottom-right
            
            (tlblX, tlblY) = self.midpoint(tl, bl)
            (trbrX, trbrY) = self.midpoint(tr, br)
            # draw the midpoints on the image
            if (tltrX == tltrY or blbrX == blbrY or tlblX == tlblY or trbrX == trbrY):
                continue

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # if(dA >= image.shape[0]  or dB >= image.shape[1]):
            #     continue

            dimA = dA / pixels_per_metric
            dimB = dB / pixels_per_metric
            if (dimA > maxDimA and dimB > maxDimB):
                maxDimA = dimA
                maxDimB = dimB
                
                max_midpoint_tltr = (tltrX, tltrY)
                max_midpoint_blbr = (blbrX, blbrY)
                max_midpoint_tlbl = (tlblX, tlblY)
                max_midpoint_trbr = (trbrX, trbrY)
                max_box = box.copy()
                

            # cv.imshow("temp_contour", orig)
            # cv.waitKey(0)
        
        # draw the object sizes on the image

        if (type(max_box) != type(0)): ##Not a point
                  
            self.draw_results(image,max_midpoint_tltr,max_midpoint_blbr,max_midpoint_tlbl,max_midpoint_trbr,maxDimA,maxDimB,max_box)


            res = pi*(3*(maxDimA+maxDimB) -
                        math.sqrt((3*maxDimA+maxDimB)*(maxDimA+3*maxDimB)))
            
            
            return (maxDimA,maxDimB)

            

