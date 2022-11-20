import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as im
import glob
import math
from imutils import perspective
from imutils import contours
import imutils
from scipy.spatial import distance as dist


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def edge_contour_search_algorithm(img_gray, image, pixelsPerMetric):
    #print("PPPPP" + str(pixelsPerMetric))

    cnts = cv.findContours(
        image=img_gray, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    # return contours
    #print(cnts)
    if(cnts == ((),None)):
        print("No valid contour found")
        return
    cnts = imutils.grab_contours(cnts)

    #(cnts, _) = contours.sort_contours(cnts)
    #pixelsPerMetric = None

    # draw contours on the original image
    # image_copy = image.copy()
    # cv.drawContours(image=image_copy, contours=cnts, contourIdx=-1,
    #              color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    # print(contours)
    existing = 0
    maxDimA = 0
    maxDimB = 0
    tl_max = 0
    tr_max = 0
    bl_max = 0
    br_max = 0
    max_midpoint_tltr = 0
    max_midpoint_blbr = 0
    max_midpoint_tlbl = 0
    max_midpoint_trbr = 0
    max_box = 0
    #print(cnts)
    for c in cnts:
        #print(c)
        # if the contour is not sufficiently large, ignore it
        if cv.contourArea(c) < 1 or len(c) < 5:
            continue
        existing = 1
        # compute the rotated bounding box of the contour
        #print("lunghezza" + str( len(c)))
        ellipse = cv.fitEllipse(c)
        orig = image.copy()
        box = cv.BoxPoints(ellipse) if imutils.is_cv2(
        ) else cv.boxPoints(ellipse)
        #box = cv.minAreaRect(c)
        #box = cv.BoxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
        box = np.array(box, dtype="int")
        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        cv.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        # loop over the original points and draw them
        for (x, y) in box:
        #     print("X" + str(x))
        #     print("Y" + str(y))
                if(x == y):
                        continue        
                cv.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
                
                
                        
        (tl, tr, br, bl) = box

        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        # draw the midpoints on the image
        if(tltrX == tltrY or blbrX == blbrY or tlblX == tlblY or trbrX == trbrY ):
                continue

        #print((int(tltrX), int(tltrY)))
        cv.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        #draw lines between the midpoints
        cv.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
        (255, 0, 255), 2)
        cv.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
        (255, 0, 255), 2)
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        # if pixelsPerMetric is None:
        #     pixelsPerMetric = dB / 2.32
        # print(dA)
        # print(pixelsPerMetric)
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric
        if (dimA > maxDimA and dimB > maxDimB):
            maxDimA = dimA
            maxDimB = dimB
            tl_max = tl
            tr_max = tr
            bl_max = bl
            br_max = br
            max_midpoint_tltr = (tltrX, tltrY)
            max_midpoint_blbr = (blbrX, blbrY)
            max_midpoint_tlbl = (tlblX, tlblY)
            max_midpoint_trbr = (trbrX, trbrY)
            max_box = box.copy()
        # cv.imshow("temp_contour", orig)
        # cv.waitKey(0)
    # draw the object sizes on the image
    if (existing and type(max_box) != type(0)):
        final = image.copy()
        #print("MaxBox: " + str(max_box))        
        cv.drawContours(image, [max_box.astype("int")], -1, (0, 255, 0), 2)
        for (x, y) in max_box:
                cv.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv.circle(image, (int(max_midpoint_tltr[0]), int(
                max_midpoint_tltr[1])), 5, (255, 0, 0), -1)
        cv.circle(image, (int(max_midpoint_blbr[0]), int(
                max_midpoint_blbr[1])), 5, (255, 0, 0), -1)
        cv.circle(image, (int(max_midpoint_tlbl[0]), int(
                max_midpoint_tlbl[1])), 5, (255, 0, 0), -1)
        cv.circle(image, (int(max_midpoint_trbr[0]), int(
                max_midpoint_trbr[1])), 5, (255, 0, 0), -1)
        cv.line(image, (int(max_midpoint_tltr[0]), int(max_midpoint_tltr[1])), (int(max_midpoint_blbr[0]), int(max_midpoint_blbr[1])),
                (255, 0, 255), 2)
        cv.line(image, (int(max_midpoint_tlbl[0]), int(max_midpoint_tlbl[1])), (int(max_midpoint_trbr[0]), int(max_midpoint_trbr[1])),
                (255, 0, 255), 2)
        cv.putText(image, "{:.1f}cm".format(dimA),
                (int(max_midpoint_tltr[0]), int(max_midpoint_tltr[1])), cv.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
        cv.putText(image, "{:.1f}cm".format(dimB),
                (int(max_midpoint_trbr[0]), int(max_midpoint_trbr[1])), cv.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)

        # show the output image
        res = math.pi*(3*(maxDimA+maxDimB)-math.sqrt((3*maxDimA+maxDimB)*(maxDimA+3*maxDimB)))
        print("Il perimetro dell'ellisse è: " + str(res))
        # cv.imshow("def_contour", final)
        # cv.waitKey(0)
