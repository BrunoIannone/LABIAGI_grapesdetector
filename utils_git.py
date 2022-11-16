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
def gradx(img):
    img = img.astype('int')
    rows, cols = img.shape
    # Use hstack to add back in the columns that were dropped as zeros
    return np.hstack((np.zeros((rows, 1)), (img[:, 2:] - img[:, :-2])/2.0, np.zeros((rows, 1))))


def grady(img):
    img = img.astype('int')
    rows, cols = img.shape
    # Use vstack to add back the rows that were dropped as zeros
    return np.vstack((np.zeros((1, cols)), (img[2:, :] - img[:-2, :])/2.0, np.zeros((1, cols))))

# Performs fast radial symmetry transform
# img: input image, grayscale
# radii: integer value for radius size in pixels (n in the original paper); also used to size gaussian kernel
# alpha: Strictness of symmetry transform (higher=more strict; 2 is good place to start)
# beta: gradient threshold parameter, float in [0,1]
# stdFactor: Standard deviation factor for gaussian kernel
# mode: BRIGHT, DARK, or BOTH


def frst(img, radii, alpha, beta, stdFactor, mode='BOTH'):
    mode = mode.upper()
    assert mode in ['BRIGHT', 'DARK', 'BOTH']
    dark = (mode == 'DARK' or mode == 'BOTH')
    bright = (mode == 'BRIGHT' or mode == 'BOTH')

    workingDims = tuple((e + 2*radii) for e in img.shape)

    # Set up output and M and O working matrices
    output = np.zeros(img.shape, np.uint8)
    O_n = np.zeros(workingDims, np.int16)
    M_n = np.zeros(workingDims, np.int16)

    # Calculate gradients
    gx = gradx(img)
    gy = grady(img)

    # Find gradient vector magnitude
    gnorms = np.sqrt(np.add(np.multiply(gx, gx), np.multiply(gy, gy)))

    # Use beta to set threshold - speeds up transform significantly
    gthresh = np.amax(gnorms)*beta

    # Find x/y distance to affected pixels
    gpx = np.multiply(np.divide(gx, gnorms, out=np.zeros(
        gx.shape), where=gnorms != 0), radii).round().astype(int)
    gpy = np.multiply(np.divide(gy, gnorms, out=np.zeros(
        gy.shape), where=gnorms != 0), radii).round().astype(int)

    # Iterate over all pixels (w/ gradient above threshold)
    for coords, gnorm in np.ndenumerate(gnorms):
        if gnorm > gthresh:
            i, j = coords
            # Positively affected pixel
            if bright:
                ppve = (i+gpx[i, j], j+gpy[i, j])
                O_n[ppve] += 1
                M_n[ppve] += gnorm
            # Negatively affected pixel
            if dark:
                pnve = (i-gpx[i, j], j-gpy[i, j])
                O_n[pnve] -= 1
                M_n[pnve] -= gnorm

    # Abs and normalize O matrix
    O_n = np.abs(O_n)
    O_n = O_n / float(np.amax(O_n))

    # Normalize M matrix
    M_max = float(np.amax(np.abs(M_n)))
    M_n = M_n / M_max

    # Elementwise multiplication
    F_n = np.multiply(np.power(O_n, alpha), M_n)

    # Gaussian blur
    kSize = int(np.ceil(radii / 2))
    kSize = kSize + 1 if kSize % 2 == 0 else kSize

    S = cv.GaussianBlur(F_n, (kSize, kSize), int(radii * stdFactor))

    return S


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
def edge_contour_search_algorithm(img_gray,image):
    
    
    cnts= cv.findContours(image=img_gray, mode = cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    #return contours
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    
    # draw contours on the original image
    # image_copy = image.copy()
    # cv.drawContours(image=image_copy, contours=cnts, contourIdx=-1,
    #              color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    #print(contours)
    

    for c in cnts:
	# if the contour is not sufficiently large, ignore it
        if cv.contourArea(c) < 1 and len(c)<5:
            continue
	# compute the rotated bounding box of the contour
        ellipse= cv.fitEllipse(c)
        orig = image.copy()
        box = cv.BoxPoints(ellipse) if imutils.is_cv2() else cv.boxPoints(ellipse)
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
            cv.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
# compute the midpoint between the top-left and top-right points,
# followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
# draw the midpoints on the image
        cv.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
# draw lines between the midpoints
        cv.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
        (255, 0, 255), 2)
        cv.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
        (255, 0, 255), 2)
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
# if the pixels per metric has not been initialized, then
# compute it as the ratio of pixels to supplied metric
# (in this case, inches)
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / 2.32
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric
# draw the object sizes on the image
        cv.putText(orig, "{:.1f}cm".format(dimA),
        (int(tltrX ), int(tltrY )), cv.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2)
        cv.putText(orig, "{:.1f}cm".format(dimB),
        (int(trbrX ), int(trbrY)), cv.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2)
# show the output image
        print("Il perimetro dell'ellisse è: " + str(math.pi*(3*(dimA+dimB)-math.sqrt((3*dimA+dimB)*(dimA+3*dimB)))))
        cv.imshow("Image", orig)
        cv.waitKey(0)
        
   