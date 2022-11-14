import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as im
import glob
from rembg import remove
#import frst as fr


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


def edge_contour_search_algorithm(img_gray,image):
    ret, thresh = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY)
# visualize the binary image
    cv.imshow('Binary image', thresh)
    cv.waitKey(0)
    #cv.imwrite('image_thres1.jpg', thresh)
    cv.destroyAllWindows()
    contours, hierarchy = cv.findContours(image=thresh, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
    #return contours

    # draw contours on the original image
    image_copy = image.copy()
    cv.drawContours(image=image_copy, contours=contours, contourIdx=-1,
                 color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    

    # see the results
    cv.imshow('None approximation', image_copy)
    cv.waitKey(0)
    cv.imwrite('contours_none_image1.jpg', image_copy)
    
    for i in range(1):
            
            
            circles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT, dp=2, minDist=50,param1= ret, param2=30, minRadius=0, maxRadius=-1)
        # # cv.imshow("detected circles", img)
        # # cv.waitKey(0)
        
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                 # draw the outer circle
                    cv.circle(image_copy, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # draw the center of the circle
                    cv.circle(image_copy, (i[0], i[1]), 2, (0, 0, 255), 3)
        
                cv.imshow("detected circles", image_copy)
                cv.waitKey(0)
                cv.destroyAllWindows()

def main():
    # for i in range(len(argv)):
    for image in glob.glob('/home/bruno/Desktop/LABIAGI_grapesdetector/test/*.jpg'):
        # for image in glob.glob('test_img.png'):

        input = cv.imread(image, 1)

        cimg = remove(input)

        cv.imshow("img", cimg)

        cv.waitKey(0)

        img = cv.cvtColor(cimg, cv.COLOR_RGB2GRAY)
        img = cv.bilateralFilter(img, 5, 20, 20)
        cv.imshow("bilateral", img)
        cv.waitKey(0)

        img = cv.Canny(img, 1, 100)

        # Performs fast radial symmetry transform
        # img: input image, grayscale
        # radii: integer value for radius size in pixels (n in the original paper); also used to size gaussian kernel
        # alpha: Strictness of symmetry transform (higher=more strict; 2 is good place to start)
        # beta: gradient threshold parameter, float in [0,1]
        # stdFactor: Standard deviation factor for gaussian kernel
        # mode: BRIGHT, DARK, or BOTH

        # def frst(img, radii, alpha, beta, stdFactor, mode='BOTH'):
        S = frst(img, 1, 2, 0.1, 0.1, 'BOTH')
        cv.imshow("S", S)
        cv.waitKey(0)

    # for i in range(0,200):
    #     print(i)
    #     test  = cv.Canny(img,0,i)
    #     cv.imshow("Canned", test)
    #     cv.waitKey(0)
        
        contour_img = edge_contour_search_algorithm(img,cimg) ## forse S al posto di img?
        
        dst = cv.cornerHarris(img,2,3,0.04)
        #result is dilated for marking the corners, not important
        dst = cv.dilate(dst,None)
        ccimg = input.copy()
        ccimg[dst>0.01*dst.max()]=[0,0,255]
        cv.imshow("Edge", ccimg)
        cv.waitKey(0)


if __name__ == "__main__":
    # main(sys.argv[1:])
    main()
