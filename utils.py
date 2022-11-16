import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as im
import glob
import math
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
    cv.imwrite('image_thres1.jpg', thresh)
    cv.destroyAllWindows()
    contours, hierarchy = cv.findContours(image=thresh, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    #return contours

    
    # draw contours on the original image
    image_copy = image.copy()
    cv.drawContours(image=image_copy, contours=contours, contourIdx=-1,
                 color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    #print(contours)
    i = 0
    for cnt in contours:
        if(len(cnt)>=5):
            res = image.copy()
            #print(cnt)
            ellipse= cv.fitEllipse(cnt)
            #print(ellipse[1])
            #print("MAJOR AXE: " + str(ellipse[1][0]))
            #print("Minoe AXE: " + str(ellipse[1][1]))
            a = ellipse[1][0]
            b = ellipse[1][1]
            ellipse_perimeter = 2*math.pi* math.sqrt((math.pow(a,2)+math.pow(b,2)/2))
            print("IL PERIMETRO DELL'ELLISSE È: "+ str(ellipse_perimeter))
            print("Il Perimetro in mm è: " + str((26.28*ellipse_perimeter)/651.43))
            #print(cv.point(cnt))
            #rect = cv.minAreaRect(cnt)
            #print(rect)
            box = cv.boxPoints(ellipse)
            box = np.int0(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
            #print(box)
            cv.drawContours(res, [box], 0, (255,0,0),thickness= 2)
            cv.ellipse(res, ellipse, color = (0,0,255),thickness = 2)
            ####################### L1 ########################################
            ptm_l1 = (round((box[0][0]+box[1][0])/2),round((box[0][1]+box[1][1])/2)) #PM L1
            # cv.circle(res,(box[0][0],box[0][1]), 3, (0,255,0), -1)#V0
            # cv.circle(res,(box[1][0],box[1][1]), 3, (0,255,0), -1)#V1
            cv.circle(res,ptm_l1, 3, (0,255,0), -1)
            ################################################################

            ###################### L2 ##########################################
            ptm_l2 = (round((box[3][0]+box[2][0])/2),round((box[3][1]+box[2][1])/2)) #PM L2
            # cv.circle(res,(box[3][0],box[3][1]), 10, (0,255,0), -1) #V3
            # cv.circle(res,(box[2][0],box[2][1]), 10, (0,255,0), -1) #V2
            cv.circle(res,ptm_l2, 3, (0,255,0), -1)
            ###################################################################

            ###################### L4 #########################################
            ptm_l4 = (round((box[0][0]+box[3][0])/2), round((box[0][1]+box[3][1])/2))
            cv.circle(res,ptm_l4, 3, (0,255,0), -1)
            ###################################################################

            ###################### L3 #########################################
            ptm_l3 = (round((box[1][0]+box[2][0])/2), round((box[1][1]+box[2][1])/2))
            cv.circle(res,ptm_l3, 3, (0,255,0), -1)
            #####################################################################





            

            cv.line(res, ptm_l1, ptm_l2, (100,255,255), 3) 
            cv.line(res,ptm_l3,ptm_l4, (100,255,255), 3)

            cv.imshow("RES",res)
            cv.waitKey(0)
            #cv.imwrite('/Results/image_thres'+str(i)+'.jpg', res)
            i+=1




    # # see the results
    # cv.imshow('None approximation', image_copy)
    # cv.waitKey(0)
    # cv.imwrite('contours_none_image1.jpg', image_copy)
    


    ###################SCOMMENTARE PER TROVARE I CENTRI#####################
    # for i in range(1):
            
            
    #         circles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT, dp=2, minDist=50,param1= ret, param2=30, minRadius=0, maxRadius=-1)
    #     # # cv.imshow("detected circles", img)
    #     # # cv.waitKey(0)
        
    #         if circles is not None:
    #             circles = np.uint16(np.around(circles))
    #             for i in circles[0, :]:
    #              # draw the outer circle
    #                 cv.circle(image_copy, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #                 # draw the center of the circle
    #                 cv.circle(image_copy, (i[0], i[1]), 2, (0, 0, 255), 3)
        
    #             cv.imshow("detected circles", image_copy)
    #             cv.waitKey(0)
    #             cv.destroyAllWindows()
