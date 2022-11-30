from abc import ABC, abstractmethod
import pytest
import os
import sys
import cv2 as cv
import berrysizeestimator as est

ROOT_DIR = os.path.realpath(os.path.dirname(__file__))
REFERIMENT_MEASURE = 4.0  # Referiment measure in cm


def check_txt_format():
    """Test method for test txt right format
    """
    with open(os.path.realpath(os.path.join(ROOT_DIR, '../dw_ellipse.txt'))) as f:  # Path to ellipse txt
        line = f.readline()
        while (line != ''):
            line = line.strip().split(" ")

            for i in range(len(line)):
                if (float(line[i] < 0)):
                    assert False, "ound negative values"
                if (i == 5 and int(line[i] > 1)):
                    assert False, "Non binary value"

            line = f.readline()
        f.close()
    with open(os.path.realpath(os.path.join(ROOT_DIR, '../dw_circle.txt'))) as f:  # Patch to circle txt
        line = f.readline()
        while (line != ''):
            line = line.strip().split(" ")

            for i in range(len(line)):
                if (float(line[i] < 0)):
                    assert False, "Found negative values"
                if (i == 5 and int(line[i] > 1)):
                    assert False, "Non binary value"

            line = f.readline()
        f.close()
    assert True


def test_referiment_measure():
    """Test method for referiment measurement
    """
    input_image = cv.imread(os.path.realpath(
        os.path.join(ROOT_DIR, '../dw.jpeg')), 1)

    estimator = est.BerrySizeEstimator(input_image)

    res = estimator.detect_referiment()

    pixel_per_metric = (res)/REFERIMENT_MEASURE

    assert ((res/pixel_per_metric) >= 4) and ((res/pixel_per_metric) <= 4.5)


def test_ellipse_measurements():
    """
        Test method for ellipse measurement. Asjust the error range assert according to desired precision
    """
    input_image = cv.imread(os.path.realpath(
        os.path.join(ROOT_DIR, '../dw.jpeg')), 1) ## Test Image
    ellipse_image = input_image.copy()
    estimator = est.BerrySizeEstimator(input_image)
    ref = estimator.detect_referiment()

    # cv.imshow("input", input_image)
    # cv.waitKey(0)

    pixel_per_metric = (ref)/REFERIMENT_MEASURE

    with open(os.path.realpath(os.path.join(ROOT_DIR, '../dw_ellipse.txt'))) as f:
        line = f.readline()

        while (line != ''):

            line = line.strip().split(" ")

            if (int(line[5]) == 0):

                line = f.readline()

                continue

            tl, tr, bl, br = estimator.normalized_to_img_coordinates_for_bbox(
                float(line[1]), float(line[2]), float(line[3]), float(line[4]))

            cropped = input_image[tl[1]:bl[1], tl[0]:br[0], :].copy()
            estimated_axes = estimator.DetectorEllipses(
                cropped, ellipse_image[tl[1]:bl[1], tl[0]:br[0], :], pixel_per_metric)

            dimA_error = (float(line[6])/pixel_per_metric)-estimated_axes[0]
            dimB_error = (float(line[7])/pixel_per_metric)-estimated_axes[1]

            if (dimA_error > 10 or dimB_error > 10):
                assert False

            line = f.readline()

    assert True


def test_circles_measures():
    """        
    Test method for Hough circle transform measurement. Adjust the error range assert according to desired precision

    """
    input_image = cv.imread(os.path.realpath(
        os.path.join(ROOT_DIR, '../dw.jpeg')), 1) ## Test image
    hough_image = input_image.copy()
  

    estimator = est.BerrySizeEstimator(input_image)
    

    ref = estimator.detect_referiment()
    

    pixel_per_metric = (ref)/REFERIMENT_MEASURE

    with open(os.path.realpath(os.path.join(ROOT_DIR, '../dw_circle.txt'))) as f:
        line = f.readline()

        while (line != ''):

            line = line.strip().split(" ")

            if (int(line[5]) == 0):

                line = f.readline()

                continue

            tl, tr, bl, br = estimator.normalized_to_img_coordinates_for_bbox(
                float(line[1]), float(line[2]), float(line[3]), float(line[4]))

            
            cropped = input_image[tl[1]:bl[1], tl[0]:br[0], :].copy()

            

            estimated_diameter = estimator.detector_hough(
                cropped, hough_image[tl[1]:bl[1], tl[0]:br[0], :], pixel_per_metric)
            if (estimated_diameter == -1):
                line = f.readline()
                continue
            error = (float(line[6])/pixel_per_metric) - estimated_diameter

            if (error > 10):
                assert False, "Errore > 10 cm"

            line = f.readline()
