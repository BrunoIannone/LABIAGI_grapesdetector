
import pytest
import os
import sys
import cv2 as cv
import unittest
ROOT_DIR = os.path.realpath(os.path.dirname(__file__))


from berrysizeestimator import BerrySizeEstimator as est


class TestBerrySizeEstimator(unittest.TestCase):

    def test_detect_referiment(self):
        input_image = cv.imread(os.path.realpath(os.path.join(ROOT_DIR, '../dw.jpeg')), 1)
        estimator = est.BerrySizeEstimator(input_image)
        self.assertEqual(estimator.de)

    
    def test_estimate(self):
        # init estimator

        # load test inputs
        input_image = cv.imread(os.path.realpath(os.path.join(ROOT_DIR, '../dw.jpeg')), 1)
        hough_image = input_image.copy()
        estimator = est.BerrySizeEstimator(input_image)
        ref = estimator.detect_referiment() 
        self.ass

        #detect referiment

        # estimate
        # compare with expected measurements
        assert False
