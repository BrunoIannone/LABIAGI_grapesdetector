from abc import ABC, abstractmethod
import cv2 as cv


class BerrySizeEstimatorAbstractClass(ABC):
    """
        Implementa i metodi astratti per misurare gli acini

    """

    def __init__(self, input_image):
        self.input_image = input_image

    def estimate(self):
        pass

    def __preprocessing(self):
        pass
    def __background_remover(self,to_remove):
        pass
    def __normalized_to_img_coordinates_for_bbox(self):
        pass
class BerrySizeEstimator(BerrySizeEstimatorAbstractClass):

    def __init__(self, input_image):
        super().__init__(input_image)
    def estimate(self):
        return super().estimate()
    
    def __preprocessing(self):
    
        no_bkgrnd_img = self.__background_remover(self.input_image)
        preprocessed_img = cv.cvtColor(no_bkgrnd_img, cv.COLOR_RGB2GRAY)
        preprocessed_img = cv.GaussianBlur(preprocessed_img, (7, 7), 0)
        preprocessed_img = cv.Canny(preprocessed_img, 50, 100)
        preprocessed_img = cv.dilate(preprocessed_img, None, iterations=1)
        preprocessed_img = cv.erode(preprocessed_img, None, iterations=1)

    def __normalized_to_img_coordinates_for_bbox(self,x_center, y_center, box_width, box_height): ##classe test?
        input_width = self.input_image.shape[1]
        input_height = self.input_image.shape[0]
        x1 = ((x_center-box_width)/2)*input_width
        x2 = ((x_center+box_width)/2)*input_width
        y1 = ((y_center-box_height)/2)*input_height
        y2 = ((y_center+box_height)/2)*input_height
        width = box_width*input_width
        height = box_height*input_height
        tr = (x1+width,y1)
        bl = (x2-width,y2)
        tl = (x1,y1)
        br = (x2,y2)

        return (tl,tr,bl,br)

        
    def __bb_vertices_drawer(self,tl,tr,bl,br): ##classe test?
        dummy_input_img=self.input_image.copy()
        cv.circle(dummy_input_img, tl, 5, (0, 0, 255), -1)
        cv.circle(dummy_input_img, br, 5, (0, 0, 255), -1)
        cv.circle(dummy_input_img, tl, 5, (0, 0, 255), -1)
        cv.circle(dummy_input_img, bl, 5, (0, 0, 255), -1)     

        