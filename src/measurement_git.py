# import sys
import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
# from PIL import Image as im
import glob
#FIXME: incapsula questa scelta di bkg removal in una classe o funzione a parte e rendila
# modificabile usando lo "strategy" pattern (se hai nozioni di ingegneria del software)
from rembg import remove
# import math
# import frst as fr
from utils_git import *

#FIXME: le funzionalità di questo main dovrebbero essere incapsulate in una classe, così che il risultato finale sia
# un codice in cui la classe può essere importata in altri progetti. Il main lo usi come uno script di data,
# o meglio ancora, ti fai una struttura di data automatico con pytest.
def main():
    # for i in range(len(argv)):
    # FIXME: rimuovi reference a path assoluti, crea una variabile ROOT_DIR in un file di configurazione python e usa
    #  il comando os.path.realpath(os.path.join(os.path.dirname(__file__), '..')) per impostarla. Tutti i path che
    #  usi nel progetto devono essere riferiti a questa
    for image in glob.glob('data/*.jpg'):
        # for image in glob.glob('test_img.png'):

        input = cv.imread(image, 1)

        # FIXME: a me da errore quando cerca di scaricare UNET perché è un file drive che è stato
        #  scaricato troppe volte. Conviene, se vuoi usare questo modulo, includere i pesi in una cartella locale. È
        #  poco carino che un progetto scrichi la roba i una cartella nascosta nella home...
        cimg = remove(input)

        cv.imshow("img", cimg)

        cv.waitKey(0)

        # FIXME: non sono sicuro che sia RGB, penso sia BGR di base. Attento anche sopra con il bkg removal. Fai un
        #  check
        img = cv.cvtColor(cimg, cv.COLOR_RGB2GRAY)
        img = cv.GaussianBlur(img, (7, 7), 0)
        # cv.imshow("bilateral", img)
        # cv.waitKey(0)

        img = cv.Canny(img, 50, 100)
        img = cv.dilate(img, None, iterations=1)
        img = cv.erode(img, None, iterations=1)

        # Performs fast radial symmetry transform
        # img: input image, grayscale
        # radii: integer value for radius size in pixels (n in the original paper); also used to size gaussian kernel
        # alpha: Strictness of symmetry transform (higher=more strict; 2 is good place to start)
        # beta: gradient threshold parameter, float in [0,1]
        # stdFactor: Standard deviation factor for gaussian kernel
        # mode: BRIGHT, DARK, or BOTH

        # def frst(img, radii, alpha, beta, stdFactor, mode='BOTH'):
        # S = frst(img, 1, 2, 0.1, 0.1, 'BOTH')
        # cv.imshow("S", S)
        # cv.waitKey(0)

        # for i in range(0,200):
        #     print(i)
        #     data  = cv.Canny(img,0,i)
        #     cv.imshow("Canned", data)
        #     cv.waitKey(0)

        contour_img = edge_contour_search_algorithm(img, cimg)  ## forse S al posto di img?

        # dst = cv.cornerHarris(img,2,3,0.04)
        # #result is dilated for marking the corners, not important
        # dst = cv.dilate(dst,None)
        # ccimg = input.copy()
        # ccimg[dst>0.01*dst.max()]=[0,0,255]
        # cv.imshow("Edge", ccimg)
        # cv.waitKey(0)


if __name__ == "__main__":
    # main(sys.argv[1:])
    main()
