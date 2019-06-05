__title__ = "MS Enterprise SWO Table Image Extractor"
__author__ = "Brendan Lauck"
__maintainer__ = "Brendan Lauck"
__email__ = "brendanlauck@gmail.com"
__status__ = "development"

import cv2 as cv
import main
import numpy as np
import sys
import utils


if __name__ == "__main__":
    """PROCESS INPUT, LOAD IMAGE, AND GET TABLE FROM IMAGE"""
    if len(sys.argv) == 2:
        imagePath = sys.argv[1]
        if imagePath[-4:] == ".pdf":
            _ = utils.pdfToJpg(imagePath)
            # get table 1
            # img = cv.imread("out1.jpg")
            # main.getMSEntTable1(img)

            # get table 2
            # img = cv.imread("out2.jpg")
            # main.getMSEntTable2(img)

            # get table 3
            img = cv.imread("out3.jpg")
            main.getMSEntTable3(img)

            # get table 1 from sample 4.pdf
            # img = cv.imread("out1.jpg")
            # main.getSample4Table1(img)

            # get tables from page 1 of sample 5
            # img = cv.imread("out1.jpg")
            # main.getSample5Table1(img)
        else:
            print("Error: file must be in PDF format")
    else:
        print("Error: Usage: python get-table.py <path to image>")
    
