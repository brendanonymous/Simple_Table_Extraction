__title__ = "Table Image Extractor"
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
            numPages = utils.pdfToJpg(imagePath)
            for pageNum in range(1, numPages + 1):
                img = cv.imread("out{}.jpg".format(pageNum))
                main.getTable(img, pageNum)
        elif imagePath[-4:] == ".jpg":
            main.getTable(cv.imread(imagePath), 1)
    else:
        print("Error: Usage: python get-table.py <path to image>")
    
