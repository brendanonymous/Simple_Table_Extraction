__title__ = "OCR POC"
__author__ = "Brendan Lauck"
__maintainer__ = "Brendan Lauck"
__email__ = "v-brlauc@microsoft.com"
__status__ = "development"

import cv2 as cv
import main
import sys
import utils


if __name__ == "__main__":
    """PROCESS INPUT, LOAD IMAGE, AND GET TABLE FROM IMAGE"""
    if len(sys.argv) == 2:
        imagePath = sys.argv[1]
        if imagePath[-4:] == ".pdf":
            _ = utils.pdfToJpg(imagePath)
            # get table 1 from Sample 1.pdf
            img = cv.imread("out2.jpg")
            main.getMSEntTable1(img)


        else:
            print("Error: file must be in PDF format")
    else:
        print("Error: Usage: python get-table.py <path to image>")
    
