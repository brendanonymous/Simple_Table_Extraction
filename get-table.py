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
            img = cv.imread("out1.jpg")
            main.getMSEntTable1(img)

            # get table 2 from Sample 1.pdf
            # img = cv.imread("out2.jpg")
            # main.getMSEntTable2(img)

            # # get table 3 from Sample 1.pdf
            # img = cv.imread("out3.jpg")
            # main.getMSEntTable3(img)

            # # get table 4 from Sample 1.pdf
            # img = cv.imread("out4.jpg")
            # main.getMSEntTable2(img)

            # # get table 5 from Sample 1.pdf
            # img = cv.imread("out5.jpg")
            # main.getMSEntTable2(img)

            # # get table 6 from Sample 1.pdf
            # img = cv.imread("out6.jpg")
            # main.getMSEntTable2(img)



            # # get table 1 from sample 4.pdf
            # img = cv.imread("out1.jpg")
            # main.getSample4Table1(img)

            # get page 2 from sample 4.pdf
            # img = cv.imread("out2.jpg")
            # main.getSample4Table1(img)



            # # get tables from page 1 of sample 7.pdf # THIS ONE IS IN DEVELOPMENT
            # img = cv.imread("out1.jpg")
            # main.getSample7Table1(img)
        else:
            print("Error: file must be in PDF format")
    else:
        print("Error: Usage: python get-table.py <path to image>")
    
