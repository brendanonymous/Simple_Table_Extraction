# Gets a single table from a jpeg and writes to an excel file
import cv2 as cv
import numpy as np
import sys
from PIL import Image
import xlsxwriter
import imutils
import utils

# show image
def showImage(image, title):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# show contours
def showContours(contours):
    contours_image = np.zeros((720,1280,3), np.uint8)
    cv.drawContours(contours_image, contours, -1, (0,255,0), 2)
    showImage(contours_image, "contours")

# show contour on image
def showContoursOnImage(contours, image):
    for i in range(len(contours) - 1, -1, -1):
        img = image
        cv.drawContours(img, contours, i, (0, 0, 255), 1)
        showImage(img, "image with contour")

# show contours iteratively
def showContoursIter(contours):
    contours_image = np.zeros((720,1280,3), np.uint8)
    for i in range(len(contours) - 1, -1, -1):
        cv.drawContours(contours_image, contours, i, (0,255,0), 2)
        showImage(contours_image, "Contours")


if __name__ == "__main__":
    # get arguments
    if len(sys.argv) == 2:
        imagePath = sys.argv[1]
    else:
        print("Error: Usage: get-table <path to image>")
        sys.exit(0)



    ###################################
    # LOAD IMAGE
    ###################################
    original_image = cv.imread(imagePath)
    
    showImage(original_image, "original") # DEBUG



    ###################################
    # CONVERT COLORSPACE TO NEGATIVE
    ###################################
    gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    
    showImage(gray_image, "gray scale") # DEBUG



    ###################################
    # APPLY ADAPTIVE THRESHOLD
    ###################################
    gray_image = cv.bitwise_not(gray_image)
    #image_withThreshold = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, -2)
    _, image_withThreshold = cv.threshold(gray_image,127,255,cv.THRESH_BINARY)
    showImage(image_withThreshold, "threshold applied") # DEBUG



    ###################################
    # EXTRACT TABLE LINES
    ###################################
    horizontal_lines = np.copy(image_withThreshold)
    vertical_lines = np.copy(image_withThreshold)

    # get horizontal lines
    cols = horizontal_lines.shape[1]
    horizontal_lines_size = cols // 30

    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_lines_size, 1))

    horizontal_lines = cv.erode(horizontal_lines, horizontal_kernel)
    horizontal_lines = cv.dilate(horizontal_lines, horizontal_kernel)

    showImage(horizontal_lines, "horizontal lines") # DEBUG


    # get vertical lines
    rows = vertical_lines.shape[0]
    vertical_lines_size = rows // 30

    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_lines_size))
    
    vertical_lines = cv.erode(vertical_lines, vertical_kernel)    
    vertical_lines = cv.dilate(vertical_lines, vertical_kernel)    

    showImage(vertical_lines, "vertical lines") # DEBUG



    ###################################
    # CREATE LINE MASK AND FIND CONTOURS
    ################################### 
    line_mask = horizontal_lines + vertical_lines

    # find contours from line mask (twice for more accurate contours)

    showImage(line_mask, "horizontal + vertical + contours") # DEBUG

    (contours, _) = cv.findContours(line_mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # retrieval method may also be cv.RETR_EXTERNAL



    ###################################
    # EXTRACT TABLE AND OCR
    ###################################
    cell_bounds = {}
    texts = []

    for i in range(len(contours) - 2, -1, -1):
        x, y, w, h = cv.boundingRect(contours[i]) # get contour coordinates
        x += 1; w -= 2 # offset
        box = original_image[y:y + h, x:x + w] # get bounding box
        texts.append(utils.run_tesseract(box, 8, 3)) # extract text from bounding box with Tesseract

        # this is to find num rows and columns
        if cell_bounds.get(x) is None:
            cell_bounds[x] = [y]
        elif y not in cell_bounds.get(x):
            cell_bounds[x].append(y)
    
    numCols = len(cell_bounds)
    numRows = len(list(cell_bounds.values())[0])



    ###################################
    # WRITE TABLE TO EXCEL
    ###################################
    workbook = xlsxwriter.Workbook('test_table.xlsx')
    worksheet = workbook.add_worksheet('table from image')

    text_index = 0
    for row in range(numRows):
        for col in range(numCols):
            worksheet.write(row, col, texts[text_index])
            text_index += 1

    workbook.close()
    

    

    

    



    



    


