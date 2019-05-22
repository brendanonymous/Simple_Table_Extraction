# Gets a single table from a jpeg and writes to an excel file
import cv2 as cv
import numpy as np
import sys
import xlsxwriter
import utils


if __name__ == "__main__":
    # get arguments
    if len(sys.argv) == 2:
        imagePath = sys.argv[1]
    else:
        print("Error: Usage: python get-table.py <path to image>")
        sys.exit(0)



    ###################################
    # LOAD IMAGE
    ###################################
    original_image = cv.imread(imagePath)



    ###################################
    # CONVERT COLORSPACE TO NEGATIVE
    ###################################
    gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)



    ###################################
    # APPLY ADAPTIVE THRESHOLD
    ###################################
    gray_image = cv.bitwise_not(gray_image)
    _, image_withThreshold = cv.threshold(gray_image,127,255,cv.THRESH_BINARY)



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

    # get vertical lines
    rows = vertical_lines.shape[0]
    vertical_lines_size = rows // 30

    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_lines_size))
    
    vertical_lines = cv.erode(vertical_lines, vertical_kernel)    
    vertical_lines = cv.dilate(vertical_lines, vertical_kernel)



    ###################################
    # CREATE LINE MASK AND FIND CONTOURS
    ################################### 
    line_mask = horizontal_lines + vertical_lines

    (contours, _) = cv.findContours(line_mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)



    ###################################
    # EXTRACT TABLE AND OCR
    ###################################
    x_bounds = set()
    y_bounds = set()
    numRows = 0
    numCols = 0
    texts = []

    for i in range(len(contours) - 2, -1, -1):
        x, y, w, h = cv.boundingRect(contours[i]) # get contour coordinates
        x += 1; w -= 2 # offset HARD CODING, LOL YIKES
        box = original_image[y:y + h, x:x + w] # get bounding box
        texts.append(utils.run_tesseract(box, 8, 3)) # extract text from bounding box with Tesseract

        # use bounding box coordinates to track num rows and num cols
        if x not in x_bounds:
            x_bounds.add(x)
            numCols += 1
        if y not in y_bounds:
            y_bounds.add(y)
            numRows += 1



    ###################################
    # WRITE TABLE TO EXCEL
    ###################################
    workbook = xlsxwriter.Workbook('test_table.xlsx')
    worksheet = workbook.add_worksheet('table from image')

    text_index = 0
    for row in range(numRows): # double for loop, sue me
        for col in range(numCols):
            worksheet.write(row, col, texts[text_index])
            text_index += 1

    workbook.close()