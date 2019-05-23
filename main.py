import cv2 as cv
from openpyxl import Workbook, load_workbook
import os
import utils

def getTable(original_image, outputSheetNum):
    """GET TABLE FROM IMAGE AND WRITE TO EXCEL"""
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
    horizontal_lines, vertical_lines = utils.extractTableLines(image_withThreshold)



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
    if not os.path.exists("test_table.xlsx"): # create new workbook with new sheet
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = "table {} from image".format(outputSheetNum)
    else: # add new sheet
        workbook = load_workbook("test_table.xlsx")
        worksheet = workbook.create_sheet("table {} from image".format(outputSheetNum))

    text_index = 0
    for Row in range(1, numRows + 1): # double for loop, sue me THERE IS A BETTER WAY TO DO THIS
        for Col in range(1, numCols + 1):
            worksheet.cell(row=Row, column=Col).value = texts[text_index]
            text_index += 1

    workbook.save(filename="test_table.xlsx")