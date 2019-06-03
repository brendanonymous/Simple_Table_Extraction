import cv2 as cv
import json
import numpy as np
from openpyxl import Workbook, load_workbook
import os
import sys
import utils

def getTable(original_image, outputSheetNum):
    """GET TABLE FROM MS ENTERPRISE SWO IMAGE AND BUILD JSON STRUCTURE"""
    ###################################
    # CONVERT COLORSPACE TO NEGATIVE
    ###################################
    original_image = cv.resize(original_image, (int(original_image.shape[1] * 0.2), int(original_image.shape[0]*0.2)), interpolation = cv.INTER_AREA)
    gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)



    ###############################################
    # APPLY ADAPTIVE THRESHOLD AND NEGATIVE
    ###############################################
    threshold = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 1)    
    threshold = cv.bitwise_not(threshold) # negative



    ###############################################
    # EXTRACT TABLE LINES
    ###############################################
    h = np.copy(threshold)
    v = np.copy(threshold)


    # get horizontal lines
    k = np.ones((1, 47), np.uint8)

    h = cv.erode(h, k, iterations=3)
    h = cv.dilate(h, k, iterations=3)


    # get vertical lines
    k = np.ones((15,1), np.uint8)

    v = cv.erode(v, k, iterations=2)
    v = cv.dilate(v, k, iterations=2)



    ###############################################
    # CREATE LINE MASK AND FIND CONTOURS
    ###############################################
    line_mask = h + v

    (contours, _) = cv.findContours(line_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)



    # ###############################################
    # EXTRACT TABLE AND OCR
    # ###############################################

    texts = []
    cells = []
    data = {}

    for i in range(len(contours) - 2, -1, -1):
        x, y, w, h = cv.boundingRect(contours[i]) # get contour coordinates
        box = gray_image[y:y + h, x:x + w] # get bounding box
        cells.append((x,y,w,h))
        texts.append(utils.run_tesseract(box, 6, 3)) # extract text from bounding box with Tesseract



    # ###############################################
    # STRUCTURE DATA
    # ###############################################
    for i in range(len(cells) - 1):
        c = cells[i][0], cells[i][2]
        if str(c) in data:
            data[str(c)].append(texts[i])
        elif str((c[0], c[1] + 1)) in data:
            data[str((c[0], c[1] + 1))].append(texts[i])
        else:
            data[str(c)] = [texts[i]]

    json_data = json.dumps(data)

    print(json_data)

    exit()