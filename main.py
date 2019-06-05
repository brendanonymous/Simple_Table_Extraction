import cv2 as cv
import json
import math
import numpy as np
from openpyxl import Workbook, load_workbook
import os
import statistics
import sys
from uncertainties import ufloat
import utils

def getMSEntTable1(original_image):
    """GET FIRST TABLE FROM MS ENTERPRISE SWO DOC AND BUILD JSON STRUCTURE"""
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
        texts.append(utils.run_tesseract(box, 3, 3)) # extract text from bounding box with Tesseract



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



def getMSEntTable2(original_image):    
    """GET SECOND TABLE FROM MS ENTERPRISE SWO DOC AND BUILD JSON STRUCTURE"""
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
    k = np.ones((1, 340), np.uint8)

    h = cv.erode(h, k)
    h = cv.dilate(h, k)

    # get vertical lines
    k = np.ones((30,1), np.uint8)

    v = cv.erode(v, k)
    v = cv.dilate(v, k)


    ###############################################
    # CREATE LINE MASK AND FIND CONTOURS
    ###############################################
    line_mask = h + v

    utils.showImage(line_mask, "line mask", 100)

    (contours, _) = cv.findContours(line_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    # ###############################################
    # EXTRACT TABLE AND OCR
    # ###############################################
    texts = []
    cells = []
    data = {}

    for i in range(len(contours) - 1, 0, -1):
        x, y, w, h = cv.boundingRect(contours[i]) # get contour coordinates
        box = gray_image[y:y + h, x:x + w] # get bounding box
        if cv.mean(box)[0] < 155:
            _, box = cv.threshold(box, 200, 255, cv.THRESH_BINARY_INV)
        cells.append((x,y))
        texts.append(utils.run_tesseract(box, 3, 3)) # extract text from bounding box with Tesseract

    # ###############################################
    # STRUCTURE DATA
    # ###############################################
    # reognize header
    data[str(cells[0])] = [texts[0]]
    for i in range(1, len(cells) - 1):
        data[str(cells[0])].append(texts[i])

    json_data = json.dumps(data)

    print(json_data)



def getMSEntTable3(original_image):
    """GET THIRD TABLE FROM MS ENTERPRISE SWO DOC AND BUILD JSON STRUCTURE"""
    ###############################################
    # CONVERT COLORSPACE TO NEGATIVE
    ###############################################
    original_image = cv.resize(original_image, (int(original_image.shape[1] * 0.2), int(original_image.shape[0]*0.2)), interpolation = cv.INTER_AREA)
    gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)


    ###############################################
    # APPLY ADAPTIVE THRESHOLD AND NEGATIVE
    ###############################################
    threshold = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 1)    
    threshold = cv.bitwise_not(threshold) # negative

    utils.showImage(threshold, "threshold", 80)


    ###############################################
    # EXTRACT TABLE LINES
    ###############################################
    h = np.copy(threshold)
    v = np.copy(threshold)

    k = np.ones((1, 150), np.uint8)

    h = cv.erode(h, k)
    h = cv.dilate(h, k)

    utils.showImage(h, "horiz", 80)

    k = np.ones((50, 1), np.uint8)

    v = cv.erode(v, k)
    v = cv.dilate(v, k)

    utils.showImage(v, "vert", 80)


    ###############################################
    # CREATE LINE MASK AND FIND CONTOURS
    ###############################################
    line_mask = h + v

    utils.showImage(line_mask, "line mask", 80)

    (contours, _) = cv.findContours(line_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    # # ###############################################
    # # EXTRACT TABLE AND OCR
    # # ###############################################
    # texts = []
    # cells = []
    # data = {}

    # for i in range(len(contours) - 1, -1, -1):
    #     x, y, w, h = cv.boundingRect(contours[i]) # get contour coordinates
    #     box = gray_image[y:y + h, x:x + w] # get bounding box
    #     if cv.mean(box)[0] < 155:
    #         _, box = cv.threshold(box, 200, 255, cv.THRESH_BINARY_INV)        
    #     # utils.showImage(box, "smidgen", 80)
    #     cells.append([x,w])
    #     texts.append(utils.run_tesseract(box, 3, 3)) # extract text from bounding box with Tesseract

    
    # # ###############################################
    # # STRUCTURE DATA
    # # ###############################################
    # helper = []

    # for i in range(len(cells) - 1):
    #     c = cells[i]
    #     print (c)
    #     if c[1] in list(range(107, 112)):
    #         c[1] = 109
    #     c = tuple(c)
    #     if str(c) in data:
    #         data[str(c)].append(texts[i])
    #     else:
    #         data[str(c)] = [texts[i]]

    # json_data = json.dumps(data)

    # print(json_data)


    ###############################################
    # ISOLATE NON TABLE DATA
    # #############################################
    table_outline_contour = contours[0]
    text_only = threshold
    cv.drawContours(text_only, contours, 0, (0, 0, 0), thickness=cv.FILLED)
    utils.showImage(text_only, "text only", 80)



def getSample4Table1(original_image):
    """GET TABLE 1 FROM SAMPLE 4.pdf DOC AND BUILD JSON STRUCTURE"""
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
    k = np.ones((1, 340), np.uint8)

    h = cv.erode(h, k)
    h = cv.dilate(h, k)

    # get vertical lines
    k = np.ones((30,1), np.uint8)

    v = cv.erode(v, k)
    v = cv.dilate(v, k)


    ###############################################
    # CREATE LINE MASK AND FIND CONTOURS
    ###############################################
    line_mask = h + v

    utils.showImage(line_mask, "line mask", 100)

    (contours, _) = cv.findContours(line_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    # ###############################################
    # EXTRACT TABLE AND OCR
    # ###############################################
    texts = []
    cells = []
    data = {}

    for i in range(len(contours) - 1, -1, -1):
        x, y, w, h = cv.boundingRect(contours[i]) # get contour coordinates
        box = gray_image[y:y + h, x:x + w] # get bounding box
        if cv.mean(box)[0] < 155:
            _, box = cv.threshold(box, 200, 255, cv.THRESH_BINARY_INV)
        cells.append((x,w))
        texts.append(utils.run_tesseract(box, 3, 3)) # extract text from bounding box with Tesseract

    
    # ###############################################
    # STRUCTURE DATA
    # ###############################################
    for i in range(len(cells) - 1):
        c = cells[i]
        print (c)
        if str(c) in data:
            data[str(c)].append(texts[i])
        else:
            data[str(c)] = [texts[i]]

    json_data = json.dumps(data)

    print(json_data)



# WE ARE HERE
def getSample5Table1(original_image):
    """GET TABLE 1 FROM SAMPLE 5.pdf DOC AND BUILD JSON STRUCTURE"""
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
    k = np.ones((1, 340), np.uint8)

    h = cv.erode(h, k)
    h = cv.dilate(h, k)

    # get vertical lines
    k = np.ones((30,1), np.uint8)

    v = cv.erode(v, k)
    v = cv.dilate(v, k)


    ###############################################
    # CREATE LINE MASK AND FIND CONTOURS
    ###############################################
    line_mask = h + v

    utils.showImage(line_mask, "line mask", 100)

    (contours, _) = cv.findContours(line_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    # ###############################################
    # EXTRACT TABLE AND OCR
    # ###############################################
    texts = []
    cells = []
    data = {}

    for i in range(len(contours) - 1, -1, -1):
        x, y, w, h = cv.boundingRect(contours[i]) # get contour coordinates
        box = gray_image[y:y + h, x:x + w] # get bounding box
        if cv.mean(box)[0] < 155:
            _, box = cv.threshold(box, 200, 255, cv.THRESH_BINARY_INV)
        cells.append((x,w))
        texts.append(utils.run_tesseract(box, 3, 3)) # extract text from bounding box with Tesseract

    
    # ###############################################
    # STRUCTURE DATA
    # ###############################################
    for i in range(len(cells) - 1):
        c = cells[i]
        print (c)
        if str(c) in data:
            data[str(c)].append(texts[i])
        else:
            data[str(c)] = [texts[i]]

    json_data = json.dumps(data)

    print(json_data)