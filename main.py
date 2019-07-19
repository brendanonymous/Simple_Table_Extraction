import cv2 as cv
import debug
import json
import math
import sys
import utils

import numpy as np

def getData_1(orig_img):
    """GENERIC FUNCTION TO GET TABULAR AND NON TABULAR DATA"""
    ###############################################
    # CONVERT COLORSPACE TO NEGATIVE
    ###############################################
    resized_image = cv.resize(orig_img, (orig_img.shape[1] // 5, orig_img.shape[0] // 5), interpolation = cv.INTER_AREA) # might have a lot of overhead depending on img size
    gray_image = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)

    debug.showImage(resized_image, "resized_image", 70)
    debug.showImage(gray_image, "gray", 80)

    ###############################################
    # DESKEW IMAGE
    ###############################################
    # deskewed = utils.deskewImage(orig_img) # thresh
    # debug.showImage(deskewed, "deskew", 80)# DEBUG


    ###############################################
    # CHECK ORIENTATION ON TEXT AND ROTATE
    ###############################################
    # orientationAngle = utils.getTextOrientationAngle(deskewed)
    # rotated = utils.rotateImage(gray_image, 360 - orientationAngle)
    
    # debug.showImage(rotated, "rotated", 80)# DEBUG


    ###############################################
    # APPLY THRESHOLDS
    ###############################################
    line_thresh, text_thresh = utils.getLineAndTextThresholds(gray_image)

    debug.showImage(line_thresh, "line thresh", 80)# DEBUG
    debug.showImage(text_thresh, "text thresh", 80)# DEBUG


    ###############################################
    # EXTRACT TABLE LINES
    ###############################################
    horizontal, vertical = utils.extractTableLines(line_thresh, math.floor(line_thresh.shape[1] * 0.3), math.floor(line_thresh.shape[0] * 0.025))


    ###############################################
    # CREATE LINE MASK AND FIND TABLE CONTOURS
    ###############################################
    line_mask = horizontal + vertical
    debug.showImage(line_mask, "isolated lines", 80)  

    table_ctrs = utils.getTableContours(line_mask)
    debug.showContours(table_ctrs, "table ctrs", 70)# DEBUG


    # ###############################################
    # EXTRACT TABLES AND INDIVIDUAL CELLS AND OCR
    # ###############################################
    data = {}
    table_num = 1

    # for each table outline contour, get cell contours then perform OCR
    for table_ctr in table_ctrs:
        x, y, w, h = cv.boundingRect(table_ctr)
        tableLines_bbox = cv.bitwise_not(line_thresh[y - 5:y + h + 5, x - 5:x + w + 5])
        tableText_bbox = cv.bitwise_not(text_thresh[y - 5:y + h + 5, x - 5:x + w + 5])

        debug.showImage(tableLines_bbox, "tableLines_bbox")# DEBUG
        debug.showImage(tableText_bbox, "tableText_bbox")# DEBUG
        

        cell_ctrs = utils.getCellContours(tableLines_bbox, tableLines_bbox.shape[1], tableLines_bbox.shape[0])

        debug.showContours(cell_ctrs, title="cell ctrs", scalePercent=80)# DEBUG
        
        key = "table {}".format(table_num)
        data[key] = {}
        visited_rows = []
        visited_cols = {} # key is y coord of cell, value is column number
        row = 0
        col = 1

        for cell_ctr in cell_ctrs:
            x, y, w, h = cv.boundingRect(cell_ctr)
            # cell_bbox = table_bbox[y:y + h, x:x + w]
            cell_bbox = tableText_bbox[y:y + h, x:x + w] # gray_image

            # detect headers for specific table style
            if cv.mean(cell_bbox)[0] < 155:
                _, cell_bbox = cv.threshold(cell_bbox, 200, 255, cv.THRESH_BINARY_INV)
            
            # debug.showImage(cell_bbox, "cell bbox")# DEBUG

            # logic to differentiate different rows and cells
            if y not in visited_rows:
                visited_rows.append(y)
                row += 1
                data[key]["row " + str(row)] = []
            if not utils.getTableColumn(visited_cols, x):
                visited_cols[x] = col
                col += 1

            # signifiy if OCR returned empty string
            v = utils.run_tesseract(cell_bbox, 3, 3) # was 3
            if v == "":
                v = "--EMPTY--"

            data[key]["row " + str(row)].append(("col " + str(utils.getTableColumn(visited_cols, x)), v))

        table_num += 1


    # #############################################
    # GET NON TABLE DATA
    # #############################################
    data["non-tabular data"] = utils.getNonTabularData(gray_image, table_ctrs)


    # #############################################
    # FORMAT JSON
    # #############################################
    json_data = json.dumps(data, indent=3, ensure_ascii=False)

    print(json_data)
