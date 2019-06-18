import cv2 as cv
import debug
import json
import math
import utils

def getData_1(orig_img):
    """GENERIC FUNCTION TO GET TABULAR AND NON TABULAR DATA"""
    ###############################################
    # CONVERT COLORSPACE TO NEGATIVE
    ###############################################
    orig_img = cv.resize(orig_img, (orig_img.shape[1] // 5, orig_img.shape[0] // 5), interpolation = cv.INTER_AREA) # might have a lot of overhead depending on img size
    gray_image = cv.cvtColor(orig_img, cv.COLOR_BGR2GRAY)


    ###############################################
    # APPLY ADAPTIVE THRESHOLD AND NEGATIVE
    ###############################################
    thresh = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)    
    thresh = cv.bitwise_not(thresh)


    ###############################################
    # DESKEW IMAGE
    ###############################################
    debug.showImage(thresh, "original", 80)# DEBUG
    thresh = utils.deskewImg(thresh)
    debug.showImage(thresh, "deskewed", 80)# DEBUG


    ###############################################
    # EXTRACT TABLE LINES
    ###############################################
    horizontal, vertical = utils.extractTableLines(thresh, math.floor(thresh.shape[1] * 0.20), math.floor(thresh.shape[0] * 0.026))


    ###############################################
    # CREATE LINE MASK AND FIND EXTERNAL CONTOURS
    ###############################################
    line_mask = horizontal + vertical

    table_ctrs, _ = cv.findContours(line_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # table outlines
    table_ctrs = utils.removeFlatContours(table_ctrs)


    # ###############################################
    # EXTRACT TABLES AND INDIVIDUAL CELLS AND OCR
    # ###############################################
    data = {}
    table_num = 1

    table_ctrs = utils.sortContours(table_ctrs, cv.boundingRect(line_mask)[2]) # sort contours left-to-right, top-to-bottom

    # for each table outline contour, get cell contours then perform OCR
    for table_ctr in table_ctrs:
        x, y, w, h = cv.boundingRect(table_ctr)
        table_bbox = gray_image[y - 1:y + h + 1, x - 1:x + w + 1]

        cell_ctrs = utils.getCellContours(table_bbox, w, h)
        
        key = "table {}".format(table_num)
        data[key] = {}
        visited_rows = []
        row = 0

        for cell_ctr in cell_ctrs:
            x, y, w, h = cv.boundingRect(cell_ctr)
            cell_bbox = table_bbox[y:y + h, x:x + w]

            # detect headers for specific table style
            if cv.mean(cell_bbox)[0] < 155:
                _, cell_bbox = cv.threshold(cell_bbox, 200, 255, cv.THRESH_BINARY_INV)

            # logic to differentiate different rows and cells
            if y not in visited_rows:
                visited_rows.append(y)
                row += 1                
                col = 1
                data[key]["row " + str(row)] = []
            else:
                col += 1

            # signifiy if OCR returned empty string
            v = utils.run_tesseract(cell_bbox, 3, 3)
            if v == "":
                v = "--EMPTY--"

            data[key]["row " + str(row)].append(("col " + str(col), v))

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

    exit()