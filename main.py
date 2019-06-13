import cv2 as cv
import debug
import json
import utils

def getMSEntTable1(original_image):
    """GET FIRST TABLE FROM MS ENTERPRISE SWO DOC AND BUILD JSON STRUCTURE"""
    ###############################################
    # CONVERT COLORSPACE TO NEGATIVE
    ###############################################
    original_image = cv.resize(original_image, (int(original_image.shape[1] * 0.2), int(original_image.shape[0]*0.2)), interpolation = cv.INTER_AREA)
    gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

    debug.showImage(gray_image, "original grayscale image", 70) ## DEBUG


    ###############################################
    # APPLY ADAPTIVE THRESHOLD AND NEGATIVE
    ###############################################
    threshold = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)    
    threshold = cv.bitwise_not(threshold)

    debug.showImage(threshold, "image with applied threshold", 70) ## DEBUG


    ###############################################
    # EXTRACT TABLE LINES
    ###############################################
    horizontal, vertical = utils.extractTableLines(threshold, 150, 30)    


    ###############################################
    # CREATE LINE MASK AND FIND EXTERNAL CONTOURS
    ###############################################
    line_mask = horizontal + vertical
    debug.showImage(line_mask, "table lines", 70) ## DEBUG

    table_ctrs, _ = cv.findContours(line_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # table outlines    
    debug.showContours(table_ctrs, 70) ## DEBUG


    # ###############################################
    # EXTRACT TABLES AND INDIVIDUAL CELLS AND OCR
    # ###############################################
    texts = []
    cells = []
    data = {}
    table_num = 1

    table_ctrs = utils.sortContours(table_ctrs, cv.boundingRect(line_mask)[2])
    debug.showContoursIter(table_ctrs, 80) ## DEBUG

    # for each table outline contour, get cell contours then perform OCR
    for table_ctr in table_ctrs:
        x, y, w, h = cv.boundingRect(table_ctr)
        table_bbox = gray_image[y - 1:y + h + 1, x - 1:x + w + 1]

        cell_ctrs = utils.getCellContours(table_bbox, w)
        
        key = "table {}".format(table_num)
        data[key] = []
        visited_r = []
        visited_c = []

        for cell_ctr in cell_ctrs:
            x, y, w, h = cv.boundingRect(cell_ctr)
            cell_bbox = table_bbox[y:y + h, x:x + w]

            # detect headers for specific table style
            if cv.mean(cell_bbox)[0] < 155:
                _, cell_bbox = cv.threshold(cell_bbox, 200, 255, cv.THRESH_BINARY_INV)

            # logic to differentiate different rows and cells
            if y not in visited_r:
                visited_r.append(y)
            
            if x not in visited_c:
                visited_c.append(x)

            # signifiy if OCR returned empty string
            v = utils.run_tesseract(cell_bbox, 3, 3)
            if v == "":
                v = "NULL"

            data[key].append((visited_r.index(y) + 1, visited_c.index(x) + 1, v))

        table_num += 1


    # #############################################
    # GET NON TABLE DATA
    # #############################################
    data["non-tabular data"] = utils.getNonTabularData(gray_image, table_ctrs)


    # #############################################
    # FORMAT JSON
    # #############################################
    json_data = json.dumps(data, indent=3)

    print(json_data)

