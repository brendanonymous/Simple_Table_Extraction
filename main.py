import cv2 as cv
import json
import utils

def getMSEntTable1(original_image):
    """GET FIRST TABLE FROM MS ENTERPRISE SWO DOC AND BUILD JSON STRUCTURE"""
    ###############################################
    # CONVERT COLORSPACE TO NEGATIVE
    ###############################################
    original_image = cv.resize(original_image, (int(original_image.shape[1] * 0.2), int(original_image.shape[0]*0.2)), interpolation = cv.INTER_AREA)
    gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

    utils.showImage(gray_image, "original grayscale image", 70) ## DEBUG


    ###############################################
    # APPLY ADAPTIVE THRESHOLD AND NEGATIVE
    ###############################################
    threshold = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 1)    
    threshold = cv.bitwise_not(threshold)

    utils.showImage(threshold, "image with applied threshold", 70) ## DEBUG


    ###############################################
    # EXTRACT TABLE LINES
    ###############################################
    horizontal, vertical = utils.extractTableLines(threshold, 90, 30)

    utils.showImage(horizontal, "horizontal lines", 70) ## DEBUG
    utils.showImage(vertical, "vertical lines", 70) ## DEBUG


    ###############################################
    # CREATE LINE MASK AND FIND CONTOURS
    ###############################################
    line_mask = horizontal + vertical

    utils.showImage(line_mask, "table lines", 70) ## DEBUG

    (contours, _) = cv.findContours(line_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    utils.showContours(contours, 70) ## DEBUG


    # ###############################################
    # EXTRACT TABLE AND OCR
    # ###############################################
    texts = []
    cells = []
    data = {}

    for i in range(len(contours) - 2, -1, -1):
        x, y, w, h = cv.boundingRect(contours[i]) # get contour coordinates
        box = gray_image[y:y + h, x:x + w] # get bounding box
        if cv.mean(box)[0] < 155: # detect headers for specific table style
            _, box = cv.threshold(box, 200, 255, cv.THRESH_BINARY_INV)
        cells.append([x,w])
        texts.append(utils.run_tesseract(box, 3, 3)) # OCR w/in bounding box


    # #############################################
    # EXTRACT NON TABLE DATA AND OCR
    # #############################################
    texts.append(utils.extractNonTableData(gray_image, contours))


    # ###############################################
    # STRUCTURE DATA
    # ###############################################
    if len(cells) > 0:
        for i in range(len(cells) - 1):
            c = cells[i]
            if str(c) in data:
                data[str(c)].append(texts[i])
            elif str((c[0], c[1] + 1)) in data:
                data[str((c[0], c[1] + 1))].append(texts[i])
            else:
                data[str(c)] = [texts[i]]

    data["non-table data"] = texts[-1]

    json_data = json.dumps(data, indent=3)

    print(json_data)



def getMSEntTable2(original_image):    
    """GET SECOND TABLE FROM MS ENTERPRISE SWO DOC AND BUILD JSON STRUCTURE"""
    ###################################
    # CONVERT COLORSPACE TO NEGATIVE
    ###################################
    original_image = cv.resize(original_image, (int(original_image.shape[1] * 0.2), int(original_image.shape[0]*0.2)), interpolation = cv.INTER_AREA)
    gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

    utils.showImage(gray_image, "original grayscale image", 70) ## DEBUG


    ###############################################
    # APPLY ADAPTIVE THRESHOLD AND NEGATIVE
    ###############################################
    threshold = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 1)    
    threshold = cv.bitwise_not(threshold) # negative

    utils.showImage(threshold, "image with applied threshold", 70) ## DEBUG


    ###############################################
    # EXTRACT TABLE LINES
    ###############################################
    horizontal, vertical = utils.extractTableLines(threshold, 340, 30)

    utils.showImage(horizontal, "horizontal lines", 70) ## DEBUG
    utils.showImage(vertical, "vertical lines", 70) ## DEBUG


    ###############################################
    # CREATE LINE MASK AND FIND CONTOURS
    ###############################################
    line_mask = horizontal + vertical

    utils.showImage(line_mask, "table lines", 70) ## DEBUG

    (contours, _) = cv.findContours(line_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    utils.showContours(contours, 70) ## DEBUG


    # ###############################################
    # EXTRACT TABLE AND OCR
    # ###############################################
    texts = []
    cells = []
    data = {}

    for i in range(len(contours) - 1, 0, -1):
        x, y, w, h = cv.boundingRect(contours[i]) # get contour coordinates
        box = gray_image[y:y + h, x:x + w] # get bounding box
        if cv.mean(box)[0] < 155: # this will detect headers for this style of table
            _, box = cv.threshold(box, 200, 255, cv.THRESH_BINARY_INV)
        cells.append((x,y))
        texts.append(utils.run_tesseract(box, 3, 3)) # extract text from bounding box with Tesseract


    # #############################################
    # EXTRACT NON TABLE DATA AND OCR
    # #############################################
    texts.append(utils.extractNonTableData(gray_image, contours))


    # ###############################################
    # STRUCTURE DATA
    # ###############################################
    if len(cells) > 0:
        data[str(cells[0])] = [texts[0]]
        for i in range(1, len(cells) - 1):
            data[str(cells[0])].append(texts[i])

    data["non-table data"] = texts[-1] # add the non-table data

    json_data = json.dumps(data, indent=3)

    print(json_data)



def getMSEntTable3(original_image):
    """GET THIRD TABLE FROM MS ENTERPRISE SWO DOC AND BUILD JSON STRUCTURE"""
    ###############################################
    # CONVERT COLORSPACE TO NEGATIVE
    ###############################################
    original_image = cv.resize(original_image, (int(original_image.shape[1] * 0.2), int(original_image.shape[0]*0.2)), interpolation = cv.INTER_AREA)
    gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

    utils.showImage(gray_image, "original grayscale image", 70) ## DEBUG


    ###############################################
    # APPLY ADAPTIVE THRESHOLD AND NEGATIVE
    ###############################################
    threshold = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 1)    
    threshold = cv.bitwise_not(threshold) # negative

    utils.showImage(threshold, "image with applied threshold", 70) ## DEBUG


    ###############################################
    # EXTRACT TABLE LINES
    ###############################################
    horizontal, vertical = utils.extractTableLines(threshold, 150, 50)

    utils.showImage(horizontal, "horizontal lines", 70) ## DEBUG
    utils.showImage(vertical, "vertical lines", 70) ## DEBUG


    ###############################################
    # CREATE LINE MASK AND FIND CONTOURS
    ###############################################
    line_mask = horizontal + vertical

    utils.showImage(line_mask, "table lines", 70) ## DEBUG

    (contours, _) = cv.findContours(line_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    utils.showContours(contours, 70) ## DEBUG


    # ###############################################
    # EXTRACT TABLE AND OCR
    # ###############################################
    texts = []
    cells = []
    data = {}

    for i in range(len(contours) - 1, -1, -1):
        x, y, w, h = cv.boundingRect(contours[i]) # get contour coordinates
        box = gray_image[y:y + h, x:x + w] # get bounding box
        if cv.mean(box)[0] < 155: # this will detect headers for this style of table
            _, box = cv.threshold(box, 200, 255, cv.THRESH_BINARY_INV)       
        cells.append([x,w])
        texts.append(utils.run_tesseract(box, 3, 3)) # extract text from bounding box with Tesseract


    # #############################################
    # EXTRACT NON TABLE DATA AND OCR
    # #############################################
    texts.append(utils.extractNonTableData(gray_image, contours))

    
    # ###############################################
    # STRUCTURE DATA
    # ###############################################
    if len(cells) > 0:
        for i in range(len(cells) - 1):
            c = cells[i]
            if c[1] in list(range(107, 112)):
                c[1] = 109
            c = tuple(c)
            if str(c) in data:
                data[str(c)].append(texts[i])
            else:
                data[str(c)] = [texts[i]]
    
    data["non-table data"] = texts[-1] # add the non-table data

    json_data = json.dumps(data, indent=3)

    print(json_data)


 











# def getSample4Table1(original_image):
#     """GET TABLE 1 FROM SAMPLE 4.pdf DOC AND BUILD JSON STRUCTURE"""
#     ###################################
#     # CONVERT COLORSPACE TO NEGATIVE
#     ###################################
#     original_image = cv.resize(original_image, (int(original_image.shape[1] * 0.2), int(original_image.shape[0]*0.2)), interpolation = cv.INTER_AREA)
#     gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)


#     ###############################################
#     # APPLY ADAPTIVE THRESHOLD AND NEGATIVE
#     ###############################################
#     threshold = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 1)    
#     threshold = cv.bitwise_not(threshold) # negative


#     ###############################################
#     # EXTRACT TABLE LINES
#     ###############################################
#     horizontal, vertical = utils.extractTableLines(threshold, 340, 30)


#     ###############################################
#     # CREATE LINE MASK AND FIND CONTOURS
#     ###############################################
#     line_mask = horizontal + vertical

#     (contours, _) = cv.findContours(line_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


#     # ###############################################
#     # EXTRACT TABLE AND OCR
#     # ###############################################
#     texts = []
#     cells = []
#     data = {}

#     for i in range(len(contours) - 1, -1, -1):
#         x, y, w, h = cv.boundingRect(contours[i]) # get contour coordinates
#         box = gray_image[y:y + h, x:x + w] # get bounding box
#         if cv.mean(box)[0] < 155: # this will detect headers for this style of table
#             _, box = cv.threshold(box, 200, 255, cv.THRESH_BINARY_INV)
#         cells.append((x,w))
#         texts.append(utils.run_tesseract(box, 3, 3)) # extract text from bounding box with Tesseract


#     # #############################################
#     # EXTRACT NON TABLE DATA AND OCR
#     # #############################################
#     texts.append(utils.extractNonTableData(gray_image, contours))

    
    # # ###############################################
    # # STRUCTURE DATA
    # # ###############################################
    # if len(cells) > 0:
    #     for i in range(len(cells) - 1):
    #         c = cells[i]
    #         if str(c) in data:
    #             data[str(c)].append(texts[i])
    #         else:
    #             data[str(c)] = [texts[i]]

#     data["non-table data"] = texts[-1]

#     json_data = json.dumps(data, indent=3)

#     print(json_data)



# def getSample7Table1(original_image):
    # """GET TABLE 1 FROM SAMPLE 7.pdf DOC AND BUILD JSON STRUCTURE"""
    # ###################################
    # # CONVERT COLORSPACE TO NEGATIVE
    # ###################################
    # original_image = cv.resize(original_image, (int(original_image.shape[1] * 0.2), int(original_image.shape[0]*0.2)), interpolation = cv.INTER_AREA)
    # gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)


    # ###############################################
    # # APPLY ADAPTIVE THRESHOLD AND NEGATIVE
    # ###############################################
    # threshold = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 1)    
    # threshold = cv.bitwise_not(threshold) # negative


    # ###############################################
    # # EXTRACT TABLE LINES
    # ###############################################    
    # horizontal, vertical = utils.extractTableLines(threshold, 340, 30)


    # ###############################################
    # # CREATE LINE MASK AND FIND CONTOURS
    # ###############################################
    # line_mask = horizontal + vertical

    # (contours, _) = cv.findContours(line_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


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
    #     cells.append((x,w))
    #     texts.append(utils.run_tesseract(box, 3, 3)) # extract text from bounding box with Tesseract

    
    # # #############################################
    # # EXTRACT NON TABLE DATA AND OCR
    # # #############################################
    # texts.append(utils.extractNonTableData(gray_image, contours))


    # # ###############################################
    # # STRUCTURE DATA
    # # ###############################################
    # if len(cells) > 0:
    #   for i in range(len(cells) - 1):
    #       c = cells[i]
    #       print (c)
    #       if str(c) in data:
    #            data[str(c)].append(texts[i])
    #       else:
    #           data[str(c)] = [texts[i]]

    # data["non-table data"] = texts[-1]

    # json_data = json.dumps(data, indent=3)

    # print(json_data)