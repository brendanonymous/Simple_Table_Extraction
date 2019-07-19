import cv2 as cv
import debug
import math
import numpy as np
import pytesseract as tess
import re
from scipy import ndimage
from wand.image import Image



def pdfToJpg(fp):
    """CONVERT PDF TO IMAGE, SAVE, AND RETURN NUMBER OF PAGES"""
    with(Image(filename=fp, resolution=500)) as source: 
        images = source.sequence
        pages = len(images)
        for i in range(pages):
            n = i + 1
            newfilename = "out" + str(n) + '.JPG'
            Image(images[i]).save(filename=newfilename)
    
    return pages
    # t0 = time.time()

    # images = convert_from_path(fp, dpi=500, output_folder=os.getcwd(), fmt="JPEG", thread_count=4)

    
    # pageNum = 1
    # for image in images:
    #     image.save("out{}".format(pageNum), "JPEG")
    #     pageNum += 1
    
    
    # t1 = time.time()
    
    # total = t1-t0
    
    
    # # return len(pages)
    # print(total)
    # exit()



def run_tesseract(image, psm, oem):
    """RUNS TESSERACT TO PERFORM OCR"""
    language = 'eng'
    configuration = "--psm " + str(psm) + " --oem " + str(oem)

    # Run tesseract
    text = tess.image_to_string(image, lang=language, config=configuration)
    # data = tess.image_to_data(image, output_type='data.frame')
    # data = data[data.conf != -1]
    # lines = data.groupby('block_num')['text'].apply(list)
    # conf = data.groupby(['block_num'])['conf'].mean()

    # print (conf)
    if len(text.strip()) == 0:
        configuration += " -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-."
        text = tess.image_to_string(image, lang=language, config=configuration)

    return text



def getLineAndTextThresholds(image):
    """APPLY THRESHOLDS FOR LINE EXTRACTION AND TEXT READABILITY"""
    line_thresh = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 3)
    line_thresh = cv.bitwise_not(line_thresh)
    line_thresh = cv.morphologyEx(line_thresh, cv.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    _, text_thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
    text_thresh = cv.bitwise_not(text_thresh)
    
    return line_thresh, text_thresh



def deskewImage(image):
    """FIND SKEW ANGLE ON IMAGE, DESKEW, AND RETURN"""
    edges = cv.Canny(image, 100, 100, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    if lines is not None:
        img_copy = image    
        angles = []

        for x1, y1, x2, y2 in lines[0]:
            cv.line(img_copy, (x1, y1), (x2, y2), (0, 0, 0), 1) # do we need this
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)

        median_angle = np.median(angles)

        return ndimage.rotate(image, median_angle)
    
    return image


def getTextOrientationAngle(image):
    """GET ORIENTATION ANGLE OF TEXT IN DOCUMENT"""
    data = tess.image_to_osd(image)
    rotation = re.search('(?<=Rotate: )\d+', data).group(0)
    return int(rotation)



def rotateImage(image, angle):
    """ROTATE AN IMAGE BY A GIVEN ANGLE"""
    if angle != 360:
        center = (image.shape[1] // 2, image.shape[0] // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)

        return cv.warpAffine(image, M, (image.shape[0], image.shape[1]))

    return image



def extractTableLines(image, horizontal_kernel_size, vertical_kernel_size):
    """EXTRACTS HORIZONTAL AND VERTICAL TABLE LINES FROM AN IMAGE"""
    horizontal_lines = np.copy(image)
    vertical_lines = np.copy(image)

    # get horizontal lines
    k = np.ones((1, horizontal_kernel_size), np.uint8)

    horizontal_lines = cv.erode(horizontal_lines, k)
    horizontal_lines = cv.dilate(horizontal_lines, k)

    # get vertical lines
    k = np.ones((vertical_kernel_size, 1), np.uint8)

    vertical_lines = cv.erode(vertical_lines, k)
    vertical_lines = cv.dilate(vertical_lines, k)

    return horizontal_lines, vertical_lines



def extractCircles(image):
    bilateral_filtered_image = cv.bilateralFilter(image, 5, 175, 175)
    edge_detected_image = cv.Canny(bilateral_filtered_image, 75, 200)
    contours, _= cv.findContours(edge_detected_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for contour in contours:
        approx = cv.approxPolyDP(contour,0.01*cv.arcLength(contour,True),True)
        area = cv.contourArea(contour)
        if ((len(approx) > 21) & (area > 30) ):
            contour_list.append(contour)
    
    



def getNonTabularData(image, table_outline_ctrs):
    """EXTRACTS ALL DATA THAT IS NOT IN TABLE"""
    text_only = image

    # draw white filled rectangles on image
    for table_outline in table_outline_ctrs:
        x, y, w, h = cv.boundingRect(table_outline)
        cv.rectangle(text_only, (x, y), (x + w, y + h), (255, 255, 255), cv.FILLED)

    return run_tesseract(text_only, 3, 3)



def getTableContours(line_mask):
    """RETURNS TABLE OUTLINE CONTOURS FROM LINE MASK"""
    table_ctrs, _ = cv.findContours(line_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # table outlines
    table_ctrs = removeFlatContours(table_ctrs)
    table_ctrs = sortContours(table_ctrs, cv.boundingRect(line_mask)[2]) # sort contours left-to-right, top-to-bottom
    
    return table_ctrs



def getCellContours(table_bbox, w, h):
    """RETURNS CELL CONTOURS FOUND WITHIN TABLE BOUNDING BOX"""
    table_bbox = cv.adaptiveThreshold(table_bbox, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 1)
    table_bbox = cv.bitwise_not(table_bbox)    
    # table_bbox = cv.morphologyEx(table_bbox, cv.MORPH_CLOSE, np.ones((6, 6), np.uint8))
    # debug.showImage(table_bbox, "table bbox", 80)# DEBUG

    h, v = extractTableLines(table_bbox, math.floor(w * 0.2), math.floor(h * 0.15))
    debug.showImage(h+v, "line mask", 150)# DEBUG

    cell_ctrs, _ = cv.findContours(h + v, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # get cell contours
    cell_ctrs = removeFlatContours(cell_ctrs)
    cell_ctrs = sortContours(cell_ctrs[1:], w)

    return cell_ctrs



def removeFlatContours(ctrs):
    """REMOVES FLAT CONTOURS FROM CONTOUR LIST"""
    return [ctr for ctr in ctrs if len(cv.approxPolyDP(ctr, 0.01*cv.arcLength(ctr, True), True)) == 4 and \
                                                                        cv.boundingRect(ctr)[3] >= 10 and \
                                                                        cv.boundingRect(ctr)[2] >= 10]



def sortContours(ctrs, w):
    """RETURNS LIST OF CONTOURS SORTED LEFT-TO-RIGHT AND TOP-TO-BOTTOM"""
    return sorted(ctrs, key=lambda c: cv.boundingRect(c)[0] + cv.boundingRect(c)[1] * w )



def getTableColumn(columns, x):
    """GETS COLUMN BY APPROXIMATE X COORDINATE"""
    rng = list(range(x - 3, x + 3))
    for n in rng:
        v = columns.get(n, None)
        if v is not None:
            return v
    
    return False



def imageIsEmpty(image):
    """CHECKS IF IMAGE IS EMPTY OR BLANK"""
    if image is None or np.sum(image) == 0 or np.shape(image) == ():
        return True
    return False