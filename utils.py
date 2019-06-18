import cv2 as cv
import debug
import math
import numpy as np
from pdf2image import convert_from_path
import pytesseract as tess


def run_tesseract(image, psm, oem):
    """RUNS TESSERACT TO PERFORM OCR"""
    language = 'eng'
    configuration = "--psm " + str(psm) + " --oem " + str(oem)

    # Run tesseract
    text = tess.image_to_string(image, lang=language, config=configuration)
    if len(text.strip()) == 0:
        configuration += " -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-."
        text = tess.image_to_string(image, lang=language, config=configuration)

    return text



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



def getNonTabularData(image, table_outline_ctrs):
    """EXTRACTS ALL DATA NOT IN TABLE"""
    text_only = image

    # draw white rectangles on image
    for table_outline in table_outline_ctrs:
        x, y, w, h = cv.boundingRect(table_outline)
        cv.rectangle(text_only, (x, y), (x + w, y + h), (255, 255, 255), cv.FILLED)

    return run_tesseract(text_only, 3, 3)



def getCellContours(table_bbox, w, h):
    """RETURNS TABLE OUTLINE BOUNDING BOX AND CELL CONTOURS"""
    table_bbox = cv.adaptiveThreshold(table_bbox, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    table_bbox = cv.bitwise_not(table_bbox)

    h, v = extractTableLines(table_bbox, math.floor(w * 0.9), math.floor(h * 0.3)) # get lines
    cell_ctrs, _ = cv.findContours(h + v, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # get cell contours
    debug.showContours(cell_ctrs)

    return sortContours(cell_ctrs[1:], w)



def removeFlatContours(ctrs):
    """REMOVES FLAT CONTOURS FROM CONTOUR LIST"""
    return [ctr for ctr in ctrs if cv.boundingRect(ctr)[3] >= 10 and cv.boundingRect(ctr)[2] >= 10]



def sortContours(ctrs, w):
    """RETURNS LIST OF CONTOURS SORTED LEFT-TO-RIGHT AND TOP-TO-BOTTOM"""
    return sorted(ctrs, key=lambda c: cv.boundingRect(c)[0] + cv.boundingRect(c)[1] * w )



def pdfToJpg(path):
    """CONVERT PDF TO IMAGE, SAVE, AND RETURN NUMBER OF PAGES"""
    pages = convert_from_path(path, 500)
    pageNum = 1
    for page in pages:
        page.save("out{}.jpg".format(pageNum), "JPEG")
        pageNum += 1
    
    return len(pages)



def deskewImg(thresh):
    """FIND SKEW ON IMAGE, DESKEW, AND RETURN DESKEWED IMAGE"""
    # get all coords of white pixels and find a boundary around them
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv.minAreaRect(coords)[-1]
    print(angle)

    # special case
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # get center of image then rotate
    (h, w) = thresh.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)

    return cv.warpAffine(thresh, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)