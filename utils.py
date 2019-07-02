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
    if len(text.strip()) == 0:
        configuration += " -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-."
        text = tess.image_to_string(image, lang=language, config=configuration)

    return text



def deskewImage(thresh):
    """FIND SKEW ANGLE ON IMAGE, DESKEW, AND RETURN"""    
    edges = cv.Canny(thresh, 100, 100, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    angles = []

    for x1, y1, x2, y2 in lines[0]:
        cv.line(thresh, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)

    return ndimage.rotate(thresh, median_angle)



def getTextOrientationAngle(image):
    """GET ORIENTATION ANGLE OF TEXT IN DOCUMENT"""
    data = tess.image_to_osd(image)
    print (data)
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



def getNonTabularData(image, table_outline_ctrs):
    """EXTRACTS ALL DATA THAT IS NOT IN TABLE"""
    text_only = image

    # draw white filled rectangles on image
    for table_outline in table_outline_ctrs:
        x, y, w, h = cv.boundingRect(table_outline)
        cv.rectangle(text_only, (x, y), (x + w, y + h), (255, 255, 255), cv.FILLED)

    return run_tesseract(text_only, 3, 3)



def getCellContours(table_bbox, w, h):
    """RETURNS TABLE OUTLINE BOUNDING BOX AND CELL CONTOURS"""
    table_bbox = cv.adaptiveThreshold(table_bbox, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    table_bbox = cv.bitwise_not(table_bbox)

    h, v = extractTableLines(table_bbox, math.floor(w * 0.5), math.floor(h * 0.3)) # get lines 0.15 0.3
    cell_ctrs, _ = cv.findContours(h + v, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # get cell contours
    cell_ctrs = removeFlatContours(cell_ctrs)

    return sortContours(cell_ctrs[1:], w)



def removeFlatContours(ctrs):
    """REMOVES FLAT CONTOURS FROM CONTOUR LIST"""
    return [ctr for ctr in ctrs if cv.boundingRect(ctr)[3] >= 10 and cv.boundingRect(ctr)[2] >= 10]



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