import cv2 as cv
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



def extractNonTableData(image, contours):
    text_only = image
    if len(contours) != 0:
        x, y, w, h = cv.boundingRect(contours[0])
        cv.rectangle(text_only, (x, y), (x + w, y + h), (255, 255, 255), cv.FILLED)

    return run_tesseract(text_only, 3, 3)



def showImageHelper(image, title):
    """SHOW IMAGE FOR DEBUGGING"""
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()



def showImage(image, title, scalePercent=100):
    """RESIZE IMAGE THEN SHOW FOR DEBUGGING"""
    newWidth = int(image.shape[1] * scalePercent / 100)
    newHeight = int(image.shape[0] * scalePercent / 100)
    dimension = (newWidth, newHeight)
    resizedImage = cv.resize(image, dimension, interpolation=cv.INTER_AREA)

    showImageHelper(resizedImage, title)



def drawLine(size):
    canvas = np.zeros((4135, 5847, 3), np.uint8)
    y = 20
    x = 20
    cv.line(canvas, (y, x), (y + size, x), (0, 255, 0), 2)
    cv.line(canvas, (y, x), (y, x + size), (0, 255, 0), 2)
    showImage(canvas, "thing")



def showContours(contours, scalePercent=100):
    """SHOW CONTOURS FOR DEBUGGING"""
    contours_image = np.zeros((4135,5847,3), np.uint8)
    cv.drawContours(contours_image, contours, -1, (97, 204, 44), 2) 
    showImage(contours_image, "contours", scalePercent)



def showContoursIter(contours, scalePercent=100, reverse=False):
    """SHOW CONTOURS ITERATIVELY ON A BLACK CANVAS FOR DEBUGGING"""
    contours_image = np.zeros((4135,5847,3), np.uint8)
    if reverse:
        for i in range(len(contours) - 1, -1, -1):
            cv.drawContours(contours_image, contours, i, (0,255,0), 2)
            showImage(contours_image, "Contours", scalePercent)
    else:
        for i in range(len(contours)):
            cv.drawContours(contours_image, contours, i, (0, 255, 0), 2)
            showImage(contours_image, "Contours", scalePercent)



def showContoursOnImage(contours, image, scalePercent=100):
    """SHOW CONTOURS ITERATIVELY ON IMAGE FOR DEBUGGING"""
    for i in range(len(contours) - 1, -1, -1):
        img = image
        cv.drawContours(img, contours, i, (0, 0, 255), 1)
        showImage(img, "image with contour", scalePercent)


# convert pdf to image
def pdfToJpg(path):
    """CONVERT PDF TO IMAGE, SAVE, AND RETURN NUMBER OF PAGES"""
    pages = convert_from_path(path, 500)
    pageNum = 1
    for page in pages:
        page.save("out{}.jpg".format(pageNum), "JPEG")
        pageNum += 1
    
    return len(pages)