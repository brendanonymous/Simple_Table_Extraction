
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
        configuration += " -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        text = tess.image_to_string(image, lang=language, config=configuration)

    return text



def extractTableLines(image):
    """EXTRACTS HORIZONTAL AND VERTICAL TABLE LINES FROM AN IMAGE"""
    horizontal_lines = np.copy(image)
    vertical_lines = np.copy(image)

    # get horizontal lines
    cols = horizontal_lines.shape[1]
    horizontal_lines_size = cols // 30

    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_lines_size, 1))

    horizontal_lines = cv.erode(horizontal_lines, horizontal_kernel)
    horizontal_lines = cv.dilate(horizontal_lines, horizontal_kernel)

    # get vertical lines
    rows = vertical_lines.shape[0]
    vertical_lines_size = rows // 30

    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_lines_size))
    
    vertical_lines = cv.erode(vertical_lines, vertical_kernel)    
    vertical_lines = cv.dilate(vertical_lines, vertical_kernel)

    return horizontal_lines, vertical_lines



def showImage(image, title):
    """SHOW IMAGE FOR DEBUGGING"""
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()



def showResizedImage(image, scalePercent, title):
    """RESIZE IMAGE THEN SHOW FOR DEBUGGING"""
    newWidth = int(image.shape[1] * scalePercent / 100)
    newHeight = int(image.shape[0] * scalePercent / 100)
    dimension = (newWidth, newHeight)
    resizedImage = cv.resize(image, dimension, interpolation=cv.INTER_AREA)

    showImage(resizedImage, title)



def showContours(contours):
    """SHOW CONTOURS FOR DEBUGGING"""
    contours_image = np.zeros((720,1280,3), np.uint8)
    cv.drawContours(contours_image, contours, -1, (0,255,0), 2)
    showImage(contours_image, "contours")



def showContoursOnImage(contours, image):
    """SHOW CONTOURS ITERATIVELY ON IMAGE FOR DEBUGGING"""
    for i in range(len(contours) - 1, -1, -1):
        img = image
        cv.drawContours(img, contours, i, (0, 0, 255), 1)
        showImage(img, "image with contour")



def showContoursIter(contours):
    """SHOW CONTOURS ITERATIVELY ON A BLACK CANVAS FOR DEBUGGING"""
    contours_image = np.zeros((720,1280,3), np.uint8)
    for i in range(len(contours) - 1, -1, -1):
        cv.drawContours(contours_image, contours, i, (0,255,0), 2)
        showImage(contours_image, "Contours")



# convert pdf to image
def pdfToJpg(path):
    """CONVERT PDF TO IMAGE, SAVE, AND RETURN NUMBER OF PAGES"""
    pages = convert_from_path(path, 500)
    pageNum = 1
    for page in pages:
        page.save("out{}.jpg".format(pageNum), "JPEG")
        pageNum += 1
    
    return len(pages)