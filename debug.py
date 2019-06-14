import cv2 as cv
import numpy as np


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



def showContours(contours, title="Contours", scalePercent=100):
    """SHOW CONTOURS FOR DEBUGGING"""
    contours_image = np.zeros((4135,5847,3), np.uint8)
    cv.drawContours(contours_image, contours, -1, (97, 204, 44), 2) 
    showImage(contours_image, title, scalePercent)



def showContoursIter(contours, scalePercent=100, reverse=False):
    """SHOW CONTOURS ITERATIVELY ON A BLACK CANVAS FOR DEBUGGING"""
    contours_image = np.zeros((1200,1000,3), np.uint8)
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

