
import cv2 as cv
import pytesseract as tess
import os

# runs tesseract to perform
def run_tesseract(image, psm, oem):
    language = 'eng'
    configuration = "--psm " + str(psm) + " --oem " + str(oem)

    # Run tesseract
    text = tess.image_to_string(image, lang=language, config=configuration)
    if len(text.strip()) == 0:
        configuration += " -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        text = tess.image_to_string(image, lang=language, config=configuration)

    return text



# show image FOR DEBUGGING
def showImage(image, title):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# show contours FOR DEBUGGING
def showContours(contours):
    contours_image = np.zeros((720,1280,3), np.uint8)
    cv.drawContours(contours_image, contours, -1, (0,255,0), 2)
    showImage(contours_image, "contours")

# show contour on image FOR DEBUGGING
def showContoursOnImage(contours, image):
    for i in range(len(contours) - 1, -1, -1):
        img = image
        cv.drawContours(img, contours, i, (0, 0, 255), 1)
        showImage(img, "image with contour")

# show contours iteratively on a black image FOR DEBUGGING
def showContoursIter(contours):
    contours_image = np.zeros((720,1280,3), np.uint8)
    for i in range(len(contours) - 1, -1, -1):
        cv.drawContours(contours_image, contours, i, (0,255,0), 2)
        showImage(contours_image, "Contours")