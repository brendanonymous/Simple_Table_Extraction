
import cv2 as cv
import pytesseract as tess
from PIL import Image
import os

"""
Run tesseract to perform optical character recognition (OCR)
"""
def run_tesseract(image, psm, oem):
    language = 'eng'
    configuration = "--psm " + str(psm) + " --oem " + str(oem)

    # Run tesseract
    text = tess.image_to_string(image, lang=language, config=configuration)
    if len(text.strip()) == 0:
        configuration += " -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        text = tess.image_to_string(image, lang=language, config=configuration)

    return text