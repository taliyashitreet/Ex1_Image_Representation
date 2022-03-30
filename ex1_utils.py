"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import cv2

import math

import numpy as np
from matplotlib import pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int_:
    return 314855099


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    image = cv2.imread(filename)
    image_norm = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    if representation == 1:
        gray_img = cv2.cvtColor(image_norm, cv2.COLOR_BGR2GRAY)
        return gray_img
    else:
        img_rgb = cv2.cvtColor(image_norm, cv2.COLOR_BGR2RGB)
        return img_rgb


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    image = imReadAndConvert(filename, representation)
    if representation == 1:
        plt.imshow(image, cmap='gray')
        plt.show()
    else:
        plt.imshow(image)
        plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    mult_mat = np.array([[0.299, 0.587, 0.114],
                         [0.59590059, -0.27455667, -0.32134392],
                         [0.21153661, -0.52273617, 0.31119955]])
    YIQ = np.dot(imgRGB, mult_mat.T.copy())
    return YIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    inverse_from_YIQ = np.linalg.inv(yiq_from_rgb)  # A*B=C ---> A=C*B^(-1)
    RGB_img = np.dot(imgYIQ, inverse_from_YIQ.T.copy())
    return RGB_img


def calHist(img: np.ndarray) -> np.ndarray:  # My function to crate a Histogram
    hist = np.zeros(256)
    for pix in range(256):
        hist[pix] = np.count_nonzero(img == pix)

    return hist


def calcBoundaries(k: int):
    z = np.zeros(k + 1, dtype=int)
    size = 256 / k
    for i in range(1, k):  # first boundary 0
        z[i] = z[i - 1] + size
    z[k] = 255  # last boundary 255
    return z


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    # from range [0,1] to [0,255]
    img = cv2.normalize(imgOrig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = (np.around(img)).astype('uint8')  # make sure its integers
    # original histogram
    histOrg = calHist(img)
    # cumsum
    cum_sum = histOrg.cumsum()

    lut = np.zeros(256)
    norm_cumSum = cum_sum / cum_sum.max()  # normalize each value of cumsum
    # create look up table
    for i in range(len(norm_cumSum)):
        new_color = int(np.floor(norm_cumSum[i] * 255))
        lut[i] = new_color

    imgEq = np.zeros_like(imgOrig, dtype=float)
    # Replace each intesity i with lut[i]
    for old_color, new_color in enumerate(lut):
        imgEq[img == old_color] = new_color

    # histogramEqualize
    histEQ = calHist(imgEq)
    # norm from range [0, 255] back to [0, 1]
    imgEq = imgEq / 255.0

    return imgEq, histOrg, histEQ


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    if (np.ndim(imOrig) == 2):  # its Gray scale image
        return quantize_chanel(imOrig.copy(), nQuant, nIter)

    # If an RGB image is given - convert it to YIQ
    rgb_to_yiq = transformRGB2YIQ(imOrig)
    # z : the borders which divide the histograms into segments
    qImage_i, error = quantize_chanel(rgb_to_yiq[:, :, 0].copy(), nQuant, nIter)  # take only Y chanel
    qImage = []
    for img in qImage_i:
        # convert the original img back from YIQ to RGB
        qImage_tmp = transformYIQ2RGB(np.dstack((img, rgb_to_yiq[:, :, 1], rgb_to_yiq[:, :, 2])))  # Rebuilds rgb arrays
        qImage.append(qImage_tmp)

    return qImage, error


def quantize_chanel(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    # return list:
    qImage = []
    Error = []

    # from range [0,1] to [0,255]
    img = imOrig * 255
    # create Histogram
    img_hist, edges = np.histogram(img.flatten(), bins=256)
    cumsum = img_hist.cumsum()
    # z = calcBoundaries(nQuant)
    z = calcBoundaries(nQuant)

    for iter in range(nIter):  # ask for make nIter times

        q = [] # contains the Weighted averages

        for i in range(nQuant):
            hist_range = img_hist[z[i]: z[i + 1]]
            i_range = range(len(hist_range))
            w_avg = np.average(i_range, weights=hist_range)
            q.append(w_avg + z[i])

        new_img = np.zeros_like(img)
        # Change all values in the range corresponding to the weighted average
        for border in range(len(q)):
            new_img[img > z[border]] = q[border]

        qImage.append(new_img / 255.0)  # back to range [0,1]

        # Mean Squared Error
        MSE = np.sqrt((img - new_img)**2).mean()
        Error.append(MSE)

        for i in range(1, len(q)):
            z[i] = (q[i - 1] + q[i]) / 2  # Change the boundaries according to q

    return qImage, Error

