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
import cv2
import numpy as np

from ex1_utils import LOAD_GRAY_SCALE


def gammaCorrection(img: np.ndarray, gamma: float) -> np.ndarray:
    g_array = np.full(img.shape, gamma)
    new_img = np.power(img, g_array)
    return new_img


def on_trackbar(x: int):  # do nothing
    pass


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    if rep == 1:  # Gray_Scale representation
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:  # =2, read as BGR image
        img = cv2.imread(img_path)
    cv2.namedWindow("Gamma display")
    cv2.imshow("Gamma display", img)
    cv2.createTrackbar("Gamma*200", "Gamma display", 100, 200, on_trackbar)
    while True:
        pos = cv2.getTrackbarPos("Gamma*200", "Gamma display")
        pos = pos / 200  # each number in the slide represents gamma * 200
        new_img = gammaCorrection(img / 255.0, pos)
        cv2.imshow("Gamma display", new_img)
        key = cv2.waitKey(1000)  # Wait until user press some key
        if key == 27:  # Esc
            break
        if cv2.getWindowProperty("Gamma display", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyWindow('image')


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
