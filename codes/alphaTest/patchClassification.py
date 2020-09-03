import cv2
import numpy as np


def createCanny(img, sigma=4, w_size=5):
    img_g = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=sigma)
    edge = cv2.Canny((255 * img_g).astype(np.uint8), 50, 80, apertureSize=3) / 255

    kernel = np.ones((w_size, w_size))
    kernel = kernel / kernel.sum()
    edge_mean = cv2.filter2D(edge, -1, kernel)

    return edge_mean

def preformDCT(img):
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255

    h, w = img_g.shape[:2]
    h = h - 1 if h % 2 > 0 else h
    w = w - 1 if w % 2 > 0 else w
    img_g = img_g[:h, :w]
    return cv2.dct(img_g)
