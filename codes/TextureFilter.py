import cv2
import numpy as np
from cv2 import GaussianBlur


def create_mask(img, var, w_size=5):
    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    g_size = w_size // 2 + 1 if (w_size // 2) % 2 == 0 else w_size // 2
    img_g = GaussianBlur(img, (g_size, g_size), -1)
    img_padd = np.pad(img_g, w_size, 'reflect')
    tt_var = np.var(img_g)
    for y in range(h):
        for x in range(w):
            patch = img_padd[y + w_size:y + 2 * w_size, x + w_size:x + 2 * w_size]
            if np.var(patch) > var:
                mask[y, x] = 1

    return mask


def create_mask_canny(img, sigma=4, w_size=5):
    img_g = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=sigma)
    edge = cv2.Canny((255 * img_g).astype(np.uint8), 50, 80, apertureSize=3) / 255
    mask = np.zeros_like(img)

    kernel = np.ones((w_size, w_size))
    kernel = kernel / kernel.sum()
    edge_mean = cv2.filter2D(edge, -1, kernel)

    thrs = np.mean(edge) * 1.1
    mask[edge_mean > thrs] = 1

    return mask


def create_mask_laplacian(img, w_size=5):
    mask = np.zeros_like(img)
    h, w = img.shape
    mini_img = cv2.GaussianBlur(img, (5, 5), -1)[::2, ::2]
    lap = np.abs(img - cv2.resize(mini_img, (w, h)))

    kernel = np.ones((w_size, w_size))
    kernel = kernel / kernel.sum()
    edge_mean = cv2.filter2D(lap, -1, kernel)

    thrs = np.mean(lap[lap > 0])
    mask[edge_mean > thrs] = 1

    return mask
