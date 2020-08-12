import itertools

import cv2
import numpy as np
from cv2 import GaussianBlur
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


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


def create_mask_segnet(img):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(Image.fromarray(img.astype(np.uint8)))
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    mask = (output_predictions.cpu().numpy() > 0).astype(np.float64)
    return mask


def create_mask_patch_group(img: np.ndarray, w_size: int = 10, stride: int = 1) -> np.ndarray:
    h, w = img.shape[:2]
    h = (np.floor(h / w_size) * w_size).astype(int)
    w = (np.floor(w / w_size) * w_size).astype(int)
    img_c = cv2.resize(img, (w, h))

    # img_c = cv2.dct(img_c)
    patches = [img_c[ys:ys + w_size, xs:xs + w_size] for ys, xs in itertools.product(
        np.arange(0, h - w_size, stride),
        np.arange(0, w - w_size, stride))]
    nh, nw = len(np.arange(0, h - w_size, stride)), len(np.arange(0, w - w_size, stride))
    patches = np.array(patches)
    patches_v = patches.reshape(-1, w_size ** 2)

    kmeans = KMeans(n_clusters=2).fit(patches_v)
    l_mat = kmeans.labels_.reshape((nh, nw))

    return (cv2.resize(l_mat.astype(np.float64), (img.shape[1::-1])) > 0).astype(int)
