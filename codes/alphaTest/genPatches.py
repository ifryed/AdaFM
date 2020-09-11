import os
from typing import Callable

import cv2
import numpy as np
import TextureFilter as texF
from alphaTest.patchClassification import createCanny, preformDCT
from sklearn.cluster import KMeans
import tqdm

DEBUG = False


def selectRandomPatches(img: np.ndarray, img_gt: np.ndarray, patch_size: int, n_patches: int = 50,
                        img_add: np.ndarray = None, img_add_fun: Callable = lambda x: x):
    h, w = img.shape[:2]

    patchs = []
    patchs_gt = []
    patchs_add = []
    for i in range(n_patches):
        x = np.random.randint(0, w - patch_size)
        y = np.random.randint(0, h - patch_size)

        patchs.append(img[y:y + patch_size, x:x + patch_size])
        patchs_gt.append(img_gt[y:y + patch_size, x:x + patch_size])
        if img_add is not None:
            patchs_add.append(img_add_fun(img_add[y:y + patch_size, x:x + patch_size]))

    return patchs, patchs_gt, patchs_add


def main():
    prefix = 'Canny_gt'
    base_folder = '../../datasets/CBSD68'
    out_folder = '../../datasets/patches/' + prefix
    patch_size = 11
    texture_type_num = 5

    noises_folders = [x for x in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, x))]
    noises_folders = [x for x in noises_folders if x.startswith('noisy')]

    for n_fld in noises_folders:
        print('Noise: {}'.format(n_fld))

        os.makedirs(os.path.join(out_folder, 'noisy', n_fld), exist_ok=True)
        os.makedirs(os.path.join(out_folder, 'gt', n_fld), exist_ok=True)

        fld_path = os.path.join(base_folder, n_fld)
        images_paths = [x for x in os.listdir(fld_path) if x.endswith('png')]

        rand_patches = []
        rand_patches_add_info = []
        rand_patches_gt = []

        # Sample random patches
        if DEBUG:
            images_paths = images_paths[:5]
        for img_path in tqdm.tqdm(images_paths):
            img_full_path = os.path.join(fld_path, img_path)
            img_gt_full_path = os.path.join(base_folder, 'original_png', img_path)

            img = cv2.imread(img_full_path)
            img_gt = cv2.imread(img_gt_full_path)
            img_canny = createCanny(img_gt)

            r_patches, r_patches_gt, r_patches_add_info = selectRandomPatches(img, img_gt, patch_size,
                                                                              img_add=img_canny,
                                                                              img_add_fun=lambda x: x.mean())
                                                                              # img_add_fun=lambda x: preformDCT(x))
            rand_patches += r_patches
            rand_patches_add_info += r_patches_add_info
            rand_patches_gt += r_patches_gt

        # Classifing them in to n levels
        rand_patches_add_info = np.array(rand_patches_add_info).reshape(len(rand_patches),-1)
        kmeans = KMeans(n_clusters=texture_type_num).fit(rand_patches_add_info)
        new_labels = kmeans.labels_.copy()
        cent_ord = np.argsort(kmeans.cluster_centers_.flatten())
        for i, cc in enumerate(cent_ord):
            new_labels[kmeans.labels_ == cc] = i

        for tex in range(texture_type_num):
            os.makedirs(os.path.join(out_folder, 'noisy', n_fld, 'text_{}'.format(tex)), exist_ok=True)
            os.makedirs(os.path.join(out_folder, 'gt', n_fld, 'text_{}'.format(tex)), exist_ok=True)

        for idx, lbl in enumerate(new_labels):
            save_path = os.path.join(out_folder, 'noisy', n_fld, 'text_{}'.format(lbl))
            save_path_gt = os.path.join(out_folder, 'gt', n_fld, 'text_{}'.format(lbl))

            cv2.imwrite(save_path + '/{}.png'.format(idx), rand_patches[idx])
            cv2.imwrite(save_path_gt + '/{}.png'.format(idx), rand_patches_gt[idx])


if __name__ == '__main__':
    main()
