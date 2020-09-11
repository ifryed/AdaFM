import os

import numpy as np
import cv2
import sys


def main():
    sigmas = [int(x) for x in sys.argv[1].split(',')]
    input_fld = sys.argv[2]

    for sigma in sigmas:
        output_fld = os.path.join(sys.argv[3], 'noise_{}'.format(sigma))
        os.makedirs(output_fld, exist_ok=True)
        for img_p in [x for x in os.listdir(input_fld) if x.endswith('png')]:
            print('\rSigma:{}, Img:{}'.format(sigma, img_p), end='')

            img = cv2.imread(os.path.join(input_fld, img_p)) / 255
            h, w, c = img.shape
            noise_kernel = np.random.normal(0, sigma / 255, (h, w, c))

            n_img = img + noise_kernel
            n_img[n_img > 1] = 1
            n_img[n_img < 0] = 0
            n_img = (n_img * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_fld, img_p), n_img)
        print()


if __name__ == '__main__':
    main()
