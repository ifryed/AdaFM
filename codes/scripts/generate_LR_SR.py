import os

import numpy as np
import cv2
import sys

from data.util import modcrop

kSR_LEVELS = 1
kINPUT_FLD = 2
kOUTPUT_FLD = 3


def main():
    sr_levels = [int(x) for x in sys.argv[kSR_LEVELS].split(',')]
    input_fld = sys.argv[kINPUT_FLD]

    for sr_zoom in sr_levels:
        output_fld = os.path.join(sys.argv[kOUTPUT_FLD], 'sr_{}'.format(sr_zoom))
        os.makedirs(output_fld, exist_ok=True)

        imgs_path_list = [x for x in os.listdir(input_fld) if x.endswith('png')]
        imgs_path_list.sort()

        for img_p in imgs_path_list:
            print('\rSR Level:{}, Img:{}'.format(sr_zoom, img_p), end='')

            img = cv2.imread(os.path.join(input_fld, img_p))
            img = modcrop(img, sr_zoom)
            img = cv2.resize(img, (0, 0), fx=1 / sr_zoom, fy=1 / sr_zoom, interpolation=cv2.INTER_CUBIC)
            img = cv2.resize(img, (0, 0), fx=sr_zoom, fy=sr_zoom, interpolation=cv2.INTER_CUBIC)
            output_path = os.path.join(output_fld, img_p)
            cv2.imwrite(output_path, img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), sr_zoom])
        print()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("[SR levels(csv)] [input folder] [output folder]")
        exit()
    main()
