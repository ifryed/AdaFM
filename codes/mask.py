import collections
import glob
import itertools
import os
import logging
import argparse

import cv2
import numpy as np
from collections import OrderedDict
import torch

import options.options as option
import utils.util as util
import TextureFilter as txtF
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
from utils.metrics import TrialStats, get_psnr_ssim
import matplotlib.pyplot as plt


def insertAlphaValue(model: list, mask: np.ndarray):
    net = model
    for n in net:
        if hasattr(n, 'mask'):
            n.mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to('cuda:0')
        elif hasattr(n, '_modules'):
            if len(list(n._modules.values())) > 0:
                insertAlphaValue(list(n._modules.values()), mask)


# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
parser.add_argument('-base_folder', type=str, required=False, default='../dataset/')
base_folder = parser.parse_args().base_folder
opt = option.parse(parser.parse_args().opt, is_train=False)
# util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
opt = option.dict_to_nonedict(opt)

# util.setup_logger(None, opt['path']['log'], 'test.log', level=logging.INFO, screen=True)
util.setup_logger(None, '../results', 'test.log', level=logging.INFO, screen=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))
# Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# Create model
model = create_model(opt)
stride = opt['interpolate_stride'] if opt['interpolate_stride'] is not None else 0.1
model_dict = torch.load(opt['path']['pretrain_model_G'])
c_dict = dict()
for i in model_dict.keys():
    if 'transformer' in i:
        c_dict[i.replace('transformer', 'transformer1')] = model_dict[i].clone()
        c_dict[i.replace('transformer', 'transformer2')] = model_dict[i].clone()
    else:
        c_dict[i] = model_dict[i]

model_dict = collections.OrderedDict(c_dict)

# img = cv2.cvtColor(cv2.imread('../soilder.png'), cv2.COLOR_BGR2RGB)
# gt_img = cv2.imread('../gt_soilder.png')
# mask = cv2.imread('../soilder_mask.png', cv2.IMREAD_GRAYSCALE) / 255.0
# mask = np.ones_like(mask,dtype=np.float32)

max_psnr = 0
max_vals = ''
INPUT_FLD = base_folder + '/CBSD68/'
GT_FLD = base_folder + '/CBSD68/original_png/'
exp_name = 'full_grid_PG_kmeans'
noisy_flds = glob.glob(INPUT_FLD + 'noisy50')

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nModulating [{:s}]...'.format(test_set_name))
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)
    need_HR = False if test_loader.dataset.opt['dataroot_HR'] is None else True

    # My code
    t_stats = TrialStats(logger, opt, exp_name)
    for noisy_input in noisy_flds:
        noise_base_fld = '../results/' + exp_name
        os.makedirs(noise_base_fld, exist_ok=True)
        for image_path in os.listdir(noisy_input):
            # img_o = cv2.imread('../soilder.png')
            img_o = cv2.imread(noisy_input + "/" + image_path)
            gt_img = cv2.imread(GT_FLD + "/" + image_path)
            img = cv2.cvtColor(img_o, cv2.COLOR_BGRA2RGB)
            img = np.expand_dims(img, 3).astype(np.float32) / 255
            img = np.transpose(img, (3, 2, 0, 1))
            data = {'LR': torch.from_numpy(img)}

            best_img = None
            best_psnr = -1

            # Create Mask
            # mask = create_mask(cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY) / 255)
            # mask = txtF.create_mask_laplacian(cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY) / 255)
            # mask = txtF.create_mask_canny(cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY) / 255, canny_sigma)
            # mask = txtF.create_mask_segnet(cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB))
            # mask = txtF.create_mask_patch_group(cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY), 10, 1)
            mask = txtF.create_mask_patch_group_DCT(cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY), 10, 1)

            for coef_fg, coef_bg in itertools.product(
                    np.arange(0, 1.01, stride),
                    np.arange(0, 1.01, stride)):
                # for coef_fg in np.arange(0, 1.01, stride):
                #     coef_bg = coef_fg
                # for coef_fg, coef_bg in itertools.product(np.arange(0.4, 0.61, stride), np.arange(0.4, 0.61, stride)):
                #     if coef_fg == coef_bg:
                #         continue
                print('setting coef to {:.2f}x{:.2f}'.format(coef_fg, coef_bg))

                interp_dict = model_dict.copy()
                net = list(model.netG.module.model._modules.values())

                # mask = np.ones_like(mask)
                insertAlphaValue(net, mask)
                for k, v in model_dict.items():
                    if k.find('transformer1') >= 0:
                        # interp_dict[k] = v * coef
                        interp_dict[k] = v * coef_fg
                    elif k.find('transformer2') >= 0:
                        # interp_dict[k] = v * (1 - coef)
                        interp_dict[k] = v * coef_bg
                # Adding the mask
                model.update(interp_dict)

                model.feed_data(data, need_HR=need_HR)
                img_path = '../results/mask'
                # img_name = os.path.splitext(os.path.basename(img_path))[0]
                # img_dir = os.path.join(dataset_dir, img_name)
                # util.mkdir(img_dir)
                img_name = image_path[:-4] + '_'

                model.test()

                visuals = model.get_current_visuals(need_HR=need_HR)

                sr_img = util.tensor2img(visuals['SR'])  # uint8

                # Picking the best
                tmp_psnr = get_psnr_ssim(sr_img, gt_img, opt['crop_size'])[0]
                if tmp_psnr > best_psnr:
                    best_psnr = tmp_psnr
                    best_img = sr_img
                    best_c_fg, best_c_bg = coef_fg, coef_bg
                    print('\t\tBest One:{}'.format(best_psnr))

            # save images
            img_part_name = '_coef_{:.2f}_{:.2f}.png'.format(best_c_fg, best_c_bg)
            suffix = opt['suffix']
            if suffix:
                save_img_path = os.path.join(noise_base_fld, img_name + suffix + img_part_name)
            else:
                save_img_path = os.path.join(noise_base_fld, img_name + img_part_name)
            util.save_img(best_img, save_img_path)
            util.save_img(mask * 255, save_img_path[:-4] + '_mask.png')

            t_stats.cal_psnr_ssim(img_name, best_img, gt_img, best_c_fg, best_c_bg)

    t_stats.final_report(False)
