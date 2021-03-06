import os
import logging
import time
import argparse
from collections import OrderedDict
import copy
from multiprocessing.spawn import freeze_support

import torch

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
import matplotlib.pyplot as plt
import numpy as np

# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
parser.add_argument('-sigmas', type=int, nargs='+', default=[15, 75], help='Test sigmas')
parser.add_argument('-b', type=int, nargs='+', default=[15, 75], help='Trained sigmas')
parser.add_argument('-best', default=False,action="store_true")

sigmas = parser.parse_args().sigmas
l_bound, u_bound = parser.parse_args().b

opt = option.parse(parser.parse_args().opt, is_train=False)
util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
opt = option.dict_to_nonedict(opt)

util.setup_logger(None, opt['path']['log'], 'test.log', level=logging.INFO, screen=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))
# Create test dataset and dataloader
test_loaders = []
# for phase, dataset_opt in sorted(opt['datasets'].items()):
#     test_set = create_dataset(dataset_opt)
#     test_loader = create_dataloader(test_set, dataset_opt)
#     logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
#     test_loaders.append(test_loader)
dataset_opt = opt['datasets']['test']
gray = opt['gray']
for sigma in sigmas:
    dataset_opt['name'] = 'test_{}'.format(sigma)
    dataset_opt['coef'] = (sigma - l_bound) / (u_bound - l_bound)
    data_lr = dataset_opt['dataroot_LR']
    old_fld = os.path.basename(data_lr)
    new_fld = ''.join([x for x in old_fld if not x.isnumeric()]) + str(sigma)
    dataset_opt['dataroot_LR'] = data_lr[:data_lr.rfind(os.path.basename(data_lr))] + new_fld
    test_set = copy.deepcopy(create_dataset(dataset_opt,gray))
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# Create model
model = create_model(opt)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    # logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    model_dict = torch.load(opt['path']['pretrain_model_G'])
    base_alpha = test_loader.dataset.opt['coef']
    alpha_lst = [0]
    if parser.parse_args().best:
        alpha_lst = np.unique([x + base_alpha for x in [-.05, -.025, 0, .025, .05]])
    best_psnr = -1
    best_ssim = -1
    for alpha in alpha_lst:
        interp_dict = model_dict.copy()
        for k, v in model_dict.items():
            if k.find('transformer') >= 0:
                interp_dict[k] = v * alpha
        model.update(interp_dict)

        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['psnr_y'] = []
        test_results['ssim_y'] = []

        for data in test_loader:
            need_HR = False if test_loader.dataset.opt['dataroot_HR'] is None else True

            model.feed_data(data, need_HR=need_HR)
            img_path = data['LR_path'][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            model.test()  # test
            visuals = model.get_current_visuals(need_HR=need_HR)

            sr_img = util.tensor2img(visuals['SR'])  # uint8

            # save images
            suffix = opt['suffix']
            if suffix:
                save_img_path = os.path.join(dataset_dir, img_name + suffix + '.png')
            else:
                save_img_path = os.path.join(dataset_dir, img_name + '.png')
            util.save_img(sr_img, save_img_path)

            # calculate PSNR and SSIM
            if need_HR:
                gt_img = util.tensor2img(visuals['HR'])
                gt_img = gt_img / 255.
                sr_img = sr_img / 255.

                crop_size = opt['crop_size']
                if crop_size > 0:
                    sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]

                psnr = util.calculate_psnr(sr_img * 255, gt_img * 255)
                ssim = util.calculate_ssim(sr_img * 255, gt_img * 255)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)

                # if gt_img.shape[2] == 3:  # RGB image
                #     sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                #     gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                #     if crop_size > 0:
                #         sr_img_y = sr_img_y[crop_size:-crop_size, crop_size:-crop_size]
                #         gt_img_y = gt_img_y[crop_size:-crop_size, crop_size:-crop_size]
                #     psnr_y = util.calculate_psnr(sr_img_y * 255, gt_img_y * 255)
                #     ssim_y = util.calculate_ssim(sr_img_y * 255, gt_img_y * 255)
                #     test_results['psnr_y'].append(psnr_y)
                #     test_results['ssim_y'].append(ssim_y)
                #     # logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.' \
                #     #             .format(img_name, psnr, ssim, psnr_y, ssim_y))
                # else:
                #     logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, psnr, ssim))
            else:
                logger.info(img_name)

        if need_HR:  # metrics
            best_psnr = np.max([best_psnr, sum(test_results['psnr']) / len(test_results['psnr'])])
            best_ssim = np.max([best_ssim, sum(test_results['ssim']) / len(test_results['ssim'])])

    if need_HR:
        # Average PSNR/SSIM results
        # logger.info('----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n' \
        #            .format(test_set_name, best_psnr, best_ssim))
        print(test_set_name, ":", best_psnr)

        # if test_results['psnr_y'] and test_results['ssim_y']:
        # ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
        # ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
        # logger.info('----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n' \
        #             .format(ave_psnr_y, ave_ssim_y))
