import itertools
import os
import logging
import argparse
import numpy as np
from collections import OrderedDict
import torch

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
opt = option.dict_to_nonedict(opt)

util.setup_logger(None, opt['path']['log'], 'test.log', level=logging.INFO, screen=True)
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
model_dict_psnr = torch.load(opt['path']['pretrain_model_PSNR'])
model_dict_ssim = torch.load(opt['path']['pretrain_model_SSIM'])

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nModulating [{:s}]...'.format(test_set_name))
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)
    need_HR = False if test_loader.dataset.opt['dataroot_HR'] is None else True

    for ns_lvl, p2s_lvl in itertools.product(
            np.arange(0, 1.01, stride),
            np.arange(0, 1.01, stride)):
        print('setting coef to {:.2f}'.format(ns_lvl))

        interp_dict_psnr = model_dict_psnr.copy()
        for k, v in model_dict_psnr.items():
            if k.find('transformer') >= 0:
                interp_dict_psnr[k] = v * ns_lvl

        interp_dict_ssim = model_dict_ssim.copy()
        for k, v in model_dict_psnr.items():
            if k.find('transformer') >= 0:
                interp_dict_ssim[k] = v * ns_lvl

        interp_dict_blend = model_dict_psnr.copy()
        for k, v in model_dict_psnr.items():
            interp_dict_blend[k] = interp_dict_psnr[k] * p2s_lvl + (1 - p2s_lvl) * interp_dict_ssim[k]
        model.update(interp_dict_blend)

        # load the test data
        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['psnr_y'] = []
        test_results['ssim_y'] = []

        for data in test_loader:
            model.feed_data(data, need_HR=need_HR)
            img_path = data['LR_path'][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img_dir = os.path.join(dataset_dir, img_name)
            util.mkdir(img_dir)

            model.test()

            visuals = model.get_current_visuals(need_HR=need_HR)

            sr_img = util.tensor2img(visuals['SR'])  # uint8

            # save images
            img_part_name = '_nslvl_{:.2f}_blend{:.2f}.png'.format(ns_lvl, p2s_lvl)
            suffix = opt['suffix']
            if suffix:
                save_img_path = os.path.join(img_dir, img_name + suffix + img_part_name)
            else:
                save_img_path = os.path.join(img_dir, img_name + img_part_name)
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

                if gt_img.shape[2] == 3:  # RGB image
                    sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                    gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                    if crop_size > 0:
                        sr_img_y = sr_img_y[crop_size:-crop_size, crop_size:-crop_size]
                        gt_img_y = gt_img_y[crop_size:-crop_size, crop_size:-crop_size]
                    psnr_y = util.calculate_psnr(sr_img_y * 255, gt_img_y * 255)
                    ssim_y = util.calculate_ssim(sr_img_y * 255, gt_img_y * 255)
                    test_results['psnr_y'].append(psnr_y)
                    test_results['ssim_y'].append(ssim_y)
                    logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.' \
                                .format(img_name, psnr, ssim, psnr_y, ssim_y))
                else:
                    logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, psnr, ssim))
            else:
                logger.info(img_name)

        if need_HR:  # metrics
            # Average PSNR/SSIM results
            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            logger.info('----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n' \
                        .format(test_set_name, ave_psnr, ave_ssim))
            if test_results['psnr_y'] and test_results['ssim_y']:
                ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
                logger.info('----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n' \
                            .format(ave_psnr_y, ave_ssim_y))