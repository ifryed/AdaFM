import collections
import glob
import itertools
import os
import logging
import argparse

import cv2
import numpy as np
from collections import OrderedDict
import tqdm
import torch

import sys
sys.path.append('../')

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

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

# Create model
model = create_model(opt)
stride = opt['interpolate_stride'] if opt['interpolate_stride'] is not None else 0.1
model_dict = torch.load(opt['path']['pretrain_model_G'])

# c_dict = dict()
# for i in model_dict.keys():
#     if 'transformer' in i:
#         c_dict[i.replace('transformer', 'transformer1')] = model_dict[i].clone()
#         c_dict[i.replace('transformer', 'transformer2')] = model_dict[i].clone()
#     else:
#         c_dict[i] = model_dict[i]
#
# model_dict = collections.OrderedDict(c_dict)

# img = cv2.cvtColor(cv2.imread('../soilder.png'), cv2.COLOR_BGR2RGB)
# gt_img = cv2.imread('../gt_soilder.png')
# mask = cv2.imread('../soilder_mask.png', cv2.IMREAD_GRAYSCALE) / 255.0
# mask = np.ones_like(mask,dtype=np.float32)

max_psnr = 0
max_vals = ''
prefix = 'Canny_gt/'
INPUT_FLD = base_folder + prefix + '/noisy/'
GT_FLD = base_folder + prefix + '/gt/'
exp_name = 'patch_test_canny_gt'

test_set_name = 'test_' + exp_name
logger.info('\nModulating [{:s}]...'.format(test_set_name))
dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
util.mkdir(dataset_dir)
need_HR = False

out_fld = '../../results/' + exp_name
os.makedirs(out_fld, exist_ok=True)
summary_file = open(os.path.join(out_fld, 'summary.txt'), 'w')
summary_file.write('noise\ttexture\talpha\tpsnr\n')

# My code
noise_folders = os.listdir(INPUT_FLD)
t_stats = TrialStats(logger, opt, exp_name)
for noise_sig in noise_folders:
    print('Noise {}'.format(noise_sig), end='')
    noise_fld = os.path.join(INPUT_FLD, noise_sig)
    noise_fld_gt = os.path.join(GT_FLD, noise_sig)

    tex_flds = os.listdir(noise_fld)
    for tex_input in tex_flds:
        print('\n\tTexture {}'.format(tex_input), end='')

        for alpha in np.arange(0, 1.01, stride):
            print('\n\t\tSetting Alpha to {:.2f}'.format(alpha))
            avg_psnr = []

            for image_path in tqdm.tqdm(os.listdir(os.path.join(noise_fld, tex_input))):
                # print('\r\t\t\tPatch {}'.format(os.path.basename(image_path)), end='')

                fld_name = os.path.basename(tex_input)
                img_o = cv2.imread(noise_fld + '/' + fld_name + "/" + image_path)
                gt_img = cv2.imread(noise_fld_gt + '/' + fld_name + "/" + image_path)
                img = cv2.cvtColor(img_o, cv2.COLOR_BGRA2RGB)
                img = np.expand_dims(img, 3).astype(np.float32) / 255
                img = np.transpose(img, (3, 2, 0, 1))
                data = {'LR': torch.from_numpy(img)}

                best_img = None
                best_psnr = -1

                interp_dict = model_dict.copy()
                net = list(model.netG.module.model._modules.values())

                # mask = np.ones_like(mask)
                for k, v in model_dict.items():
                    if k.find('transformer') >= 0:
                        interp_dict[k] = v * alpha
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
                gt_h, gt_w = gt_img.shape[:2]
                tmp_psnr = util.calculate_psnr(sr_img[:gt_h, :gt_w], gt_img)
                avg_psnr.append(tmp_psnr)

            avg_psnr = np.array(avg_psnr).mean()
            # Summary
            summary_file.write('{}\t{}\t{:.1f}\t{:.5f}\n'.format(noise_sig, tex_input, alpha, avg_psnr))
            summary_file.flush()

# t_stats.final_report(False)
