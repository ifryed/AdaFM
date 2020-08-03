# calculate PSNR and SSIM
from collections import OrderedDict
from data.util import bgr2ycbcr
import utils.util as util


class TrialStats():
    def __init__(self, logger, opt, test_name):
        self.test_results = OrderedDict()
        self.test_results['psnr'] = []
        self.test_results['ssim'] = []
        self.test_results['psnr_y'] = []
        self.test_results['ssim_y'] = []

        self.logger = logger
        self.max_psnr = 0
        self.opt = opt
        self.test_set_name = test_name

    def cal_psnr_ssim(self, img_name, sr_img, gt_img, coef_fg, coef_bg):
        if gt_img is not None:
            gt_img = gt_img / 255.
            sr_img = sr_img / 255.

            crop_size = self.opt['crop_size']
            if crop_size > 0:
                sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
            gt_h, gt_w = gt_img.shape[:2]
            sr_img = sr_img[:gt_h, :gt_w]

            psnr = util.calculate_psnr(sr_img * 255, gt_img * 255)
            ssim = util.calculate_ssim(sr_img * 255, gt_img * 255)
            if self.max_psnr < psnr:
                max_psnr = psnr
                max_vals = '{:.2f}_{:.2f}'.format(coef_fg, coef_bg)
            self.test_results['psnr'].append(psnr)
            self.test_results['ssim'].append(ssim)

            if gt_img.shape[2] == 3:  # RGB image
                sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                if crop_size > 0:
                    sr_img_y = sr_img_y[crop_size:-crop_size, crop_size:-crop_size]
                    gt_img_y = gt_img_y[crop_size:-crop_size, crop_size:-crop_size]
                psnr_y = util.calculate_psnr(sr_img_y * 255, gt_img_y * 255)
                ssim_y = util.calculate_ssim(sr_img_y * 255, gt_img_y * 255)
                self.test_results['psnr_y'].append(psnr_y)
                self.test_results['ssim_y'].append(ssim_y)
                self.logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.' \
                                 .format(img_name, psnr, ssim, psnr_y, ssim_y))
            else:
                self.logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, psnr, ssim))
        else:
            self.logger.info(img_name)

    def final_report(self, save=False):
        # metrics
        # Average PSNR/SSIM results
        print()
        ave_psnr = sum(self.test_results['psnr']) / len(self.test_results['psnr'])
        ave_ssim = sum(self.test_results['ssim']) / len(self.test_results['ssim'])
        self.logger.info('----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n' \
                         .format(self.test_set_name, ave_psnr, ave_ssim))
        if self.test_results['psnr_y'] and self.test_results['ssim_y']:
            ave_psnr_y = sum(self.test_results['psnr_y']) / len(self.test_results['psnr_y'])
            ave_ssim_y = sum(self.test_results['ssim_y']) / len(self.test_results['ssim_y'])
            self.logger.info('----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n' \
                             .format(ave_psnr_y, ave_ssim_y))
        print()
        print('Max:', self.max_psnr)
        print('Vals:', self.max_vals)
        if save:
            with open('res.txt', 'w') as o_file:
                o_file.write("Image:{}, Max PSNR: {:.5f}".format(self.max_vals, self.max_psnr))
