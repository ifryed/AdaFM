import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import data.util as util


class KERMODDataset(data.Dataset):
    '''
    Read LR and HR image pairs.
    If only HR image is provided, generate LR image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(KERMODDataset, self).__init__()
        self.opt = opt
        self.paths_ORG = None
        self.paths_TRGT = None
        self.ORG_env = None  # environment for lmdb
        self.TRGT_env = None

        self.TRGT_env, self.paths_TRGT = util.get_kernel_paths(opt['data_type'], opt['dataroot_TRGT'])
        self.ORG_env, self.paths_ORG = util.get_kernel_paths(opt['data_type'], opt['dataroot_ORG'])

        assert self.paths_TRGT, 'Error: HR path is empty.'
        if self.paths_ORG and self.paths_TRGT:
            assert len(self.paths_ORG) == len(self.paths_TRGT), \
                'Target and Origin datasets have different number of images - {}, {}.'.format( \
                    len(self.paths_ORG), len(self.paths_TRGT))

    def __getitem__(self, index):
        TRGT_path, ORG_path = None, None

        # get HR image
        TRGT_path = self.paths_TRGT[index]
        ker_TRGT = util.read_model2img(TRGT_path)

        # get LR image
        ORG_path = self.paths_ORG[index]
        ker_ORG = util.read_model2img(ORG_path)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(ker_TRGT, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(ker_ORG, (2, 0, 1)))).float()

        if ORG_path is None:
            ORG_path = TRGT_path
        return {'LR': img_LR, 'HR': img_HR, 'LR_path': ORG_path, 'HR_path': TRGT_path}

    def __len__(self):
        return len(self.paths_TRGT)
