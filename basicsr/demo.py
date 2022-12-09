# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch

# from basicsr.data import create_dataloader, create_dataset
from models import create_model
from train import parse_options
from utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite

# from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
#                            make_exp_dirs)
# from basicsr.utils.options import dict2str
from math import log10, sqrt
import cv2
import numpy as np
import os
from glob import glob
import time


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def myPSNR(tar_img, prd_img):
    imdff = np.clip(prd_img,0,1) - np.clip(tar_img,0,1)
    mse = np.mean((imdff**2))
    ps = 20*np.log10(1/np.sqrt(mse)) #MAXf is 1 since our range is from 0 to 1
    return ps

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()

    img_paths = opt['img_path'].get('input_img')
    img_paths = sorted(glob(os.path.join(img_paths, '*.png')))
    output_paths = opt['img_path'].get('output_img')
    print(img_paths)
    print(output_paths)
    opt['dist'] = False
    model = create_model(opt)

    for path in img_paths:
        ## 1. read image
        filename = path.split('/')[-1]
        file_client = FileClient('disk')

        img_bytes = file_client.get(path, None)
        try:
            img = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("path {} not working".format(path))

        img = img2tensor(img, bgr2rgb=True, float32=True)



        ## 2. run inference
        

        model.feed_data(data={'lq': img.unsqueeze(dim=0)})

        if model.opt['val'].get('grids', False):
            model.grids()

        model.test()

        if model.opt['val'].get('grids', False):
            model.grids_inverse()

        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        output_path = os.path.join(output_paths, filename)
        imwrite(sr_img, output_path)

        print(f'inference {path} .. finished. saved to {output_path}')
     
if __name__ == '__main__':
    main()

