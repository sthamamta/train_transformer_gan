'''
checkpoints to load the models
model_name to load correct model and use their function to load weights from checkpoints
factor: determines on which downsampled factor images model is tested
plot-dir: dir where plots is saved---leave as default
output-dir: dir where outputs are saved --- leave as deault
addition: boolean (not implemented) if True add the pred with images to get the final output of the model

THIS FILE IS CREATED TO EVALUATE GAUSSIAN IMAGE TRAINED MODEL WITH KSPACE PADDED, KSPACE PADDED AND GAUSSIAN DOWSAMPLE 50 MICRON IMAGES FROM ALL SUBJECTS

'''

from venv import create
import torch
import torch.nn as nn
import  cv2
import matplotlib.pyplot as plt
import argparse
import statistics
import json
import numpy as np
import test_utils as utils
from dataset.dataset_utils import prepare_lr_image
import random
import pickle

# from models.densenet_new import SRDenseNet

from utils.preprocess import tensor2image, image2tensor
from utils.image_quality_assessment import PSNR,SSIM
from utils.general import  NRMSELoss
import os
import numpy as np
from utils.preprocess import hfen_error
from models.esrt import  ESRT
from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore")

def min_max_normalize(image):
    max_img = image.max()
    min_img = image.min()
    denom = max_img-min_img
    norm_image = (image-min_img)/denom
    return norm_image 

def get_list(path, ext):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]

def forward_chop(model, x, shave=10, min_size=60000):
    scale = 2   #self.scale[self.idx_scale]
    n_GPUs = 1    #min(self.n_GPUs, 4)
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            sr_batch = model(lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(model, patch, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


def load_state_dict_func(path):

    # state_dict = torch.load(path)
    state_dict=path
    new_state_dcit = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dcit[name] = v
    return new_state_dcit


def load_model(opt):
    checkpoint = torch.load(opt.checkpoint,map_location=torch.device(opt.device))
    model = ESRT(
            upscale=checkpoint['upscale_factor'],
            n_feats=checkpoint['n_feats'],
            n_blocks=checkpoint['n_blocks'], 
            kernel_size=checkpoint['kernel_size']
            ).to(device=opt.device)
    model_dict = load_state_dict_func(checkpoint['model_state_dict'])
    model.load_state_dict(model_dict,strict=False)
    return model



 

def predict_model(model,lr_input,label_image,psnr,ssim,mse,nrmse):

    with torch.no_grad():
        pre = forward_chop(model, lr_input) #model(im_input)
        torch.cuda.synchronize()

    pre = pre.clamp(0.,1.)

    # print(pre.shape)
    # print(lr_input.shape)
    

    model_psnr = psnr(pre, label_image).item()
    model_ssim = ssim(pre, label_image).item()
    model_mse = mse(pre, label_image).item()
    model_nrmse = nrmse(pre, label_image).item()


    ref_arr = (label_image.squeeze().detach().cpu().numpy()) *255.
    pre_arr = (pre.squeeze().detach().cpu().numpy())*255.
    model_hfen = hfen_error(ref_arr,pre_arr).astype(np.float16).item()

    return  {
            'model_psnr':model_psnr,
            'model_ssim':model_ssim,
            'model_mse': model_mse,
            'model_nrmse':model_nrmse,
            'model_hfen': model_hfen}


# new function for evaluating upsample 50 micron as dictionary structure is different
def evaluate_model(opt):
    
    model_metric = {'psnr':[],'ssim':[],'mse':[],'nrmse':[],'hfen':[]}

    for index,image_path in enumerate(opt.image_list): 
        # image_path = os.path.join(opt.image_path,file)
        
        downsample_method = opt.degradation_method[index]
        hr_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        lr_image = prepare_lr_image(hr_image,downsample_method, 2)

        hr_image = min_max_normalize(hr_image)
        lr_image = min_max_normalize(lr_image)

        hr_image = torch.from_numpy(hr_image)
        lr_image = torch.from_numpy(lr_image)

        lr_image = torch.unsqueeze(lr_image.float(),0).to(opt.device)
        hr_image = torch.unsqueeze(hr_image.float(),0).to(opt.device)
        lr_image = torch.unsqueeze(lr_image,0)
        hr_image = torch.unsqueeze(hr_image,0)

        # print("shape of lr and hr image", lr_image.shape, hr_image.shape)
  

        output = predict_model(model=opt.model,lr_input=lr_image,label_image=hr_image,psnr=opt.psnr,ssim=opt.ssim,mse = opt.mse,nrmse=opt.nrmse)
            
        # append initial metric
        model_metric['psnr'].append(output['model_psnr'])
        model_metric['ssim'].append(output['model_ssim'])
        model_metric['mse'].append(output['model_mse'])
        model_metric['nrmse'].append(output['model_nrmse'])
        model_metric['hfen'].append(output['model_hfen'])

    #print min, max std and median

    print('metric for model')
    for key in model_metric.keys():
        print('key is',key) 
        print('min : ',min(model_metric[key])) 
        print('max : ',max(model_metric[key])) 
        if key == 'hfen':
            pass
        else:
            print( ' std :', statistics.pstdev(model_metric[key]))
        print( ' mean :', statistics.mean(model_metric[key]))
        print( ' median :', statistics.median(model_metric[key]))
        print('************************************************************************************')

    # with open(opt.model_name+'.yaml', 'w') as f:
    #     json.dump(model, f, indent=2)

    # with open(opt.initial_name+'.yaml', 'w') as f:
    #     json.dump(initial, f, indent=2)

    return model_metric


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ESRT')
    parser.add_argument("--test_hr_folder", type=str, default='../model_bias_experiment/mri_dataset_25/test',
                        help='the folder of the target images')
    parser.add_argument("--output_folder", type=str, default='results/Set5/x2_mri_result')
    opt = parser.parse_args()

    print(opt)

    
    # model_names = ['ph2_standard','ph2_lsgan','ph3_lsgan', 'ph1_l1_ssim','ph1_l1_ssim_v2', 'ph1_l1_ssim_ema','ph2_stand_l1ssimtv']
    # checkpoints =  [
    #     'outputs/esrt_gan/esrt_gan_factor_2_standardgan(l1_ssim)_(l1_ssim)_all_degradation/checkpoints/patch/patch-200/factor_2/ema_beta_0.9.pth',
    #     'outputs/esrt_gan/esrt_gan_factor_2_standardgan_(l1_ssim_tv)_all_degradation/checkpoints/patch/patch-200/factor_2/ema_beta_0.9.pth',
    #     'outputs/esrt_gan/esrt_gan_factor_2_lsgan_25_50_micron/checkpoints/patch/patch-200/factor_2/ema_beta_0.9.pth',
    #     'outputs/esrt_gan/esrt_factor_2_l1_ssim_all_degradation/checkpoints/patch/patch-200/factor_2/epoch_2500_f_2.pth' ,
    #     'outputs/esrt/esrt_factor_2_l1_ssim_all_degradation_version2/checkpoints/patch/patch-200/factor_2/epoch_2500_f_2.pth',
    #     'outputs/esrt/esrt_factor_2_l1_ssim_all_degradation_version2/checkpoints/patch/patch-200/factor_2/ema_beta_0.9.pth',
    #     'outputs/esrt_gan/esrt_gan_factor_2_standardgan_(l1_ssim_tv)_all_degradation/checkpoints/patch/patch-200/factor_2/ema_beta_0.9.pth'
    # ]


    model_names = ['ph1_l1_ssim','ph1_l1_ssim_v2', 'ph1_l1_ssim_ema']
    checkpoints = [
                'outputs/esrt/esrt_factor_2_l1_ssim_all_degradation_version2/checkpoints/patch/patch-200/factor_2/ema_beta_0.9.pth',
        'outputs/esrt_gan/esrt_factor_2_l1_ssim_all_degradation/checkpoints/patch/patch-200/factor_2/epoch_2500_f_2.pth' ,
        'outputs/esrt/esrt_factor_2_l1_ssim_all_degradation_version2/checkpoints/patch/patch-200/factor_2/epoch_2500_f_2.pth',
            ]


    # model_names = ['l1','l1_ssim','l1_ssim_tv', 'l1_ssim_tv_pyramid', 'mse', 'ssim']
    # checkpoints = [
    #     'outputs/esrt_gan/esrt_factor_2_l1_loss_all_degradation/checkpoints/patch/patch-200/factor_2/epoch_1000_f_2.pth',
    #     'outputs/esrt_gan/esrt_factor_2_l1_ssim_all_degradation/checkpoints/patch/patch-200/factor_2/epoch_2500_f_2.pth',
    #     'outputs/esrt_gan/esrt_factor_2_l1_ssim_tv_regularizer_all_degradation/checkpoints/patch/patch-200/factor_2/epoch_2500_f_2.pth',
    #     'outputs/esrt_gan/esrt_factor_2_l1_ssim_tv_regurlaizer_pyramid_loss_all_degradation/checkpoints/patch/patch-200/factor_2/epoch_2500_f_2.pth',
    #     'outputs/esrt_gan/esrt_factor_2_mse_loss_all_degradation/checkpoints/patch/patch-200/factor_2/epoch_1000_f_2.pth',
    #     'outputs/esrt_gan/esrt_factor_2_ssim_loss_all_degradation/checkpoints/patch/patch-200/factor_2/epoch_1000_f_2.pth',
    # ]
    
    
    dataset_names = ['combine']


    opt.image_path = opt.test_hr_folder
    

    '''set device'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device

    metrics = ['psnr', 'ssim', 'mse', 'nrmse','hfen']

    psnr = PSNR()
    ssim = SSIM()

    psnr = psnr.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)
    ssim = ssim.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)

    opt.psnr= psnr
    opt.ssim=ssim
    opt.mse = nn.MSELoss().to(device=opt.device)
    opt.nrmse = NRMSELoss().to(device=opt.device)


    model_pad_list = []
    model_sigma_list = []

    opt.image_list = get_list(opt.test_hr_folder, ext='.png')  #full image path

    # degradation_type_list = ['bicubic','nearest','bilinear','lanczos','kspace','kspace_gaussian_100','hanning', 'hamming']

    # test_size = len(opt.image_list)
    # repetition_factor = test_size / len(degradation_type_list)

    # opt.degradation_method = []

    # for degradation in degradation_type_list:
    #     repeated_elements = [degradation] * int(repetition_factor)
    #     opt.degradation_method.extend(repeated_elements)

    # random.shuffle(opt.degradation_method)
    # print(opt.degradation_method)

    # with open('test_degradation_list.pickle', 'wb') as file:
    #     pickle.dump(opt.degradation_method, file)

    with open('test_degradation_list.pickle', 'rb') as file:
        opt.degradation_method = pickle.load(file)


    
    for idx,checkpoint in enumerate(checkpoints):

        #loading the model
        opt.checkpoint = checkpoint
        model = load_model(opt)
        if device != 'cpu':
            num_of_gpus = torch.cuda.device_count()
            model = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])
        model.to(device)
        model.eval()
        opt.model= model

        model_metric = evaluate_model(opt)  # each dictionary of list with keys psnr,ssim,mse,nrmse,hfen

        print("checkpoint", checkpoint)
        print("Evaluating for model", model_names[idx])
        print("MODEL METRIC")

        for metric in metrics:
            print("Average Values for {} is {} ".format(metric,statistics.mean(model_metric[metric])))        
        print("**************************************************************************************************")





