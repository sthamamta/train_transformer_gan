import argparse
import torch
import os
import numpy as np
import utils
import skimage.color as sc
import cv2
from models.esrt import  ESRT
import matplotlib.pyplot as plt
from collections import OrderedDict
import test_utils as utils
import glob
from PIL import Image

# Testing settings

# Test_Datasets/Set5/
# Test_Datasets/Set5_LR/x2/

checkpoints =  [
        'outputs/esrt_gan/esrt_factor_2_l1_ssim_all_degradation/checkpoints/patch/patch-200/factor_2/epoch_2500_f_2.pth',
        'outputs/esrt_gan/esrt_gan_factor_2_standardgan(l1_ssim)_(l1_ssim)_all_degradation/checkpoints/patch/patch-200/factor_2/epoch_2360_f_2.pth',
        'outputs/esrt_gan/esrt_gan_factor_2_standard_gan_all_degradation_25_50_micron/checkpoints/patch/patch-200/factor_2/epoch_1400_f_2.pth',
        'outputs/esrt_gan/esrt_gan_factor_2_standardgan_(l1_ssim_tv)_all_degradation/checkpoints/patch/patch-200/factor_2/epoch_2480_f_2.pth',
        # 'outputs/esrt_gan/esrt_gan_factor_2_lsgan_25_50_micron/checkpoints/patch/patch-200/factor_2/epoch_1800_f_2.pth',
        'outputs/esrt_gan/esrt_gan_factor_2_lsgan_25_50_micron/checkpoints/patch/patch-200/factor_2/epoch_1520_f_2.pth',
        'outputs/esrt_gan/esrt_gan_factor_2_lsgan(l1_ssim)_(l1_ssim)_all_degradation/checkpoints/patch/patch-200/factor_2/epoch_2200_f_2.pth',
        'outputs/esrt/esrt_l1_ph1/checkpoints/patch/patch-200/factor_2/epoch_3500_f_2.pth',
        'outputs/esrt_gan/esrt_standardgan(l1)_(l1)_ph2_debug/checkpoints/patch/patch-200/factor_2/epoch_2400_f_2.pth',
        'outputs/esrt_gan/esrt_standardgan(l1)_(l1)_ph2/checkpoints/patch/patch-200/factor_2/epoch_150_f_2.pth'
    ]
    
names =  ['ph1_l1_ssim','ph2_standard','ph3_standard','ph2_lstandard', 'ph3_lsgan', 'ph2_lsgan', 'ph1_l1', 'ph2_standard_debug', 'ph2_standard_default']

# names = ['l1','l1_ssim','l1_ssim_tv', 'l1_ssim_tv_pyramid', 'mse', 'ssim']
# checkpoints = [
#     'outputs/esrt_gan/esrt_factor_2_l1_loss_all_degradation/checkpoints/patch/patch-200/factor_2/epoch_1000_f_2.pth',
#     'outputs/esrt_gan/esrt_factor_2_l1_ssim_all_degradation/checkpoints/patch/patch-200/factor_2/epoch_2500_f_2.pth',
#     'outputs/esrt_gan/esrt_factor_2_l1_ssim_tv_regularizer_all_degradation/checkpoints/patch/patch-200/factor_2/epoch_2500_f_2.pth',
#     'outputs/esrt_gan/esrt_factor_2_l1_ssim_tv_regurlaizer_pyramid_loss_all_degradation/checkpoints/patch/patch-200/factor_2/epoch_2500_f_2.pth',
#     'outputs/esrt_gan/esrt_factor_2_mse_loss_all_degradation/checkpoints/patch/patch-200/factor_2/epoch_1000_f_2.pth',
#     'outputs/esrt_gan/esrt_factor_2_ssim_loss_all_degradation/checkpoints/patch/patch-200/factor_2/epoch_1000_f_2.pth'

# ]

# names = ['ph1_l1_ssim_ema','ph1_l1_ssim_v2', 'ema', 'ema']
# checkpoints = [
#     'outputs/esrt/esrt_factor_2_l1_ssim_all_degradation_version2/checkpoints/patch/patch-200/factor_2/ema_beta_0.9.pth',
#     'outputs/esrt/esrt_factor_2_l1_ssim_all_degradation_version2/checkpoints/patch/patch-200/factor_2/epoch_2500_f_2.pth',
#     'outputs/esrt_gan/esrt_standardgan(l1_ssim_tv_pyramid)_(l1_ssim_tv_pyramid)_ph2/checkpoints/patch/patch-200/factor_2/ema_beta_0.9.pth',
#     'outputs/esrt_gan/esrt_lsgan(l1_ssim_tv_pyramid)_(l1_ssim_tv_pyramid)_ph2/checkpoints/patch/patch-200/factor_2/ema_beta_0.9.pth'

#     ]

# checkpoint_folder = 'outputs/esrt_gan/esrt_standardgan(l1_ssim_tv_pyramid)_(l1_ssim_tv_pyramid)_ph2/checkpoints/patch/patch-200/factor_2/'
# checkpoint_folder = 'outputs/esrt_gan/esrt_lsgan(l1_ssim_tv_pyramid)_(l1_ssim_tv_pyramid)_ph2/checkpoints/patch/patch-200/factor_2/'
# checkpoint_folder = 'outputs/esrt_gan/esrt_standardgan(l1)_(l1)_ph2_debug/checkpoints/patch/patch-200/factor_2/'
#### Get a list of all checkpoint files in the folder
# checkpoints = glob.glob(os.path.join(checkpoint_folder, '*.pth'))
# print(checkpoints)

# names = [chkpt.split('/')[7].split('_')[1] for chkpt in checkpoints]

# print(checkpoints)
# print(names)
# print(len(checkpoints))
# # quit()

idx = 8

parser = argparse.ArgumentParser(description='ESRT')
parser.add_argument("--test_lr_folder", type=str, default='../model_bias_experiment/mri_dataset_50/test',
                    help='the folder of the input images')
# parser.add_argument("--output_folder", type=str, default='results/'+ str(names[idx]))
# parser.add_argument("--checkpoint", type=str, default=checkpoints[idx],
#                     help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument("--upscale_factor", type=int, default=2,
                    help='upscaling factor')
parser.add_argument("--idx", type=int, default=8,
help='upscaling factor')
opt = parser.parse_args()

# print(opt)

opt.checkpoint = checkpoints[opt.idx]
print("Using checkpoint", opt.checkpoint)

opt.output_folder = 'results/'+ str(names[opt.idx])
# opt.output_folder = 'results/'+ 'lanczos'

if not os.path.exists(opt.output_folder):
    os.makedirs(opt.output_folder)

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
    
cuda = opt.cuda
opt.device = torch.device('cuda' if cuda else 'cpu')

filepath = opt.test_lr_folder

ext = '.png'


print('file path',filepath)
filelist = get_list(filepath, ext=ext)
timelist=[]

def load_state_dict_func(path):

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
    # model = ESRT(upscale = opt.upscale_factor)#
    model1 = ESRT(
            upscale=checkpoint['upscale_factor'],
            n_feats=checkpoint['n_feats'],
            n_blocks=checkpoint['n_blocks'], 
            kernel_size=checkpoint['kernel_size']
            ).to(device=opt.device)

    model = ESRT(
            upscale=checkpoint['upscale_factor'],
            n_feats=checkpoint['n_feats'],
            n_blocks=checkpoint['n_blocks'], 
            kernel_size=checkpoint['kernel_size']
            ).to(device=opt.device)
    model_dict = load_state_dict_func(checkpoint['model_state_dict'])
    # print(model_dict.keys())
    print(model_dict['reduce.bias'])
    print("*****************************************************************************************************")
    # print(model['reduce.bias'])

    model.load_state_dict(model_dict,strict=True)

    print("Random model parameter")
    for name, param in model1.named_parameters():
        if name == 'reduce.bias':
            print(f"Parameter name: {name}")
            print("Parameter values:")
            print(param)
    print("****************************************************************************")
    print("LOaded model parameter")
    for name, param in model.named_parameters():
            if name == 'reduce.bias':
                print(f"Parameter name: {name}")
                print("Parameter values:")
                print(param)

    return model

model = load_model(opt)
# model = model = ESRT(
#             upscale=2,
#             n_feats=16,
#             n_blocks=2, 
#             kernel_size=3
#             ).to(device=opt.device)

i = 0
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print(filelist)

print("starting  a loop")

# using model
# for imname in filelist:
#     im_lr = cv2.imread(imname, 0) 
#     im_lr = utils.modcrop(im_lr, opt.upscale_factor)

#     ##### im_lr= im_lr[:200,:200]

#     im_input = im_lr / 255.0
#     ##### im_input = im_input[np.newaxis, ...]
#     im_input = torch.from_numpy(im_input).float()
#     im_input = im_input.unsqueeze(0)
#     im_input = im_input.unsqueeze(0)
#     print("input shape", im_input.shape)

#     if cuda:
#         model = model.to(opt.device)
#         im_input = im_input.to(opt.device)

#     print ("Reached here")
#     with torch.no_grad():
#         out = forward_chop(model, im_input) #model(im_input)
#         print("input shape and output shape",im_input.shape, out.shape)
#         torch.cuda.synchronize()

#     out_img = utils.tensor2np(out.detach()[0])
#     plt.imshow(out_img)
#     crop_size = opt.upscale_factor
#     cropped_sr_img = utils.shave(out_img, crop_size)
#     cropped_lr_img = utils.shave(im_lr, crop_size)

#     output_image_folder = os.path.join(opt.output_folder,'images/')
#     if not os.path.exists(output_image_folder):
#         os.makedirs(output_image_folder)

#     output_image_path = os.path.join(output_image_folder,
#                                   imname.split('/')[-1].split('.')[0] + '.png')
#     cv2.imwrite(output_image_path, out_img)

#     i=2
#     input_image = im_input.squeeze().detach().cpu().numpy().astype('float')
#     output_image = out.squeeze().detach().cpu().numpy().astype('float')


#     input_image = input_image[20:60,20:60]
#     output_image = output_image[40:120,40:120]

#     #### input_image = input_image[20:160,20:160]
#     #### output_image = output_image[40:320,40:320]

#     fig = plt.figure()
#     ax1 = fig.add_subplot(1,i,1)
#     ax1.imshow(input_image, cmap='gray')
#     ax1.set_title("Input image")


#     ax2 = fig.add_subplot(1,i,2)
#     ax2.imshow(output_image,cmap='gray')
#     ax2.set_title("Output image")

#     output_plot_folder = os.path.join(opt.output_folder,'plots')
#     if not os.path.exists(output_plot_folder):
#         os.makedirs(output_plot_folder)

#     image_path = os.path.join(output_plot_folder,
#                                  imname.split('/')[-1].split('.')[0] + '.png')

#     # Save the full figure...
#     fig.savefig(image_path)
#     print("Image saved at", image_path)

#     i += 1


# using interpolation based method
for imname in filelist:
    im_lr = cv2.imread(imname, 0) 
    im_lr = utils.modcrop(im_lr, opt.upscale_factor)

    #using interpolation based method
    lr_image = im_lr.astype(np.uint8)
    lr_image = Image.fromarray(lr_image)
    shape = (lr_image.size[0]*2, lr_image.size[1]*2)
    im_pred = lr_image.resize(shape,Image.LANCZOS)
    im_pred = np.array(im_pred).astype('float32')


    output_image_folder = os.path.join(opt.output_folder,'images/')
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    output_image_path = os.path.join(output_image_folder,
                                  imname.split('/')[-1].split('.')[0] + '.png')
                              
    cv2.imwrite(output_image_path, im_pred)


    i=2

    input_image = im_lr[20:60,20:60]
    output_image = im_pred[40:120,40:120]

    # input_image = input_image[20:160,20:160]
    # output_image = output_image[40:320,40:320]

    fig = plt.figure()
    ax1 = fig.add_subplot(1,i,1)
    ax1.imshow(input_image, cmap='gray')
    ax1.set_title("Input image")


    ax2 = fig.add_subplot(1,i,2)
    ax2.imshow(output_image,cmap='gray')
    ax2.set_title("Output image")

    output_plot_folder = os.path.join(opt.output_folder,'plots')
    if not os.path.exists(output_plot_folder):
        os.makedirs(output_plot_folder)

    image_path = os.path.join(output_plot_folder,
                                 imname.split('/')[-1].split('.')[0] + '.png')

    # Save the full figure...
    fig.savefig(image_path)
    print("Image saved at", image_path)

    i += 1


