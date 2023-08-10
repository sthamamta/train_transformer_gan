import torch
import os
import glob
from models.esrt import  ESRT

from collections import OrderedDict
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

def load_model(checkpoint, device='cpu'):
    checkpoint = torch.load(checkpoint,map_location=torch.device(device))
    model = ESRT(upscale = 2)#
    model = ESRT(
            upscale=checkpoint['upscale_factor'],
            n_feats=checkpoint['n_feats'],
            n_blocks=checkpoint['n_blocks'], 
            kernel_size=checkpoint['kernel_size']
            ).to(device=device)
    model_dict = load_state_dict_func(checkpoint['model_state_dict'])
    model.load_state_dict(model_dict,strict=False)
    return model


def exponential_moving_average(model, ema_model, beta):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(beta).add_((1 - beta) * param.data)

# Set the folder path where the checkpoints are saved
checkpoint_folders = [
        'outputs/esrt_gan/esrt_factor_2_l1_ssim_all_degradation/checkpoints/patch/patch-200/factor_2/',
        'outputs/esrt_gan/esrt_gan_factor_2_standardgan(l1_ssim)_(l1_ssim)_all_degradation/checkpoints/patch/patch-200/factor_2/',
        'outputs/esrt_gan/esrt_gan_factor_2_standard_gan_all_degradation_25_50_micron/checkpoints/patch/patch-200/factor_2/',
        'outputs/esrt_gan/esrt_gan_factor_2_standardgan_(l1_ssim_tv)_all_degradation/checkpoints/patch/patch-200/factor_2/',
        'outputs/esrt_gan/esrt_gan_factor_2_lsgan_25_50_micron/checkpoints/patch/patch-200/factor_2/',
        'outputs/esrt/esrt_factor_2_l1_ssim_all_degradation_version2/checkpoints/patch/patch-200/factor_2/',
        'outputs/esrt_gan/esrt_standardgan(l1_ssim_tv_pyramid)_(l1_ssim_tv_pyramid)_ph2/checkpoints/patch/patch-200/factor_2/',
        'outputs/esrt_gan/esrt_lsgan(l1_ssim_tv_pyramid)_(l1_ssim_tv_pyramid)_ph2/checkpoints/patch/patch-200/factor_2/',
        'outputs/esrt/esrt_l1_ssim_ph1/checkpoints/patch/patch-200/factor_2/'
]
checkpoint_folder = checkpoint_folders[8]
# checkpoint_folder = 'outputs/esrt_gan/esrt_gan_factor_2_standard_gan_checkpoint_ph1(l1,ssim)_25_50_micron/checkpoints/patch/patch-200/factor_2/'

# Get a list of all checkpoint files in the folder
checkpoint_files = glob.glob(os.path.join(checkpoint_folder, '*.pth'))
print("checkpoint files", checkpoint_files)

# Initialize the EMA model with the first checkpoint
ema_model = load_model(checkpoint_files[0])
ema_model.eval()

# Set the beta value for the EMA calculation (typically between 0.9 and 0.99)
beta = 0.9

# Load and process remaining checkpoints
for checkpoint_file in checkpoint_files[1:]:
    print("inside folder")
    model = load_model(checkpoint_file)
    model.eval()

    # Update the EMA model parameters
    exponential_moving_average(model, ema_model, beta)

# Save the final EMA checkpoint
save_path = os.path.join(checkpoint_folder,'ema_beta_0.9.pth')
ema_model.save(model_weights=ema_model.state_dict(),path= save_path,epoch='ema')
