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
from models.transformer_discriminator import RankDiscriminator
from dataset.dataset_utils import prepare_lr_image

def get_list(path, ext):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]
  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

filepath = '../model_bias_experiment/mri_dataset_25/test'
checkpoint_path = 'outputs/transformer_classifier/transformer_classifier_25_micron_gaussian/checkpoints/patch/patch-200/epoch_3480_f_2.pth'

ext = '.png'


# print('file path',filepath)
filelist = get_list(filepath, ext=ext)

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


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path,map_location=torch.device(device))
    model = RankDiscriminator(
           patch_size = checkpoint['patch_size'], 
           in_chans = checkpoint['in_chans'], 
           num_classes = checkpoint['num_classes'],
           embed_dim = checkpoint['embed_dim'],
           num_heads = checkpoint['num_heads'], 
           mlp_ratio = checkpoint['mlp_ratio'],
           qkv_bias = checkpoint['qkv_bias'],
           qk_scale = checkpoint['qk_scale'], 
           drop_rate = checkpoint['drop_rate'],
           norm_layer = checkpoint['norm_layer'],
           depth = checkpoint['depth'],
           act_layer = checkpoint['act_layer'],
           diff_aug = checkpoint['diff_aug'],
           apply_sigmoid = checkpoint['apply_sigmoid']
            ).to(device = device)

    model_dict = load_state_dict_func(checkpoint['model_state_dict'])
    model.load_state_dict(model_dict,strict=False)
    return model

model = load_model(checkpoint_path=checkpoint_path, device=device)


i = 0

# print(filelist)

prediction_list = []
output_list = []

def create_patch(img, patch_size):
        _,_,height, width = img.shape
        print(height, width)
        height_limit = height-patch_size-10
        width_limit  = width-patch_size-10
        height_index = np.random.randint(10, height_limit)
        width_index = np.random.randint(10, width_limit)
        # height_limit = height-patch_size-40
        # width_limit  = width-patch_size-40
        # height_index = np.random.randint(40, height_limit)
        # width_index = np.random.randint(40, width_limit)
        img = img[:,:,height_index: height_index+patch_size, width_index:width_index+patch_size]
        return img

def min_max_normalize(image):
    max_img = image.max()
    min_img = image.min()
    denom = max_img-min_img
    norm_image = (image-min_img)/denom
    return norm_image 

print("starting  a loop")
downsample_method = 'kspace_gaussian_50'
for imname in filelist:
    im_lr = cv2.imread(imname, cv2.IMREAD_UNCHANGED) 

    print('reached here')
    im_lr = prepare_lr_image(im_lr,downsample_method, 1)
    im_input = min_max_normalize(im_lr)


    # im_input = im_lr / 255.0
    im_input = torch.from_numpy(im_input).float()
    im_input = im_input.unsqueeze(0)
    im_input = im_input.unsqueeze(0)
    # print("input shape", im_input.shape)

    # im_lr= im_lr[:,:,:200,:200]
    im_input = create_patch(img=im_input, patch_size=200)

    im_input = im_input.to(device)
    model = model.to(device)
    
    with torch.no_grad():
        out = model(im_input)
        preds = torch.argmax(out, dim=1)
        prediction_list.append(preds.item())
        output_list.append(out)
          
    i += 1

print("******************************************************************************")
print(prediction_list)
# print(output_list)