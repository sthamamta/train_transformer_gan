import torch
from models.transformer_discriminator import Discriminator
from models.esrt import ESRT
import torch.nn as nn
from collections import OrderedDict
import cv2
import matplotlib.pyplot as plt
import os

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
    model = ESRT(
            upscale=checkpoint['upscale_factor'],
            n_feats=checkpoint['n_feats'],
            n_blocks=checkpoint['n_blocks'], 
            kernel_size=checkpoint['kernel_size']
            ).to(device=device)
    model_before= model
    print("checkpoint weights", checkpoint['model_state_dict']['body.0.encoder1.encoder.layer2.conv.bias'])
    model_dict = load_state_dict_func(checkpoint['model_state_dict'])
    model.load_state_dict(model_dict,strict=False)
    return model

checkpoint_path = 'outputs/esrt_gan/esrt_gan_factor_2_11_loss_gan_Loss_all_degradation/checkpoints/patch/patch-200/factor_2/epoch_950_f_2.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = load_model(checkpoint_path=checkpoint_path,device=device)

sample_model= ESRT(upscale=2,n_feats=16,n_blocks=1,kernel_size=3).to(device=device)
sample_model2= ESRT(upscale=2,n_feats=16,n_blocks=1,kernel_size=3)
sample_model3= ESRT(upscale=2,n_feats=16,n_blocks=1,kernel_size=3)
sample_model.eval()

image_path = '../model_bias_experiment/mri_dataset_25/train/hr_f1_160_z_148.png'
generator.eval()
input = cv2.imread(image_path,0)
input = input/255.0
input_tensor = torch.from_numpy(input[:100,:100]).unsqueeze(0).unsqueeze(0).float().to(device=device)
# input = torch.rand(1,1,200,200).to(device=device)
output1 = generator(input_tensor)
output2 = sample_model(input_tensor)


output1 = output1.squeeze().detach().cpu().numpy().astype('float')*255.
output2 = output2.squeeze().detach().cpu().numpy().astype('float')*255.



i=2
fig = plt.figure()
ax1 = fig.add_subplot(1,i,1)
ax1.imshow(output1, cmap='gray')
ax1.set_title("output from checkpoint")


ax2 = fig.add_subplot(1,i,2)
ax2.imshow(output2,cmap='gray')
ax2.set_title("Output from intiial")

# Save the full figure...
fig.savefig('sample_plot.png')



quit();

for name, param in generator.named_parameters():
    # print(f"Parameter name: {name}")
    # print(f"Parameter shape: {param.shape}")
    if name == 'body.0.encoder1.encoder.layer2.conv.bias': 
        print("sample 1 generator")      
        print(f"Parameter values: {param}")
    # print("------------------------------------")

for name, param in sample_model.named_parameters():
    # print(f"Parameter name: {name}")
    # print(f"Parameter shape: {param.shape}")
    if name == 'body.0.encoder1.encoder.layer2.conv.bias':       
        print(f"Parameter values: {param}")
    # print("------------------------------------")

are_weights_same = all(torch.equal(p1, p2) for p1, p2 in zip(generator.parameters(), generator_before.parameters()))

if are_weights_same:
    print("The weights are the same.")
else:
    print("The weights are different.")