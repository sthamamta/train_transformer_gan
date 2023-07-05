import argparse
from dataset.dataset import MRIDataset
import sys
import torch
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, 
default='yaml/srdense.yaml')
opt  = parser.parse_args()

opt.label_dir = 'mri_dataset/val'
opt.scale_factor = 2
opt.downsample_method=['bicubic','hamming']
opt.patch_size = 120
opt.augment = True
opt.normalize = True
opt.batch_size = 1

train_datasets = MRIDataset( 
                            label_dir = opt.label_dir,  
                            scale_factor= opt.scale_factor,
                            downsample_method= opt.downsample_method,
                            patch_size=opt.patch_size,
                            augment=opt.augment,
                            normalize=opt.normalize  )

train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size = opt.batch_size,shuffle=False,
    num_workers=8,pin_memory=False,drop_last=False)


for index, data in enumerate(train_dataloader):
    lr_image = data['lr']
    hr_image = data['hr']

    print("range of lr", lr_image.min(), lr_image.max())
    print("range of hr", hr_image.min(), hr_image.max())

    lr_image =  lr_image.squeeze().cpu().numpy().astype('float')
    hr_image = hr_image.squeeze().cpu().numpy().astype('float')

    print("shape of lr", lr_image.shape)
    print("shape of hr", hr_image.shape)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(lr_image,cmap='gray')
    ax1.set_title("Input image")


    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(hr_image,cmap='gray')
    ax2.set_title("Label image")


    # Save the full figure...
    fig.savefig('image_plot_'+str(index)+'.png')