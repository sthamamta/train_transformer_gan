import torch
import  os
from torch.utils.data import Dataset
import numpy as np

from utils.preprocess import min_max_normalize
import cv2
import random
from .dataset_utils import prepare_lr_image


class MRIDataset(Dataset):
    def __init__(self, label_dir,transform=None, scale_factor = 2,downsample_method=['bicubic'],patch_size = None, augment= True, normalize=True):
        self.label_dir = label_dir
        self.transform = transform
        self.labels = os.listdir(label_dir)
        self.downsample_methods = downsample_method
        self.patch_size = patch_size
        self.scale_factor = scale_factor

        self.normalize = normalize
        self.augment = augment
        self.length_of_label_image =  len(self.labels)

        self.downsample_list = []
        self.labels_list = self.labels
        print("length of labels list", len(self.labels_list))
        for downsample_method in self.downsample_methods:
            downsample_list_repeat = [downsample_method]* ((self.length_of_label_image)//len(self.downsample_methods))
            self.downsample_list += downsample_list_repeat
            # self.labels_list += self.labels

        if len(self.downsample_list) != len(self.labels_list):
            downsample_m = random.choice(self.downsample_methods) 
            downsample_len =  self.length_of_label_image - ((self.length_of_label_image)//len(self.downsample_methods)*len(self.downsample_methods) )
            downsample_list_repeat = [downsample_m] * downsample_len
            self.downsample_list += downsample_list_repeat
            
        print("length of downsample list is ",len(self.downsample_list))
        assert len(self.downsample_list) == len(self.labels_list), ('list of label images and list of downsample method should have the same length, but got '
                                                f'{len(self.downsample_list)} and {len(self.labels_list)}.')            

        random.shuffle(self.downsample_list)
        
    def __len__(self):
        return len(self.labels_list)

    def create_patch(self,img, patch_size):
        height, width = img.shape
        height_limit = height-patch_size-40
        width_limit  = width-patch_size-40
        height_index = np.random.randint(40, height_limit)
        width_index = np.random.randint(40, width_limit)
        img = img[height_index: height_index+patch_size, width_index:width_index+patch_size]
        return img

    def extract_patch_hr_lr(self,lr_image, hr_image, lr_patch_size):
        lr_height, lr_width = lr_image.shape
        hr_height, hr_width = hr_image.shape
        
        # Generate a random index for the patch
        rand_y = np.random.randint(0, lr_height - lr_patch_size)
        rand_x = np.random.randint(0, lr_width - lr_patch_size)

        # Convert to integers
        rand_y = int(rand_y)
        rand_x = int(rand_x)

        lr_patch_size = int(lr_patch_size)
            
        # Extract the patch from LR image
        lr_patch = lr_image[rand_y:rand_y + lr_patch_size, rand_x:rand_x + lr_patch_size]
        
        # Calculate the corresponding HR patch index
        hr_patch_size = lr_patch_size * 2  # Upscale factor of 2
        hr_start_y = rand_y * 2
        hr_start_x = rand_x * 2
        
        # Extract the patch from HR image
        hr_patch = hr_image[hr_start_y:hr_start_y + hr_patch_size, hr_start_x:hr_start_x + hr_patch_size]
        
        return lr_patch, hr_patch

    def augment_image(self,img, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        if hflip: img = img[:, ::-1]
        if vflip: img = img[::-1, :]
        if rot90: img = img.transpose(1, 0)

        return img

    def __getitem__(self, index):

        label_path = os.path.join(self.label_dir, self.labels_list[index])
        downsample_method = self.downsample_list[index]

        # print("reading input image from :", input_path)
        # print("reading label image from :", label_path)
        # print(downsample_method)

        #read hr image
        hr_image = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        # if self.augment:
        #     hr_image = self.augment_image(hr_image,True,True)


        # ***************************************************************************************************

        # create hr patch
        if self.patch_size is not None:
            hr_image = self.create_patch(hr_image, self.patch_size)

        if self.augment:
            hr_image = self.augment_image(hr_image,True,True)


        #********************************************************************************************************

        # create lr image
        lr_image = prepare_lr_image(hr_image,downsample_method, self.scale_factor)

        # if self.patch_size is not None:
        #     lr_image, hr_image = self.extract_patch_hr_lr(lr_image, hr_image, self.patch_size/self.scale_factor)

        hr_image = hr_image[:lr_image.shape[0]* self.scale_factor,:lr_image.shape[1]* self.scale_factor]
        

        #normalize input and label image
        if self.normalize:
            lr_image = min_max_normalize(lr_image)
            hr_image = min_max_normalize(hr_image)

        hr_image = torch.from_numpy(hr_image)
        lr_image = torch.from_numpy(lr_image)

        if self.transform is not None:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        
        # adding the channel dimension
        lr_image = torch.unsqueeze(lr_image.float(),0)
        hr_image = torch.unsqueeze(hr_image.float(),0)

        # print("lr_image shape", lr_image.shape)
        # print("hr image shape", hr_image.shape)
    
        return {'lr': lr_image, 'hr':hr_image, 'index': index}

