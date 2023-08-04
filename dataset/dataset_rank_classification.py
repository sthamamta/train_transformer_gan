import torch
import  os
from torch.utils.data import Dataset
import numpy as np

from utils.preprocess import min_max_normalize
import cv2
import random
from .dataset_utils import prepare_lr_image


class MRIDataset(Dataset):
    def __init__(self, label_dir,transform=None, scale_factor = 1,downsample_method=['kspace_gaussian_125','kspace_gaussian_150','kspace_gaussian_75','kspace_gaussian_100','hr'],patch_size = None, augment= True, normalize=True):
        self.label_dir = label_dir
        self.transform = transform
        self.labels = os.listdir(label_dir)
        self.downsample_methods = downsample_method
        self.patch_size = patch_size
        self.scale_factor = scale_factor

        self.normalize = normalize
        self.augment = augment
        self.length_of_label_image =  len(self.labels)

        # self.downsample_list = []
        # self.labels_list = []
        # print("length of labels list", len(self.labels_list))

        # for downsample_method in self.downsample_methods:
        #     downsample_list_repeat = [downsample_method]* (self.length_of_label_image)
        #     self.downsample_list += downsample_list_repeat
        #     self.labels_list += self.labels

        self.downsample_list = downsample_method
        self.labels_list = self.labels
        self.n_sample = 1200
        while len(self.labels_list) < self.n_sample:
            random_sample = random.choice(self.labels_list)
            self.labels_list.append(random_sample)

        while len(self.downsample_list) < self.n_sample:
            random_sample = random.choice(self.downsample_list)
            self.downsample_list.append(random_sample)

        # print(self.labels_list)
        # print(self.downsample_list)
        print("length of downsample list is ",len(self.downsample_list))
        print("length of labels list", len(self.labels_list))
        assert len(self.downsample_list) == len(self.labels_list), ('list of label images and list of downsample method should have the same length, but got '
                                                f'{len(self.downsample_list)} and {len(self.labels_list)}.')            
        # quit();
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

    def extract_patch_lr(self,lr_image, lr_patch_size):
        lr_height, lr_width = lr_image.shape
        
        # Generate a random index for the patch
        rand_y = np.random.randint(0, lr_height - lr_patch_size)
        rand_x = np.random.randint(0, lr_width - lr_patch_size)

        # Convert to integers
        rand_y = int(rand_y)
        rand_x = int(rand_x)

        lr_patch_size = int(lr_patch_size)
            
        # Extract the patch from LR image
        lr_patch = lr_image[rand_y:rand_y + lr_patch_size, rand_x:rand_x + lr_patch_size]
       
        return lr_patch

    def augment_image(self,img, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        if hflip: img = img[:, ::-1]
        if vflip: img = img[::-1, :]
        if rot90: img = img.transpose(1, 0)

        return img

    def get_label(self, downsample_method):
        if downsample_method=='kspace_gaussian_50':
            label = torch.tensor(0)
        elif downsample_method == 'kspace_gaussian_75':
            label = torch.tensor(1)
        elif downsample_method == 'kspace_gaussian_100':
            label = torch.tensor(2)
        elif downsample_method == 'kspace_gaussian_125':
            label = torch.tensor(3)
        elif downsample_method == 'hr':
            label = torch.tensor(4)
        else:
            print ("Label value not found for downsample method ", downsample_method)
            label = None
        return label

    def __getitem__(self, index):

        label_path = os.path.join(self.label_dir, self.labels_list[index])
        downsample_method = self.downsample_list[index]
        label = self.get_label(downsample_method=downsample_method)

        #read hr image
        hr_image = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        if self.augment:
            hr_image = self.augment_image(hr_image,True,True)

        if downsample_method == 'hr':
            lr_image = hr_image
        else:
            # create lr image
            lr_image = prepare_lr_image(hr_image,downsample_method, self.scale_factor)

        if self.patch_size is not None:
            lr_image = self.extract_patch_lr(lr_image, self.patch_size)        

        #normalize input and label image
        if self.normalize:
            lr_image = min_max_normalize(lr_image)

        lr_image = torch.from_numpy(lr_image)

        if self.transform is not None:
            lr_image = self.transform(lr_image)
        
        # adding the channel dimension
        lr_image = torch.unsqueeze(lr_image.float(),0)


        # print("lr_image shape", lr_image.shape)
        # print("downsample method", downsample_method)
        # print("label value", label)
    
        return {'lr': lr_image, 'label':label, 'index': index}

