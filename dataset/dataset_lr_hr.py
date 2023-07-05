import torch
import  os
from torch.utils.data import Dataset
import numpy as np
from utils.preprocess import min_max_normalize
import cv2
import random
from .dataset_utils import prepare_lr_image
import pickle

# dictionary_path = '../../model_bias_experiment/mri_dataset_50/lr_hr_dictionary.pkl'


def read_dictionary(dictionary_path= ''):
    with open(dictionary_path, 'rb') as handle:
        image_dictionary = pickle.load(handle)
    return image_dictionary


class MRIDataset(Dataset):
    def __init__(self, label_dir, input_dir, lr_patch_size= 100, dictionary_path = '../../model_bias_experiment/mri_dataset_50/lr_hr_dictionary.pkl', transform=None, scale_factor = 2,patch_size = None, augment= False, normalize=True):
        self.label_dir = label_dir
        self.input_dir = input_dir
        self.transform = transform
        self.dictionary_path = dictionary_path
        self.lr_patch_size = lr_patch_size
        self.image_dictionary = read_dictionary(self.dictionary_path)
        # print("length of dictionary", len(self.image_dictionary))
        self.prepare_list()

        self.patch_size = patch_size
        self.scale_factor = scale_factor

        self.normalize = normalize
        self.augment = augment
        self.length_of_label_image =  len(self.labels)
        self.length_of_input_image =  len(self.inputs)

        self.labels_list = self.labels
        self.input_list = self.inputs
    
    def prepare_list(self):
        self.inputs = []
        self.labels = []
        for key,value in self.image_dictionary.items():
            self.inputs.append(key)
            self.labels.append(value)


    def __len__(self):
        return len(self.inputs)

    def create_patch(self,img, patch_size):
        height, width = img.shape
        height_limit = height-patch_size-20
        width_limit  = width-patch_size-20
        height_index = np.random.randint(20, height_limit)
        width_index = np.random.randint(20, width_limit)
        img = img[height_index: height_index+patch_size, width_index:width_index+patch_size]
        return img

    def create_patch_lr_hr(self,img_hr,img_lr, patch_size):
        height, width = img_lr.shape
        height_limit = height-patch_size-20
        width_limit  = width-patch_size-20
        height_index = np.random.randint(20, height_limit)
        width_index = np.random.randint(20, width_limit)
        hr_height_index = 2*height_index
        hr_width_index = 2* width_index
        hr_patch_size = patch_size *2
        img_lr = img_lr[height_index: height_index+patch_size, width_index:width_index+patch_size]
        img_hr = img_hr[hr_height_index: hr_height_index+hr_patch_size, hr_width_index:hr_width_index+hr_patch_size]
        return img_lr,img_hr

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
        input_path = os.path.join(self.input_dir,self.input_list[index])

        #read hr image
        hr_image = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        lr_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

        # print('index is', index)
        # print("reading input image from :", input_path)
        # print("reading label image from :", label_path)

        #create hr patch
        if self.patch_size is not None:
            # hr_image = self.create_patch(hr_image, self.lr_patch_size*self.scale_factor)
            # lr_image = self.create_patch(lr_image, self.lr_patch_size)
            lr_image, hr_image = self.create_patch_lr_hr(img_hr=hr_image, img_lr= lr_image, patch_size= self.lr_patch_size)

        # if self.augment:
        #     hr_image = self.augment_image(hr_image,True,True)
        #     lr_image = self.augment_image(lr_image,True,True)


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

        # print ("lr shape", lr_image.shape)
        # print("hr shape", hr_image.shape)

        return {'lr': lr_image, 'hr':hr_image}

