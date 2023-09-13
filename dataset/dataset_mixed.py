import torch
import  os
from torch.utils.data import Dataset
import numpy as np

from utils.preprocess import min_max_normalize
import cv2
import random
from .dataset_utils import prepare_lr_image
import pickle

def read_dictionary(dictionary_path= ''):
    with open(dictionary_path, 'rb') as handle:
        image_dictionary = pickle.load(handle)
    return image_dictionary

class MRIDataset(Dataset):
    def __init__(self, label_dir,real_label_dir, real_input_dir,dictionary_path = '../../model_bias_experiment/mri_dataset_50/lr_hr_dictionary.pkl', transform=None, scale_factor = 1,downsample_method=['bicubic'],patch_size = None, augment= True, normalize=True, lr_patch_size=120):
        
        
        self.label_dir = label_dir
        self.real_label_dir = real_label_dir
        self.real_input_dir = real_input_dir
        self.dictionary_path = dictionary_path
        self.image_dictionary = read_dictionary(self.dictionary_path)
        self.image_dictionary = {value: key for key, value in self.image_dictionary.items()}  # now label is the key
    
        self.prepare_list()  # we have a list of labels and input images full path

        self.transform = transform
        self.lr_patch_size = lr_patch_size


        self.labels_list = os.listdir(label_dir)
        self.labels_list = [os.path.join(self.label_dir,label) for label in self.labels_list]


        self.downsample_methods = downsample_method
        self.patch_size = patch_size
        self.scale_factor = scale_factor

        self.normalize = normalize
        self.augment = augment
        self.length_of_label_image =  len(self.labels_list)

        self.downsample_list = []
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

        self.image_and_downsample_dictionary = {}
        # concatenate labels full path and degradation into dictionary:
        for dowsample_method,image_path in zip (self.downsample_list, self.labels_list):
            self.image_and_downsample_dictionary[image_path]= dowsample_method

        # append degradation method as real for real dataset
        for real_labels in self.real_labels:
            self.image_and_downsample_dictionary[real_labels] = 'real'

        # prepare dataset of real label and real input to load corresponding images
        self.real_dataset_dictionary = {}
        for real_labels, real_input in zip(self.real_labels, self.real_inputs):
            self.real_dataset_dictionary[real_labels] = real_input
            # print(real_labels)
            

        self.keys_list = list(self.image_and_downsample_dictionary.keys())
        print("The length of dataset is ", len(self.keys_list))
        # print(self.keys_list)
        print(self.image_and_downsample_dictionary)

        # quit();


    def prepare_list(self):
        self.real_inputs = []
        self.real_labels = []
        for key,value in self.image_dictionary.items():
            key = os.path.join(self.real_label_dir, key)
            value = os.path.join(self.real_input_dir,value)
            self.real_inputs.append(value)
            self.real_labels.append(key)

    def __len__(self):
        return len(self.image_and_downsample_dictionary)

    def create_patch(self,img, patch_size):
        height, width = img.shape
        height_limit = height-patch_size-40
        width_limit  = width-patch_size-40
        height_index = np.random.randint(40, height_limit)
        width_index = np.random.randint(40, width_limit)
        img = img[height_index: height_index+patch_size, width_index:width_index+patch_size]
        return img

    def create_patch_lr_hr(self,img_hr,img_lr, patch_size):
            height, width = img_lr.shape
            height_limit = height-patch_size-20
            width_limit  = width-patch_size-20

            height_index = np.random.randint(20, height_limit)
            width_index = np.random.randint(20, width_limit)
            
            img_lr = img_lr[height_index: height_index+patch_size, width_index:width_index+patch_size]
            if self.scale_factor==1:
                img_hr = img_hr[height_index: height_index+ patch_size, width_index:width_index+ patch_size]
            else:
                hr_height_index = 2*height_index
                hr_width_index = 2* width_index
                hr_patch_size = patch_size *2
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

        label_path = self.keys_list[index]
        downsample_method = self.image_and_downsample_dictionary[label_path]

        # print("reading label image from :", label_path)
        # print("reading ldownsample method :", downsample_method)
        # print(downsample_method)

        #read hr image
        hr_image = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        # if self.augment:
        #     hr_image = self.augment_image(hr_image,True,True)

        # ***************************************************************************************************

        if downsample_method == 'real':
            input_path = self.real_dataset_dictionary[label_path]
            # print("reading input image from :", input_path)
            lr_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

            if self.scale_factor==1:
                x, y = hr_image.shape
                lr_image = cv2.resize(lr_image,(y,x),  interpolation=cv2.INTER_CUBIC)

            if self.patch_size is not None:
                lr_image, hr_image = self.create_patch_lr_hr(img_hr=hr_image, img_lr= lr_image, patch_size= self.lr_patch_size)

        
        else:
            # create hr patch
            if self.patch_size is not None:
                hr_image = self.create_patch(hr_image, self.patch_size)

            if self.augment:
                hr_image = self.augment_image(hr_image,True,True)

            #********************************************************************************************************
            # create lr image
            lr_image = prepare_lr_image(hr_image,downsample_method, self.scale_factor)


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

        # print("image path", label_path)
        # print("lr_image shape", lr_image.shape)
        # print("hr image shape", hr_image.shape)
        # print("******************************************************************************************************")
    
        return {'lr': lr_image, 'hr':hr_image, 'index': index}

