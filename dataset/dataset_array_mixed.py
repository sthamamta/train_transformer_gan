import torch
import  os
from torch.utils.data import Dataset
import numpy as np
from utils.preprocess import min_max_normalize
import cv2
import random
from .dataset_utils import prepare_lr_image
import pickle
import nibabel as nib
# dictionary_path = '../../model_bias_experiment/mri_dataset_50/lr_hr_dictionary.pkl'


def read_dictionary(dictionary_path= ''):
    with open(dictionary_path, 'rb') as handle:
        image_dictionary = pickle.load(handle)
    return image_dictionary


class MRIDataset(Dataset):
    def __init__(self, label_dir, input_dir, axis=2, lr_patch_size= 100, dictionary_path = '../../model_bias_experiment/mri_dataset_50/lr_hr_dictionary.pkl', transform=None, scale_factor = 2,downsample_methods=['bicubic'],patch_size = None, augment= False, normalize=True, upsample_image=True):
        
        self.label_dir = label_dir
        self.input_dir = input_dir
        self.transform = transform

        self.downsample_methods = downsample_methods

        self.dictionary_path = dictionary_path
        self.lr_patch_size = lr_patch_size

        self.image_dictionary = read_dictionary(self.dictionary_path)  # ditionary of lr:hr
        # print("length of dictionary", len(self.image_dictionary))
    
        self.hr_array , self.lr_array, self.index_dictionary =  self.prepare_array(labels_array_dir=self.label_dir, inputs_array_dir=self.input_dir)
        self.hr_image_index, self.lr_image_index = self.prepare_index() #contains lr index and hr index list
    
        self.downsample_list = ['real'] * len(self.hr_image_index) # creating the degradation list as real for real dataset
        
        original_list = list(range(20, 261))
        self.hr_simulated_index = original_list + [x + 304 for x in original_list] + [x + 608 for x in original_list]

        self.hr_image_index.extend(self.hr_simulated_index) # appending index for simulated dataset
        self.lr_image_index.extend([None]*len(self.hr_simulated_index)) # appending lr index as none for simulated dataset


        self.length_of_simulated_dataset = len (self.hr_simulated_index)
        self.downsample_list_simulated = []

        print("length of labels list", self.length_of_simulated_dataset)
        for downsample_method in self.downsample_methods:
            downsample_list_repeat = [downsample_method]* ((self.length_of_simulated_dataset)//len(self.downsample_methods))
            self.downsample_list_simulated += downsample_list_repeat
            # self.labels_list += self.labels

        if len(self.downsample_list_simulated) != self.length_of_simulated_dataset:
            downsample_m = random.choice(self.downsample_methods) 
            downsample_len =  self.length_of_simulated_dataset - ((self.length_of_simulated_dataset)//len(self.downsample_methods)*len(self.downsample_methods) )
            downsample_list_repeat = [downsample_m] * downsample_len
            self.downsample_list_simulated += downsample_list_repeat
            
        print("length of downsample list is ",len(self.downsample_list_simulated))
        assert len(self.downsample_list_simulated) == len(self.hr_simulated_index), ('list of label images and list of downsample method should have the same length, but got '
                                                f'{len(self.downsample_list_simulated)} and {len(self.hr_simulated_index)}.')            

        random.shuffle(self.downsample_list_simulated)

        self.downsample_list.extend(self.downsample_list_simulated)


        self.list_of_dataset = [(x, y, z) for x, y, z in zip(self.hr_image_index, self.lr_image_index, self.downsample_list)]
        random.shuffle(self.list_of_dataset)
    

        self.upsample_image = upsample_image

        self.axis = axis

        self.patch_size = patch_size
        self.scale_factor = scale_factor

        self.normalize = normalize
        self.augment = augment
        
        self.length_of_label_image =  len(self.image_dictionary)
        self.length_of_input_image =  len(self.image_dictionary)

        # print('The shape of HR array', self.hr_array.shape)
        # print('The shape of LR array', self.lr_array.shape)

        print("length of dataset", len(self.list_of_dataset))
    

    def prepare_index(self):
        hr_image_index = []
        lr_image_index = []
        for key, value in self.image_dictionary.items():
            lr_index = int(key.split('_')[4].split('.')[0])
            hr_index = int(value.split('_')[4].split('.')[0])

            multiplier = int(self.index_dictionary[key.split('_')[1]] )

            # print(key.split('_')[1], multiplier)

            lr_index = lr_index + (multiplier*152)
            hr_index = hr_index + (multiplier*304)
            hr_image_index.append(hr_index)
            lr_image_index.append(lr_index)
            
        return hr_image_index, lr_image_index

    def prepare_array(self, labels_array_dir, inputs_array_dir):
        labels_list = os.listdir(labels_array_dir)
        input_list = os.listdir(inputs_array_dir)

        hr_array_full = [[[]]]
        lr_array_full = [[[]]]
        index_dictionary = {}
        count = 0
        for index,(file_label_array,file_input_array)  in enumerate(zip(labels_list, input_list)):
            if file_label_array == 'f4_25.nii':
                pass
            else:
                label_file_path =  os.path.join(labels_array_dir, file_label_array)
                single_label_array = self.read_array(label_file_path)
                single_input_array =  self.read_array(os.path.join(inputs_array_dir, file_input_array) )

                print(single_input_array.shape)
                print(single_label_array.shape)
                print(count, file_label_array)
                if index == 0:
                    hr_array_full = single_label_array
                    lr_array_full = single_input_array
                else:
                    hr_array_full =  np.concatenate((hr_array_full, single_label_array), axis=2)
                    lr_array_full =  np.concatenate((lr_array_full, single_input_array), axis=2)
                
                index_dictionary[file_label_array.split('_')[0]] =  count
                count +=1

        return hr_array_full, lr_array_full, index_dictionary

    def read_array(self, file_path):
        img = nib.load(file_path)
        affine_mat=img.affine
        hdr = img.header
        data = img.get_fdata()
        data_norm = data
        return data_norm

    def __len__(self):
        return len(self.list_of_dataset)

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
        
        img_lr = img_lr[height_index: height_index+patch_size, width_index:width_index+patch_size]
        if self.upsample_image:
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

        hr_index, lr_index, downsample_method = self.list_of_dataset[index]

        hr_image =  self.hr_array[:,:,hr_index]

        # print("hr index", hr_index)
        # print("lr index", lr_index)
        # print("downsample method", downsample_method)
        # print("shape of hr array", self.hr_array.shape)
        # print("shape of hr image", hr_image.shape)
        # print(hr_image.min(), hr_image.max())

        if downsample_method == 'real':
            lr_image =  self.lr_array[:,:,lr_index]
            if self.upsample_image:
                x, y = hr_image.shape
                lr_image = cv2.resize(lr_image,(y,x),  interpolation=cv2.INTER_CUBIC)

            #create hr patch
            if self.patch_size is not None:
                lr_image, hr_image = self.create_patch_lr_hr(img_hr=hr_image, img_lr= lr_image, patch_size= self.lr_patch_size)

        else:
                # create hr patch
            if self.patch_size is not None:
                hr_image = self.create_patch(hr_image, self.patch_size)

            if self.augment:
                hr_image = self.augment_image(hr_image,True,True)

            lr_image = prepare_lr_image(hr_image,downsample_method, self.scale_factor)
    
            # lr_image, hr_image = self.create_patch_lr_hr(img_hr=hr_image, img_lr= lr_image, patch_size= self.lr_patch_size)
        
    
        if self.upsample_image:
            pass
        else:
            hr_image = hr_image[:lr_image.shape[0]* self.scale_factor,:lr_image.shape[1]* self.scale_factor]
        
        #normalize input and label image
        if self.normalize:
            lr_image = min_max_normalize(lr_image)
            hr_image = min_max_normalize(hr_image)
            # print("lr and hr image minmax", lr_image.min(), lr_image.max(), hr_image.min(), hr_image.max())


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
        # print("**********************************************************************************")

        return {'lr': lr_image, 'hr':hr_image}

