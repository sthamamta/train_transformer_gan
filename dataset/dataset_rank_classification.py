import torch
import  os
from torch.utils.data import Dataset
import numpy as np

from utils.preprocess import min_max_normalize
import cv2
import random
from .dataset_utils import prepare_lr_image


class MRIDataset(Dataset):
    def __init__(self, dataset_dir='classifier_dataset', transform=None, patch_size = None, augment= True, normalize=True):

        self.dataset_dir = dataset_dir
        self.transform = transform
        self.normalize = normalize
        self.augment = augment
        self.patch_size = patch_size
        self.image_path_list = self.get_image_list()

      
        self.length_of_image =  len(self.image_path_list)

        print("The length of dataset is", self.length_of_image)         
        # quit();


    def get_image_list(self):
        image_path_list = []
        for folder in range(5):  # Assuming subfolders are named 0, 1, 2, 3, and 4
            folder_path = os.path.join(self.dataset_dir, str(folder))
            print(folder_path)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        image_path_list.append(os.path.join(folder_path, filename))
        return image_path_list



    def __len__(self):
        return self.length_of_image


    def extract_patch(self,image, patch_size):
        lr_height, lr_width = image.shape
        
        # Generate a random index for the patch
        rand_y = np.random.randint(0, lr_height - patch_size)
        rand_x = np.random.randint(0, lr_width - patch_size)

        # Convert to integers
        rand_y = int(rand_y)
        rand_x = int(rand_x)

        patch_size = int(patch_size)
            
        # Extract the patch from LR image
        lr_patch = image[rand_y:rand_y + patch_size, rand_x:rand_x + patch_size]
       
        return lr_patch

    def augment_image(self,img, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        if hflip: img = img[:, ::-1]
        if vflip: img = img[::-1, :]
        if rot90: img = img.transpose(1, 0)

        return img

    def get_label(self, image_path):
        label = torch.tensor(int(image_path.split('/')[1]))
        return label


    def __getitem__(self, index):

        image_path = self.image_path_list[index]
        # print("image path is", image_path)
        label = self.get_label(image_path)
        # print("label is", label)

        #read hr image
        input_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if self.augment:
            input_image = self.augment_image(input_image,True,True)

    
        if self.patch_size is not None:
            input_image = self.extract_patch(input_image, self.patch_size)        

        #normalize input and label image
        if self.normalize:
            input_image = min_max_normalize(input_image)

        input_image = torch.from_numpy(input_image)

        if self.transform is not None:
            input_image = self.transform(input_image)
        
        # adding the channel dimension
        input_image = torch.unsqueeze(input_image.float(),0)


        # print("lr_image shape", input_image.shape)
        
    
        return {'lr': input_image, 'label':label, 'index': index}

