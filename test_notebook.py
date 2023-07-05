import torch
from dataset.dataset_lr_hr import MRIDataset
from torch.utils.data import Dataset, DataLoader

train_input_path= '../model_bias_experiment/mri_dataset_50/train'
train_label_path= '../model_bias_experiment/mri_dataset_25/train'
train_dictionary_path= '../model_bias_experiment/mri_dataset_50/lr_hr_dictionary.pkl'
patch_size=  200  # this is the patch size of hr therefore the lr patch size will be hr_patch_size/factor
lr_patch_size= 100
factor= 2
augment= False
train_batch_size = 8

class CustomDataset(Dataset):
    def __init__(self, data_length=100):
        self.data_length = data_length

        self.count = 0
        
    def __len__(self):
        return self.data_length
    
    def __getitem__(self, index):
        lr = torch.rand(1,200,200)
        hr = torch.rand(1,200,200)
        self.count += 1
        print("taking data", self.count)
        return {'input': lr, 'label': hr}


train_dataset = MRIDataset(label_dir = train_label_path,  
                        input_dir = train_input_path,
                        scale_factor = factor,
                        dictionary_path = train_dictionary_path,
                        patch_size = patch_size,
                        augment = augment)
# train_dataset = CustomDataset()
train_dataloader = torch.utils.data.DataLoader (train_dataset, batch_size = train_batch_size,shuffle=True,drop_last=False)



it = iter(train_dataloader)
first = next(it)
print ("************************************************************************************************************************")
second = next(it)
print ("************************************************************************************************************************")
third = next(it)
print ("************************************************************************************************************************")
fourth = next(it)
print ("************************************************************************************************************************")
fifth = next(it)