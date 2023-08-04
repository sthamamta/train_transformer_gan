import torch
from dataset.dataset import MRIDataset


train_label_dir = '../model_bias_experiment/mri_dataset_25/train'
patch_size = 200  # this is the patch size of hr therefore the lr patch size will be hr_patch_size/factor
lr_patch_size = 100
factor = 2

downsample_method = ['bicubic','nearest','bilinear','lanczos','kspace','kspace_gaussian_100','hanning', 'hamming']  
augment = True
normalize = True
train_batch_size = 8

train_datasets = MRIDataset(label_dir = train_label_dir,  
                            scale_factor= factor,
                            downsample_method= downsample_method,
                            patch_size= patch_size,
                            augment= augment,
                            normalize= normalize)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size = train_batch_size,shuffle=True,
        num_workers=8,pin_memory=False,drop_last=False)

count=0
count_main_iter =0
for idx, batch in enumerate(train_dataloader):
    # count += 1
    # batch_indices = [2,4]  # Indices of batches to load
    print(" outer index", batch['index'])
    i=0
    for inner_batch in train_dataloader:
        if i <= 2:
            print("inside loop")
            print(" inner index", inner_batch['index'])
            count +=1
        i += 1
    print(count)
    print(count_main_iter)
    print("***********************************************************")
    count_main_iter += 1

    # for idx in batch_indices:
    #     batch_data = next(iter(train_dataloader))[idx]
    #     print(batch_data.shape)
    #     count += 1
    #     # print(batch)

print(count)