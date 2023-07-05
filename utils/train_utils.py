
# from skimage.metrics import peak_signal_noise_ratio
# from skimage.metrics import structural_similarity

import torch
# from utils.preprocess import 
import torch.nn as nn
from dataset.dataset import MRIDataset
import torch.optim as optim
# from loss.content_loss import ContentLoss
import pickle
from loss.ssim_loss import SSIM

def read_dictionary(dir_dict):
    '''Read annotation dictionary pickle'''
    a_file = open(dir_dict, "rb")
    output = pickle.load(a_file)
    # print(output)
    a_file.close()
    return output

''' set the dataset path based on opt.dataset,opt.factor values and load & return the same dataset/dataloader'''
def load_dataset(opt, load_eval=True):
    train_dataloader,train_datasets =load_train_dataset(opt)
    if load_eval:
        eval_dataloader,val_datasets = load_val_dataset(opt)
        return train_dataloader,train_datasets,eval_dataloader,val_datasets
    else: 
        return train_dataloader,train_datasets


def load_train_dataset(opt):
 
    train_datasets = MRIDataset(label_dir = opt.train_label_dir,  
                            scale_factor= opt.factor,
                            downsample_method= opt.downsample_method,
                            patch_size=opt.patch_size,
                            augment=opt.augment,
                            normalize=opt.normalize)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size = opt.train_batch_size,shuffle=True,
        num_workers=8,pin_memory=False,drop_last=False)

    return train_dataloader,train_datasets


def load_val_dataset(opt):

    val_datasets = MRIDataset(label_dir = opt.eval_label_dir,  
                            scale_factor = opt.factor,
                            downsample_method = opt.downsample_method,
                            patch_size = 100,
                            augment = False,
                            normalize = opt.normalize)
    eval_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size = opt.eval_batch_size,shuffle=True,
        num_workers=1,pin_memory=False,drop_last=False)
    return eval_dataloader,val_datasets


'''reduce learning rate of optimizer by half on every  150 and 225 epochs'''
def adjust_learning_rate(optimizer, epoch,lr,lr_factor=0.5):
    if lr <= 0.0000001:
        return lr
    else:
        if epoch % 50 == 0 or epoch % 150 == 0:
            lr = lr * lr_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr




'''get the optimizer based on opt.criterion value'''
def get_criterion(opt):
    if opt.criterion in ['mse']:
        print("Using MSE Loss")
        criterion = nn.MSELoss()
    elif opt.criterion in ['l1']:
        criterion = nn.L1Loss()
        print("Using L1 lOSS")
    elif opt.criterion in ['SSIM','ssim']:
        print("Using SSIM loss")
        criterion  = SSIM()
    else:
        print("Criterion not implemented")
        criterion = None
    return criterion




'''get the optimizer based on opt.optimizer value'''
def get_optimizer(optimizer_type,lr,model, betas=(0.9,0.999), eps=1e-08, weight_decay=0, momentum =0, alpha=0.99 ):
    if optimizer_type in ['adam','ADAM']:
        optimizer = optim.Adam(model.parameters(), lr= lr, betas= betas, eps=1e-08, weight_decay= weight_decay)
        print('Using ADAm Optimizer')
        return optimizer
    elif optimizer_type in ['sgd','SGD']:
        print('Using SGD Optimizer')
        optimizer = optim.SGD(model.parameters(), lr= lr, momentum= momentum, weight_decay= weight_decay)
        return optimizer   
    elif optimizer_type in ['rms','rmsprop','RMSprop']:
        print('Using SGD Optimizer')
        optimizer = torch.optim.RMSprop(model.parameters(), lr= lr, alpha=alpha, weight_decay=weight_decay, momentum=momentum)
        return optimizer
    else:
        print(f'optimizer type {optimizer_type} not found')
        return None



# torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0,