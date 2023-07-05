
import torch
from torch import nn 

class TVRegularizer(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, images):
        # print(images.shape)
        # print('reached here')
        loss  = (torch.sum(torch.abs(images[:,:,:,:-1]-images[:,:,:,1:]))+ torch.sum(torch.abs(images[:,:,:-1,:]-images[:,:,1:,:])))/(images.shape[0]*images.shape[1]*images.shape[2]*images.shape[3])
        return loss
