
import torch
from torch import nn 
# code from https://notebook.community/zklgame/CatEyeNets/test/StyleTransfer-PyTorch

class TVRegularizer(nn.Module):
    """
    Compute total variation loss.
    
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, images):
        # loss  = (torch.sum(torch.abs(images[:,:,:,:-1]-images[:,:,:,1:]))+ torch.sum(torch.abs(images[:,:,:-1,:]-images[:,:,1:,:])))/(images.shape[0]*images.shape[1]*images.shape[2]*images.shape[3])
        N, C, H, W = images.size()
        loss = torch.sum(torch.pow(images[:, :, :H - 1, :] - images[:, :, 1:, :], 2))
        loss += torch.sum(torch.pow(images[:, :, :, :W - 1] - images[:, :, :, 1:], 2)) 
        return loss


