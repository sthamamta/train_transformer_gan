# functions for loading the checkpoints


import torch.nn as nn
import torch
from models.densenet import SRDenseNet
from models.patch_gan import PatchGAN
from models.resunet import ResUNet
from models.unet import Unet, UnetSmall

from models.dense3d import SR3DDenseNet
from models.rrdbnet import RRDBNet
from models.rrdbnet3d import RRDBNet3D



def load_model(opt,model,checkpoint):
    state_dict = model.state_dict()
    for n, p in checkpoint['model_state_dict'].items():
        # new_key = n[7:]
        new_key = n
        if new_key in state_dict.keys():
            state_dict[new_key].copy_(p)
        else:
            raise KeyError(new_key)
    if opt.device != 'cpu':
        return data_parallel(opt,model)
    else:
         return model

def data_parallel(opt,model):
    num_of_gpus = torch.cuda.device_count()
    model = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])
    model.to(opt.device)
    return model

def load_rrdbnet(opt, load_3d=False):
    checkpoint = torch.load(opt.checkpoint,map_location=torch.device(opt.device))
    num_blocks=checkpoint['num_blocks']
    if load_3d:
        model = RRDBNet3D(num_block = num_blocks).to(opt.device)
    else:
        model = RRDBNet(num_block = num_blocks).to(opt.device)
    return load_model(opt,model,checkpoint)


def load_dense(opt,load_3d=False):
    checkpoint = torch.load(opt.checkpoint,map_location=torch.device(opt.device))
    growth_rate=checkpoint['growth_rate']
    num_blocks=checkpoint['num_blocks']
    num_layers=checkpoint['num_layers']
    if load_3d:
        model = SR3DDenseNet(num_channels=1, growth_rate=growth_rate, num_blocks = num_blocks, num_layers=num_layers).to(opt.device)
    else:
        model = SRDenseNet(num_channels=1, growth_rate=growth_rate, num_blocks = num_blocks, num_layers=num_layers).to(opt.device)

    return load_model(opt,model,checkpoint)


def load_resunet(opt):
    checkpoint = torch.load(opt.checkpoint,map_location=torch.device(opt.device))
    model = ResUNet().to(opt.device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    return load_model(opt,model,checkpoint)
    

def load_patch_gan(opt):
    checkpoint = torch.load(opt.checkpoint,map_location=torch.device(opt.device))
    model = PatchGAN()
    return load_model(opt,model,checkpoint)

def load_small_unet(opt):
    checkpoint = torch.load(opt.checkpoint,map_location=torch.device(opt.device))
    model = UnetSmall()
    return load_model(opt,model,checkpoint)


def load_unet(opt):
    checkpoint = torch.load(opt.checkpoint,map_location=torch.device(opt.device))
  
    n_blocks = checkpoint['n_blocks']
    start_filters = checkpoint['start_filters']
    activation = checkpoint['activation']
    normalization = checkpoint['normalization']
    conv_mode = checkpoint['conv_mode']
    dim = checkpoint['dim']
    up_mode= checkpoint['up_mode']

    model = Unet(in_channels= 1,
                 out_channels= 1,
                 n_blocks=n_blocks,
                 start_filters=start_filters,
                 activation= activation,
                 normalization= normalization,
                 conv_mode= conv_mode,
                 dim=dim,
                 up_mode=up_mode
                 )
    return load_model(opt,model,checkpoint)

def load_model_main(opt):
    if opt.model_name in ['dense']:
        return load_dense(opt)
    elif opt.model_name in ['resunet']:
        return load_resunet(opt)
    elif opt.model_name in ['rrdbnet']:
        return load_rrdbnet(opt)
    elif opt.model_name in ['unet_small']:
        return load_small_unet(opt)
    elif opt.model_name in ['unet']:
        return load_unet(opt)
    elif opt.model_name in ['patch_gan']:
        return load_patch_gan(opt)
    else:
        print(f"model {opt.model_name} load function not found")


# ****************************** CODE FOR 3D MODELS  ********************************************************************

def load_model_3d_main(opt):
    if opt.model_name in ['dense','srdense']:
        return load_dense(opt,load_3d=True)
    elif opt.model_name in ['rrdbnet']:
        return load_rrdbnet(opt,load_3d=True)
    else:
        print(f"model {opt.model_name} load function not found")