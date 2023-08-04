# process_yaml.py file
#imports
import yaml
import argparse
import sys
from utils.train_utils import load_dataset,get_criterion,get_optimizer
from models.transformer_discriminator import Discriminator
from models.esrt import ESRT
import torch
import torch.nn as nn
from utils.image_quality_assessment import PSNR,SSIM
import copy
from utils.logging_metric import LogMetric,create_loss_meters_srdense
from utils.general import save_configuration_yaml,LogOutputs
from utils.config import set_outputs_dir,set_training_metric_dir,set_plots_dir
import torch.optim as optim
import os
import wandb
import time
import matplotlib.pyplot as plt

from train_utils.realistic_gan import RealisticGANTrainer
from train_utils.standard_gan import StandardGANTrainer
from train_utils.lsgan import LSGANTrainer
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"]='1'

def clean_opt(opt):
    opt.g_optimizer = None      
    opt.d_optimizer = None           
    opt.psnr = None #remove model from yaml file before saving configuration
    opt.ssim = None
    opt.device = None
    opt.wandb_obj = None
    opt.train_dataloader= None
    opt.train_datasets= None
    opt.train_dataset = None
    opt.eval_dataloader= None
    opt.eval_datasets = None
    opt.generator =  None
    opt.discriminator =  None
    return opt

def load_state_dict_func(path):

    # state_dict = torch.load(path)
    state_dict=path
    new_state_dcit = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dcit[name] = v
    return new_state_dcit


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path,map_location=torch.device(device))
    model = ESRT(
            upscale=checkpoint['upscale_factor'],
            n_feats=checkpoint['n_feats'],
            n_blocks=checkpoint['n_blocks'], 
            kernel_size=checkpoint['kernel_size']
            ).to(device=device)
    model_dict = load_state_dict_func(checkpoint['model_state_dict'])
    model.load_state_dict(model_dict,strict=False)
    return model




if __name__ == "__main__":
    '''get the configuration file'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, 
    default='train_config_yaml/ls_gan.yaml')
    sys.argv = ['-f']
    opt   = parser.parse_known_args()[0]

    '''load the configuration file and append to current parser arguments'''
    ydict = yaml.load(open(opt.config), Loader=yaml.FullLoader)
    for k,v in ydict.items():
        if k=='config':
            pass
        else:
            parser.add_argument('--'+str(k), required=False, default=v)
    opt  = parser.parse_args()

    '''adding seed for reproducibility'''
    torch.manual_seed(opt.seed)

    '''set device'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device
   

    '''load dataset (loading dataset based on dataset name and factor on arguments)'''
    if opt.eval:
        opt.train_dataloader,opt.train_dataset,opt.eval_dataloader,opt.eval_datasets = load_dataset(opt=opt,load_eval=opt.eval)
    else:
        opt.train_dataloader,opt.train_dataset= load_dataset(opt=opt,load_eval=opt.eval)


    '''load discriminator'''
    opt.discriminator = Discriminator(img_size=opt.patch_size,
                                 patch_size=opt.dist_patch_size, 
                                 in_chans=opt.in_chans,
                                 num_classes=opt.num_classes,
                                 embed_dim= opt.embed_dim,
                                 num_heads= opt.num_heads,
                                 mlp_ratio= opt.mlp_ratio,
                                 qkv_bias= opt.qkv_bias,
                                 qk_scale =opt.qk_scale,
                                 drop_rate= opt.drop_rate,
                                 attn_drop_rate=opt.attn_drop_rate,
                                 drop_path_rate=opt.drop_path_rate,
                                 norm_layer=opt.norm_layer,
                                 depth=opt.depth,
                                 act_layer= opt.act_layer,
                                 diff_aug=opt.diff_aug, 
                                 apply_sigmoid=opt.apply_sigmoid
                                 ).to(device=device)

    if opt.resume:
         # load checkpoint
        opt.generator = load_model(checkpoint_path=opt.checkpoint_path,device=device)
        print("MOdel checkpoints loaded from {}".format(opt.checkpoint_path))

    else:
        opt.start_epoch = 0
        '''load generator'''
        opt.generator = ESRT(upscale=opt.factor,n_feats=opt.n_feats,n_blocks=opt.n_blocks, kernel_size=opt.kernel_size).to(device=device)


    ''' print model '''
    # print(opt.generator)


    '''setup the outputs and logging metric dirs on '''
    set_outputs_dir(opt) 
    set_training_metric_dir(opt) 
    set_plots_dir(opt)

    
    '''wrap model for data parallelism'''
    num_of_gpus = torch.cuda.device_count()
    print("Number of GPU available", num_of_gpus)
    if num_of_gpus>1:
        opt.generator = nn.DataParallel(opt.generator,device_ids=[*range(num_of_gpus)])
        opt.discriminator = nn.DataParallel(opt.discriminator,device_ids=[*range(num_of_gpus)])
        opt.data_parallel = True
        print("Multiple GPU Training")
    else:
        opt.data_parallel=False

    
    '''set up optimizer for generator and discriminator'''
    opt.g_optimizer = get_optimizer(optimizer_type= opt.g_optimizer_type, model= opt.generator, lr=opt.g_lr)
    opt.d_optimizer = get_optimizer(optimizer_type=opt.d_optimizer_type,model=opt.discriminator, lr=opt.d_lr)
    print("Optimizer is Loaded")

    print('training for factor ',opt.factor)


    #setting metric for evaluation
    psnr = PSNR()
    ssim = SSIM()
    opt.psnr = psnr.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)
    opt.ssim = ssim.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)
    print("PSNR and SSIM is loaded")

    if opt.wandb:
        wandb.init(
        project=opt.project_name,
                name = opt.exp_name,
                config = opt )

        wandb.watch(opt.generator,log="all",log_freq=1)

    else:
        wandb=None

    opt.wandb_obj =  wandb

    if opt.gan_loss_type == 'standard':
        print("Running Standard GAN Training")
        trainer = StandardGANTrainer(args=opt,use_pixel_loss=opt.use_pixel_loss)
    elif opt.gan_loss_type == 'lsgan':
        print("Running LS GAN Training")
        trainer = LSGANTrainer(args=opt,use_pixel_loss=opt.use_pixel_loss)
    elif opt.gan_loss_type == 'realistic':
        print("Running Realistic GAN Training")
        trainer = RealisticGANTrainer(args=opt,use_pixel_loss=opt.use_pixel_loss)
    else:
        print( "Gan trainer type {} not implemented".format(opt.gan_loss_type))


    try:
        trainer.train()
 
    except KeyboardInterrupt:
        clean_opt(opt=opt)
        _ = save_configuration_yaml(opt)


    if opt.wandb:
        wandb.unwatch(opt.generator)
        wandb.finish()


    clean_opt(opt=opt)
    # from pprint import pprint
    # print("Arguments Properties")
    # pprint(vars(opt))
    _ = save_configuration_yaml(opt)