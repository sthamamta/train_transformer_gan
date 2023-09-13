# process_yaml.py file
#imports
import yaml
import argparse
import sys
from utils.train_utils import load_dataset,get_optimizer
from models.transformer_discriminator import RankDiscriminator
import torch
import torch.nn as nn


from utils.general import save_configuration_yaml
from utils.config import set_outputs_dir,set_training_metric_dir,set_plots_dir
import torch.optim as optim
import os
import wandb
import time
import matplotlib.pyplot as plt
from dataset.dataset_rank_classification import MRIDataset

from train_utils.rank_trainer import RankTrainer


os.environ["CUDA_VISIBLE_DEVICES"]='0,1'

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


if __name__ == "__main__":
    '''get the configuration file'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, 
    default='train_config_yaml/rank_classifier.yaml')
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
    opt.train_dataset = MRIDataset( patch_size=opt.patch_size,
                            augment = opt.augment,
                            normalize = opt.normalize)
                            
    opt.train_dataloader = torch.utils.data.DataLoader(opt.train_dataset, batch_size = opt.train_batch_size,shuffle=True,
        num_workers=1,pin_memory=False,drop_last=False)


    '''load discriminator'''
    opt.model = RankDiscriminator(img_size=opt.patch_size,
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

    
    ''' print model '''
    # print(opt.generator)


    '''setup the outputs and logging metric dirs on '''
    opt.save_checkpoints_dir = 'outputs/{}/{}/checkpoints/patch/patch-{}/'.format(opt.model_name,opt.exp_name, opt.patch_size)
    opt.loss_dir = 'outputs/{}/{}/losses/patch/patch-{}/factor/'.format(opt.model_name, opt.exp_name, opt.patch_size)
   
    
    '''wrap model for data parallelism'''
    num_of_gpus = torch.cuda.device_count()
    print("Number of GPU available", num_of_gpus)
    if num_of_gpus>1:
        opt.model = nn.DataParallel(opt.model,device_ids=[*range(num_of_gpus)])
        opt.data_parallel = True
        print("Multiple GPU Training")
    else:
        opt.data_parallel=False

    
    '''set up optimizer for generator and discriminator'''
    opt.optimizer = get_optimizer(optimizer_type=opt. optimizer_type,model=opt.model, lr=opt.lr)
    print("Optimizer is Loaded")



    if opt.wandb:
        wandb.init(
        project=opt.project_name,
                name = opt.exp_name,
                config = opt )

        wandb.watch(opt.model,log="all",log_freq=1)

    else:
        wandb=None
    opt.wandb_obj =  wandb


   
    trainer = RankTrainer(args=opt)
    try:
        trainer.train()
    except KeyboardInterrupt:
        clean_opt(opt=opt)
        _ = save_configuration_yaml(opt)


    if opt.wandb:
        wandb.unwatch(opt.model)
        wandb.finish()


    clean_opt(opt=opt)
    _ = save_configuration_yaml(opt)