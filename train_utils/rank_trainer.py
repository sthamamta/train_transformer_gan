# code taken from: https://github.com/Lornatang/ESRGAN-PyTorch/blob/main/train_esrgan.py#L270
from train_utils.trainer import Trainer
from tqdm import tqdm
from train_utils.general import log_results, LogOutputs, create_loss_meters_from_keys, LogMetric
import os
import torch
import torch.nn as nn
import copy
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

class RankTrainer(object):
    def __init__(self, args):
        self.args = args
    
        self.data_parallel = args.data_parallel

        self.wandb = args.wandb
        self.start_epoch = 0
      
        self.wandb_obj = args.wandb_obj

        self.device = args.device

        self.train_dataset = args.train_dataset
        self.train_dataloader =  args.train_dataloader
        self.train_batch_size = args.train_batch_size

        self.optimizer = args.optimizer
    

        self.lr_decay_factor = args.lr_decay_factor
        self.lr_schedular_step = args.lr_schedular_step
        self.scheduler = StepLR(self.optimizer, step_size=self.lr_schedular_step, gamma=self.lr_decay_factor)

        self.num_epochs = args.num_epochs
        self.n_freq = args.n_freq
    
        '''save directory'''
        self.save_checkpoints_dir = args.save_checkpoints_dir
        self.loss_dir = args.loss_dir

    
        self.loss_keys_list = ['loss']
        self.metric_dict = LogMetric({key: [] for key in self.loss_keys_list})
        self.cross_entrophy_loss = nn.CrossEntropyLoss()

        self.model = args.model

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def train(self):

        if self.wandb:
            log_table_output = LogOutputs()  #create a dictionary of list of input, output and label
        
        start_time = time.time()

        self.model.train()

        print("Starting epoch from ", self.start_epoch)
        for epoch in range(self.start_epoch,self.num_epochs+1):
            '''train one epoch and evaluate the model'''

            output, epoch_losses = self.train_epoch(epoch)

            self.scheduler.step()

            if self.wandb:

                for key in epoch_losses.keys():
                    self.wandb_obj.log({"train/{}".format(key) : epoch_losses[key].avg,  #logging each epoch loss to wandb
                    })
                self.wandb_obj.log({
                    "epoch": epoch,
                    "other/rate": self.scheduler.get_last_lr()[0]
                })
                if epoch % self.n_freq == 0:
                    log_table_output.append_list_classification(epoch=epoch,images = output['inputs'],labels = output['labels'],predictions = output['preds'])  #create a class with list and function to loop through list and add to log table

                
            # update the dictionary of list of losses for every epoch
            self.metric_dict.update_dict_with_dictionary(epoch_losses)
 
            if self.wandb:
                print("logging output table")
                log_table_output.log_images(columns = ["epoch","image", "pred", "label"],wandb=self.wandb_obj, images=False) 


        end_time = time.time()
        time_taken = (end_time - start_time) / 60.0

        print("Time taken for training: %.2f minutes" % time_taken)


    def train_epoch(self,epoch):

        epoch_losses = create_loss_meters_from_keys(self.loss_keys_list)  # create average meter for all loss
        start_time = time.time()

        with tqdm(total=(len(self.train_dataset) - len(self.train_dataset) % self.train_batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, self.num_epochs - 1))

            for idx, (data) in enumerate(self.train_dataloader):
                images = data['lr'].to(self.device)
                labels = data['label'].to(self.device)

                # print("shape of images  during training", images.shape)
                # print("shape of labels  during training", labels.shape)
                
                batch_size = images.shape[0]

                #training generator
                self.set_requires_grad(self.model,True)

                self.optimizer.zero_grad()
                prediction= self.model(images)
                loss_model = self.cross_entrophy_loss(prediction, labels)

                
                epoch_losses['loss'].update(loss_model.item(),batch_size)
                  
                loss_model.backward()
                self.optimizer.step()

            log_results(epoch_losses) # function to print out the average per epoch


            t.set_postfix(loss='{:.6f}'.format(epoch_losses['loss'].avg))
            t.update(len(images))

            if epoch % self.n_freq==0:
                if not os.path.exists(self.save_checkpoints_dir):
                    os.makedirs(self.save_checkpoints_dir)
                path = os.path.join(self.save_checkpoints_dir, 'epoch_{}_f_{}.pth'.format(epoch,2))
                if self.data_parallel:
                    self.model.module.save(model_weights=self.model.state_dict(),path=path,epoch=epoch)
                else:
                    self.model.save(model_weights=self.model.state_dict(),path=path,epoch=epoch)
                    
            with torch.no_grad():
                    preds = self.model(images.to(self.device)).to(self.device)
                    preds = torch.argmax(preds, dim=1)    

                    
        end_time = time.time()
        time_taken = (end_time - start_time) / 60.0
        print("Time taken for a epoch: %.2f minutes" % time_taken)
        return {
            'epoch':epoch,
            'labels': labels,
            'inputs':images,
            'preds':preds,
        }, epoch_losses