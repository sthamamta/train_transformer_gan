from train_utils.trainer import Trainer
from tqdm import tqdm
from train_utils.general import log_results, LogOutputs, adjust_learning_rate, create_loss_meters_gan, LogMetric
import os
import torch
import torch.nn as nn
import copy
import time

class VanillaGanTrainer(Trainer):
    def __init__(self, args, gen_loss_wt=[1.0,0.5],gen_loss_type=['l1','mse']):
        super().__init__(args)

        self.gen_loss_wt = gen_loss_wt
        self.gen_losses = []
        self.gen_loss_type = gen_loss_type
        self.set_gen_loss()
        self.criterion = torch.nn.BCELoss()
        # assert gen loss and wt list have equal length
        assert len(self.gen_losses) == len(self.gen_loss_wt), ('list of generator pixel losses and list of those loss weights should have the same length, but got '
                                                f'{len(self.gen_losses)} and {len(self.gen_loss_wt)}.')
        self.metric_dict = LogMetric()
        

    def train_epoch(self,epoch,epoch_losses):
        start_time = time.time()
        with tqdm(total=(len(self.train_dataset) - len(self.train_dataset) % self.train_batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, self.num_epochs - 1))

            for idx, (data) in enumerate(self.train_dataloader):
                images = data['image']
                labels = data['label']
                fake_images = self.G(images)

                batch_size = images.shape[0]

                # updating the discriminator
                real_labels = torch.FloatTensor(batch_size).fill_(1)
                fake_labels = torch.FloatTensor(batch_size).fill_(0)

                self.set_requires_grad(self.D,True)
                self.set_requires_grad(self.G,False)
                self.d_optimizer.zero_grad()

                fake_preds = self.D(fake_images)
                real_preds = self.D (labels)
                loss_D_fake = self.criterion(fake_preds,fake_labels)
                loss_D_real = self.criterion(real_preds,real_labels)
                loss_D = (loss_D_fake + loss_D_real)/2
                loss_D.backward(retain_graph=True)

                # updating the generator
                self.set_requires_grad(self.D, False)
                self.set_requires_grad(self.G,True)
                fake_preds = self.D(fake_images)
                loss_G_Gan = self.criterion(fake_preds,real_labels)
                for (gen_loss,gen_loss_wt) in zip(self.gen_losses,self.gen_loss_wt)
                    loss_G_pix += gen_loss(fake_images,labels) * gen_loss_wt
                loss_G = loss_G_Gan + loss_G_pix
                loss_G.backward(retain_graph=True)


            log_results(epoch_losses) # function to print out the losses


            t.set_postfix(loss='{:.6f}'.format(epoch_losses['loss_G_Gan'].avg))
            t.update(len(images))

            if epoch % self.n_freq==0:
                if not os.path.exists(self.checkpoints_dir):
                    os.makedirs(self.checkpoints_dir)
                path = os.path.join(self.checkpoints_dir, 'epoch_{}_f_{}.pth'.format(epoch,self.args.factor))
                self.G.save(model=self.G,model_weights=self.G.net_G.state_dict(),opt=self.args,path=path,epoch=epoch)
                with torch.no_grad():
                    preds = self.G(images.to(self.device)).to(self.device)

        end_time = time.time()
        time_taken = (end_time - start_time) / 60.0
        print("Time taken for a epoch: %.2f minutes" % time_taken)
        return {
            'epoch':epoch,
            'hr': labels,
            'lr':images,
            'preds':preds,
        }
    

    def val_epoch(self):
        pass

    def train(self):

        if self.wandb:
            log_table_output = LogOutputs()  #create a dictionary of list of input, output and label
        best_weights = copy.deepcopy(self.G.state_dict())
        best_epoch = 0
        best_psnr = 0.0
        start_time = time.time()
        for epoch in range(self.num_epochs):
            self.lr_G = adjust_learning_rate(self.g_optimizer, epoch,self.lr_G)
            self.lr_D= adjust_learning_rate(self.d_optimizer, epoch,self.lr_D)

            '''setting model in train mode'''
            self.G.train()

            '''train one epoch and evaluate the model'''
            epoch_losses = create_loss_meters_gan()  # create average meter for all loss

            output = self.train_epoch(epoch,epoch_losses)
            eval_loss, eval_l1,eval_psnr, eval_ssim,eval_hfen = self.val_epoch()

            if self.wandb:
                self.wandb_obj.log({"val/val_loss" : eval_loss,
                "val/val_l1_error":eval_l1,
                "val/val_psnr": eval_psnr,
                "val/val_ssim":eval_ssim,
                "val/val_hfen":eval_hfen,
                "epoch": epoch
                })
                for key in epoch_losses.keys():
                    self.wandb_obj.log({"train/{}".format(key) : epoch_losses[key].avg,
                    })
                self.wandb_obj.log({
                    "other/learning_G": self.lr_G,
                    "other/learning_D": self.lr_D,
                })
                # log_output_images(images, preds, labels) #overwrite on same table on every epoch
                if epoch % self.n_freq == 0:
                    log_table_output.append_list(epoch=epoch,images = output['lr'],labels = output['hr'],predictions = output['preds'])  #create a class with list and function to loop through list and add to log table
            # apply_model(model.net_G,epoch,opt,addition=opt.addition)
            print('eval psnr: {:.4f}'.format(eval_psnr))

            if eval_psnr > best_psnr:
                best_epoch = epoch
                best_psnr = eval_psnr
                best_weights = copy.deepcopy(self.G.state_dict())

                '''adding to the dictionary'''
                self.metric_dict.update_dict([eval_loss,eval_l1,eval_psnr,eval_ssim,eval_hfen],training=False)

                
                self.metric_dict.update_dict([epoch_losses['loss_D_fake'].avg,
                epoch_losses['loss_D_real'].avg,
                epoch_losses['loss_D'].avg,
                epoch_losses['loss_G_GAN'].avg,
                epoch_losses['loss_G_L1'].avg,
                epoch_losses['loss_G'].avg])

            if self.wandb:
                print("logging output table")
                log_table_output.log_images(columns = ["epoch","image", "pred", "label"],wandb=self.wandb_obj) 

            # opt.generator = None  #remove model from yaml file before saving configuration
            # path = metric_dict.save_dict(opt)
            # _ = save_configuration_yaml(opt)
            # # print(metric_dict.log_dict)

            path="best_weights_factor_{}_epoch_{}.pth".format(opt.factor,best_epoch)
            path = os.path.join(self.checkpoints_dir, path)
            self.G.save(self.G,best_weights,self.args,path,best_epoch)
            print('model saved')

        end_time = time.time()
        time_taken = (end_time - start_time) / 60.0

        print("Time taken for training: %.2f minutes" % time_taken)

            