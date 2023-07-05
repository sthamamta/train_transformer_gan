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
from loss.tv_regularizer import TVRegularizer

class RealisticGANTrainer(Trainer):
    def __init__(self, args,use_pixel_loss=True):
        super().__init__(args)

        self.data_parallel = args.data_parallel
        self.plot_train_example = args.plot_train_example
        self.plot_dir = args.plot_dir
        self.adversarial_weight = args.adversarial_weight
        self.use_pixel_loss = use_pixel_loss

        self.d_iter = args.d_iter # generator is trained for every 5 iter when discriminator for 1 iter if d_iter=5
        if self.use_pixel_loss:
            self.gen_loss_wt = args.gen_loss_wt
            self.gen_loss_type = args.gen_loss_type
            self.gen_losses = {}
            self.generator_loss_key_list = ['loss_G_pix']
            self.set_gen_loss()
            print("Using pixel based loss for generator")
        # assert gen loss and wt list have equal length
            assert len(self.gen_losses) == len(self.gen_loss_wt), ('list of generator pixel losses and list of those loss weights should have the same length, but got '
                                                f'{len(self.gen_losses)} and {len(self.gen_loss_wt)}.')

        # self.criterion = torch.nn.BCELoss()  # outputs should be raw probability score (softmax should be applied beforehand)
        # self.criterion = torch.nn.BCEWithLogitsLoss() # takes raw logits and applies softmax before calculating loss
        # self.criterion = torch.nn.BCEWithLogitsLoss(reduce=False)#computes the binary cross-entropy loss with sigmoid function, but does not reduce the loss over all examples in the batch
        if self.use_pixel_loss:
            self.loss_keys_list = self.generator_loss_key_list + ['loss_D_fake','loss_D_real','loss_D', 'loss_G_real', 'loss_G_fake','loss_G_Gan','loss_G']
        else:
            self.loss_keys_list = ['loss_D_fake','loss_D_real','loss_D', 'loss_G_real', 'loss_G_fake','loss_G_Gan','loss_G']
        self.metric_dict = LogMetric({key: [] for key in self.loss_keys_list})
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.tv_regularizer = TVRegularizer()
        

    def train_epoch(self,epoch):

        epoch_losses = create_loss_meters_from_keys(self.loss_keys_list)  # create average meter for all loss
        start_time = time.time()
        with tqdm(total=(len(self.train_dataset) - len(self.train_dataset) % self.train_batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, self.num_epochs - 1))

            for idx, (data) in enumerate(self.train_dataloader):
                images = data['lr'].to(self.device)
                labels = data['hr'].to(self.device)

                # print("shape of images", images.shape)
                # print("REACHED HERE")

                batch_size = images.shape[0]

                # creating real and fake labels
                real_labels = torch.FloatTensor(batch_size,1).fill_(1).to(self.device)
                fake_labels = torch.FloatTensor(batch_size,1).fill_(0).to(self.device)

                #training generator
                self.set_requires_grad(self.G,True)
                # self.set_requires_grad(self.D,False)

                self.g_optimizer.zero_grad()

                # print("the shape of input image is", images.shape)
                fake_images = self.G(images)

                fake_output = self.D(fake_images)
                # fake_output = self.D(fake_images.detach().clone())
                
                real_output = self.D(labels)

                d_loss_gt = self.adversarial_criterion(real_output - torch.mean(fake_output), fake_labels) * 0.5
                d_loss_sr = self.adversarial_criterion(fake_output - torch.mean(real_output), real_labels) * 0.5
                loss_G_Gan = self.adversarial_weight * (d_loss_gt + d_loss_sr)

                # if self.use_pixel_loss:
                #     loss_G_pix = 0
                #     for (gen_loss,gen_loss_wt) in zip(self.gen_losses,self.gen_loss_wt):
                #         loss_G_pix += gen_loss(fake_images,labels) * gen_loss_wt
                #     loss_G = loss_G_Gan + loss_G_pix
                # else:
                #     loss_G = loss_G_Gan

                if self.use_pixel_loss:
                    loss_G_pix = 0
                    for idx ,(key, loss_obj) in enumerate(self.gen_losses.items()):
                        pix_loss_wt = self.gen_loss_wt[idx]
                        if key == 'tv_regularizer':
                            pix_loss_value = loss_obj(fake_images)
                        else:
                            pix_loss_value = loss_obj(fake_images,labels)
                        loss_G_pix += pix_loss_value * pix_loss_wt
                        
                        epoch_losses[key].update(pix_loss_value.item(),batch_size)
                    loss_G = loss_G_pix + loss_G_Gan
                else:
                    loss_G = loss_G_Gan

                loss_G.backward()
                self.g_optimizer.step()


                # training discriminator
                self.set_requires_grad(self.D,True)
                self.d_optimizer.zero_grad()
                gt_output = self.D(labels)
                loss_D_real = self.adversarial_criterion(gt_output - torch.mean(sr_output), real_labels) * 0.5
                (loss_D_real).backward(retain_graph=True)

                sr_output = self.D(fake_images.detach().clone())
                loss_D_fake = self.adversarial_criterion(sr_output - torch.mean(gt_output), fake_labels) * 0.5
                # back-propagate the gradient information of the fake samples
                (loss_D_fake).backward()
                self.d_optimizer.step()


                loss_D =  loss_D_real + loss_D_fake

                # update average meter on every iteration
                epoch_losses['loss_D'].update(loss_D.item(), batch_size)
                epoch_losses['loss_D_fake'].update(loss_D_fake.item(), batch_size)
                epoch_losses['loss_D_real'].update(loss_D_real.item(), batch_size)

                epoch_losses['loss_G_fake'].update(d_loss_sr.item(), batch_size)
                epoch_losses['loss_G_real'].update(d_loss_gt.item(), batch_size)
                epoch_losses['loss_G_Gan'].update(loss_G_Gan.item(),batch_size)
                epoch_losses['loss_G'].update(loss_G.item(),batch_size)

                if self.use_pixel_loss:
                    epoch_losses['loss_G_pix'].update(loss_G_pix.item(),batch_size)

                # print("complete on iteration")
                # print ("***********************************")

            log_results(epoch_losses) # function to print out the average per epoch


            t.set_postfix(loss='{:.6f}'.format(epoch_losses['loss_G_Gan'].avg))
            t.update(len(images))

            if epoch % self.n_freq==0:
                if not os.path.exists(self.save_checkpoints_dir):
                    os.makedirs(self.save_checkpoints_dir)
                path = os.path.join(self.save_checkpoints_dir, 'epoch_{}_f_{}.pth'.format(epoch,self.args.factor))

                if self.data_parallel:
                    self.G.module.save(model_weights=self.G.state_dict(),path=path,epoch=epoch)
                else:
                   self.G.save(model_weights=self.G.state_dict(),path=path,epoch=epoch) 
                with torch.no_grad():
                    preds = self.G(images.to(self.device)).to(self.device)

                if self.plot_train_example:
                    self.save_train_example(images=images,labels=labels,fake_images=fake_images,epoch=epoch)


        end_time = time.time()
        time_taken = (end_time - start_time) / 60.0
        print("Time taken for a epoch: %.2f minutes" % time_taken)
        return {
            'epoch':epoch,
            'hr': labels,
            'lr':images,
            'preds':fake_images,
        }, epoch_losses


    def train(self):

        if self.wandb:
            log_table_output = LogOutputs()  #create a dictionary of list of input, output and label
        best_weights = copy.deepcopy(self.G.state_dict())
        best_epoch = 0
        best_psnr = 0.0
        start_time = time.time()

        self.D.train()
        self.G.train()
        print("Starting epoch from ", self.start_epoch)
        for epoch in range(self.start_epoch,self.num_epochs+1):
            '''train one epoch and evaluate the model'''

            # print(torch.cuda.memory_summary())

            # output, epoch_losses = self.train_epoch(epoch)
            output, epoch_losses = self.train_epoch_generator_more_epoch(epoch)
            self.g_scheduler.step()
            self.d_scheduler.step()


            if self.eval and epoch % self.eval_freq == 0 :
                eval_loss, eval_l1,eval_psnr, eval_ssim,eval_hfen = self.val_epoch()

            if self.wandb:
                if self.args.eval:
                    self.wandb_obj.log({"val/val_loss" : eval_loss,
                    "val/val_l1_error":eval_l1,
                    "val/val_psnr": eval_psnr,
                    "val/val_ssim":eval_ssim,
                    "val/val_hfen":eval_hfen
                    })
                    print('eval psnr: {:.4f}'.format(eval_psnr))

                for key in epoch_losses.keys():
                    self.wandb_obj.log({"train/{}".format(key) : epoch_losses[key].avg,  #logging each epoch loss to wandb
                    })
                self.wandb_obj.log({
                    "epoch": epoch,
                    "other/learning_G": self.g_scheduler.get_last_lr()[0],
                    "other/learning_D": self.d_scheduler.get_last_lr()[0],
                })
                # log_output_images(images, preds, labels) #overwrite on same table on every epoch
                if epoch % self.n_freq == 0:
                    log_table_output.append_list(epoch=epoch,images = output['lr'],labels = output['hr'],predictions = output['preds'])  #create a class with list and function to loop through list and add to log table

                
            if self.args.eval:
                if eval_psnr > best_psnr:
                    best_epoch = epoch
                    best_psnr = eval_psnr
                    best_weights = copy.deepcopy(self.G.state_dict())

                '''adding to the dictionary'''
                self.metric_dict.update_dict([eval_loss,eval_l1,eval_psnr,eval_ssim,eval_hfen],training=False)

            # update the dictionary of list of losses for every epoch
            self.metric_dict.update_dict_with_dictionary(epoch_losses)
 
            if self.wandb:
                print("logging output table")
                log_table_output.log_images(columns = ["epoch","image", "pred", "label"],wandb=self.wandb_obj) 

            if self.args.eval:
                path="best_weights_factor_{}_epoch_{}.pth".format(self.factor,best_epoch)
                path = os.path.join(self.save_checkpoints_dir, path)
                if self.data_parallel:
                    self.G.module.save(model_weights=best_weights,path=path,epoch=best_epoch)
                else:
                    self.G.save(model_weights=best_weights,path=path,epoch=best_epoch)
                print('model saved')

        end_time = time.time()
        time_taken = (end_time - start_time) / 60.0

        print("Time taken for training: %.2f minutes" % time_taken)


    def train_epoch_generator_more_epoch(self,epoch):

        epoch_losses = create_loss_meters_from_keys(self.loss_keys_list)  # create average meter for all loss
        start_time = time.time()
        with tqdm(total=(len(self.train_dataset) - len(self.train_dataset) % self.train_batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, self.num_epochs - 1))

            for idx, (data) in enumerate(self.train_dataloader):
                images = data['lr'].to(self.device)
                labels = data['hr'].to(self.device)

                batch_size = images.shape[0]

                # creating real and fake labels
                real_labels = torch.FloatTensor(batch_size,1).fill_(1).to(self.device)
                fake_labels = torch.FloatTensor(batch_size,1).fill_(0).to(self.device)

                #training generator
                self.set_requires_grad(self.G,True)
                # self.set_requires_grad(self.D,False)

                self.g_optimizer.zero_grad()

                fake_images = self.G(images)
                fake_output = self.D(fake_images)
                real_output = self.D(labels)

                d_loss_gt = self.adversarial_criterion(real_output - torch.mean(fake_output), fake_labels) * 0.5
                d_loss_sr = self.adversarial_criterion(fake_output - torch.mean(real_output), real_labels) * 0.5
                loss_G_Gan = self.adversarial_weight * (d_loss_gt + d_loss_sr)

                if self.use_pixel_loss:
                    loss_G_pix = 0
                    for idx ,(key, loss_obj) in enumerate(self.gen_losses.items()):
                        pix_loss_wt = self.gen_loss_wt[idx]
                        if key == 'tv_regularizer':
                            pix_loss_value = loss_obj(fake_images)
                        else:
                            pix_loss_value = loss_obj(fake_images,labels)
                        loss_G_pix += pix_loss_value * pix_loss_wt
                        
                        epoch_losses[key].update(pix_loss_value.item(),batch_size)
                    loss_G = loss_G_pix + loss_G_Gan
                else:
                    loss_G = loss_G_Gan

                 # Using TV regularizer
                tv_loss = self.tv_regularizer(fake_images)
                loss_G = loss_G + tv_loss
                    
                loss_G.backward()
                self.g_optimizer.step()


                # training discriminator
                if (idx + 1) % self.d_iter == 0:
                    self.set_requires_grad(self.D,True)
                    self.d_optimizer.zero_grad()

                    #using label smoothing for discriminator
                    real_labels = torch.FloatTensor(batch_size, 1).uniform_(0.8, 1.1).to(self.device)
                    fake_labels = torch.FloatTensor(batch_size, 1).uniform_(0, 0.3).to(self.device)

                    gt_output = self.D(labels)
                    sr_output = self.D(fake_images.detach().clone())
                    loss_D_real = self.adversarial_criterion(gt_output - torch.mean(sr_output), real_labels) * 0.5
                    (loss_D_real).backward(retain_graph=True)

                    loss_D_fake = self.adversarial_criterion(sr_output - torch.mean(gt_output), fake_labels) * 0.5
                    # back-propagate the gradient information of the fake samples
                    (loss_D_fake).backward()
                    self.d_optimizer.step()


                    loss_D =  loss_D_real + loss_D_fake

                    # update average meter on every iteration
                    epoch_losses['loss_D'].update(loss_D.item(), batch_size)
                    epoch_losses['loss_D_real'].update(loss_D_real.item(), batch_size)
                    epoch_losses['loss_D_fake'].update(loss_D_fake.item(), batch_size)

            epoch_losses['loss_G_fake'].update(d_loss_sr.item(), batch_size)
            epoch_losses['loss_G_real'].update(d_loss_gt.item(), batch_size)
            epoch_losses['loss_G_Gan'].update(loss_G_Gan.item(),batch_size)
            if self.use_pixel_loss:
                epoch_losses['loss_G_pix'].update(loss_G_pix.item(),batch_size)
            epoch_losses['loss_G'].update(loss_G.item(),batch_size)


            log_results(epoch_losses) # function to print out the average per epoch


            t.set_postfix(loss='{:.6f}'.format(epoch_losses['loss_G_Gan'].avg))
            t.update(len(images))

            if epoch % self.n_freq==0:
                if not os.path.exists(self.save_checkpoints_dir):
                    os.makedirs(self.save_checkpoints_dir)
                path = os.path.join(self.save_checkpoints_dir, 'epoch_{}_f_{}.pth'.format(epoch,self.args.factor))
                if self.data_parallel:
                    self.G.module.save(model_weights=self.G.state_dict(),path=path,epoch=epoch)
                else:
                    self.G.save(model_weights=self.G.state_dict(),path=path,epoch=epoch)
                with torch.no_grad():
                    preds = self.G(images.to(self.device)).to(self.device)

                if self.plot_train_example:
                    self.save_train_example(images=images,labels=labels,fake_images=fake_images,epoch=epoch)



        end_time = time.time()
        time_taken = (end_time - start_time) / 60.0
        print("Time taken for a epoch: %.2f minutes" % time_taken)
        return {
            'epoch':epoch,
            'hr': labels,
            'lr':images,
            'preds':fake_images,
        }, epoch_losses