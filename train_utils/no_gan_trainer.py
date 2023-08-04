from train_utils.general import *
import wandb
import copy
import matplotlib.pyplot as plt
from loss.gram_loss import GramLoss
from loss.ssim_loss import SSIM
from loss.laplacian_pyramid_loss import LaplacianPyramidLoss
from loss.tv_regularizer import TVRegularizer
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from train_utils.general import AverageMeter
from loss.rank_loss import RankLoss

def forward_chop(model, x, shave=10, min_size=60000):
    scale = 2   #self.scale[self.idx_scale]
    n_GPUs = 1    #min(self.n_GPUs, 4)
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            sr_batch = model(lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(model, patch, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


class NoGanTrainer(object):

    def __init__(self, args):
        print("Trainer super init model.")
        self.args = args
        self.data_parallel = args.data_parallel
        self.G = args.generator
        self.wandb_obj = args.wandb_obj
        self.plot_train_example = args.plot_train_example
        self.gen_loss_wt = args.gen_loss_wt
        self.gen_loss_type = args.gen_loss_type
        self.plot_train_example = args.plot_train_example
        self.gen_losses = {}  #list of each loss class objects
        self.loss_keys_list = ['loss_G_pix']
        self.gen_loss_wt = args.gen_loss_wt
        self.set_gen_loss()
        self.plot_image_example = args.output_image_dir

        # print(self.gen_losses)
        # print(self.gen_loss_wt)


        # assert gen loss and wt list have equal length
        assert len(self.gen_losses) == len(self.gen_loss_wt), ('list of generator pixel losses and list of those loss weights should have the same length, but got '
                                                f'{len(self.gen_losses)} and {len(self.gen_loss_wt)}.')

        self.device = args.device
        self.wandb = args.wandb  # true of false
        self.eval = args.eval #True or false
        
        self.metric_dict = LogMetric({key: [] for key in self.loss_keys_list})


        self.epoch_losses = {key: AverageMeter() for key in self.loss_keys_list}


        # print(self.loss_keys_list)
        # print(self.gen_loss_wt)
        # print(self.epoch_losses)
        # print(self.metric_dict)
        # quit();


        # WGAN values from paper
        self.g_lr = args.g_lr

        self.factor = args.factor


        self.train_dataset = args.train_dataset
        self.train_dataloader =  args.train_dataloader
        self.train_batch_size = args.train_batch_size

        if self.eval:
            self.eval_dataloader = args.eval_dataloader
            self.eval_datasets = args.eval_datasets 
            self.eval_batch_size = args.eval_batch_size

        if self.eval:
            self.eval_freq = args.eval_freq
        else:
            self.eval_freq = None

        self.g_optimizer = args.g_optimizer
        self.g_lr_decay_factor = args.g_lr_decay_factor
        self.g_lr_schedular_step = args.g_lr_schedular_step
        self.g_scheduler = StepLR(self.g_optimizer, step_size=self.g_lr_schedular_step, gamma=self.g_lr_decay_factor)

        self.num_epochs = args.num_epochs
        self.n_freq = args.n_freq

        '''set psnr and ssim'''
        self.psnr = args.psnr
        self.ssim = args.ssim

        '''save directory'''
        self.save_checkpoints_dir = args.save_checkpoints_dir
        self.train_images_dir = args.train_images_dir
        self.plot_dir = args.plot_dir
        self.loss_dir = args.loss_dir


    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def set_gen_loss(self):
        for loss in self.gen_loss_type:
            if loss in ['l1', 'L1']:
                print("Using L1 loss")
                self.gen_losses['l1_loss'] = nn.L1Loss()
                self.loss_keys_list.append('l1_loss')
            elif loss in ['l2','L2','mse','MSE']:
                print("Using MSE loss")
                self.gen_losses['mse_loss'] = nn.MSELoss()
                self.loss_keys_list.append('mse_loss')
            elif loss in ['gram','Gram']:
                print("Using Gram loss")
                self.gen_losses['gram_loss'] = GramLoss()
                self.loss_keys_list.append('gram_loss')
            elif loss in ['ssim','SSIM']:
                print("Using SSIM loss")
                self.gen_losses['ssim_loss'] = SSIM()
                self.loss_keys_list.append('ssim_loss')
            elif loss in ['laplacian','laplacian_pyramid', 'pyramid']:
                print("Using Laplacian Pyramid loss")
                self.gen_losses['laplacian_pyramid'] = LaplacianPyramidLoss()
                self.loss_keys_list.append('laplacian_pyramid')
            elif loss in ['tv_regularizer','regularizer', 'tv']:
                print("Using TV regularizer")
                self.gen_losses['tv_regularizer'] = TVRegularizer()
                self.loss_keys_list.append('tv_regularizer')
            elif loss in ['ranker_loss','ranker', 'classification_loss', 'classification']:
                print("Using Ranker Loss")
                self.gen_losses['ranker_loss'] = RankLoss(checkpoint_path=self.args.ranker_checkpoint_path, device=self.devie)
                self.loss_keys_list.append('ranker_loss')

    def val_epoch(self):
        self.G.eval()
        l1_loss = nn.L1Loss()
        mse = nn.MSELoss()
        count,psnr,ssim,loss,l1,hfen = 0,0,0,0,0,0
        with torch.no_grad():
            for data in self.eval_dataloader:
                images = data['lr'].to(self.device)
                labels = data['hr'].to(self.device)

                output = self.G(images)  

                loss += mse(output,labels) 
                l1 += l1_loss(output,labels)

                count += len(labels)
                output = output.clamp(0.0,1.0)

                #psnr and ssim using tensor
                psnr += self.psnr(output, labels)
                ssim += self.ssim(output,labels)

                output = output.squeeze().detach().to('cpu').numpy()
                label = labels.squeeze().detach().to('cpu').numpy()
                hfen += hfen_error(output, label)
        return loss.item()/count, l1.item()/count,psnr.item()/count, ssim.item()/count,hfen.item()/count

    def save_train_example(self,images,fake_images,epoch, plot_label=True, labels=None,):
        batch_size, channels, height, width = images.shape

        print("The shape if images, labels and fake images is", images.shape,labels.shape,fake_images.shape)

        # Choose a random image index from the batch
        # random_image_index = torch.randint(0, batch_size, size=(1,)).item()
        random_image_index = torch.tensor(0)


        input_image =  images[random_image_index,:,:,:].squeeze().detach().cpu().numpy().astype('float')
        output_image = fake_images[random_image_index,:,:,:].squeeze().detach().cpu().numpy().astype('float')

        if plot_label:
            label_image =  labels[random_image_index,:,:,:].squeeze().detach().cpu().numpy().astype('float')
            i = 3
        else:
            i = 2


        fig = plt.figure()
        ax1 = fig.add_subplot(1,i,1)
        ax1.imshow(input_image,cmap='gray')
        ax1.set_title("Input image")


        ax2 = fig.add_subplot(1,i,2)
        ax2.imshow(output_image, cmap='gray')
        ax2.set_title("Output image")

        if plot_label:
            ax3 = fig.add_subplot(1,i,3)
            ax3.imshow(label_image,cmap='gray')
            ax3.set_title("label image")

        image_name = 'image_plot_'+str(epoch)+'.png'
        image_path = os.path.join(self.plot_dir, image_name)

        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Save the full figure...
        fig.savefig(image_path)


    def train_epoch(self,epoch):
        epoch_losses = create_loss_meters_from_keys(self.loss_keys_list)  # create average meter for all loss
        start_time = time.time()
        with tqdm(total=(len(self.train_dataset) - len(self.train_dataset) % self.train_batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, self.num_epochs - 1))

            for idx, (data) in enumerate(self.train_dataloader):
                images = data['lr'].to(self.device)
                labels = data['hr'].to(self.device)
                batch_size = images.shape[0]

                #training generator
                self.set_requires_grad(self.G,True)
                self.g_optimizer.zero_grad()

                fake_images = self.G(images)
        
                loss_G_pix = 0
                # for idx,(gen_loss,gen_loss_wt) in enumerate(zip(self.gen_losses,self.gen_loss_wt)):
                #     pix_loss_item = gen_loss(fake_images,labels)
                #     print(idx, pix_loss_item)
                #     loss_G_pix += pix_loss_item * gen_loss_wt

                for idx ,(key, loss_obj) in enumerate(self.gen_losses.items()):
                    pix_loss_wt = self.gen_loss_wt[idx]
                    if key in ['tv_regularizer','tv','ranker_loss','ranker', 'classification_loss', 'classification']:
                        pix_loss_value = loss_obj(fake_images)
                    else:
                        pix_loss_value = loss_obj(fake_images,labels)

                    loss_G_pix += pix_loss_value * pix_loss_wt
                    # print(key, pix_loss_value)
                    epoch_losses[key].update(pix_loss_value.item(),batch_size)
                    

                loss_G = loss_G_pix
            
                loss_G.backward()
                self.g_optimizer.step()

                # update average meter on every iteration
                epoch_losses['loss_G_pix'].update(loss_G_pix.item(),batch_size)

                # print("Finished one iteration")
                # print("*****************************************************************************************")


            log_results(epoch_losses) # function to print out the average per epoch


            t.set_postfix(loss='{:.6f}'.format(epoch_losses['loss_G_pix'].avg))
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
                    self.save_50_micron_train_example(model=self.G, epoch=epoch)



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

        self.G.train()

        for epoch in range(self.num_epochs+1):
            # self.lr_G = adjust_learning_rate(self.g_optimizer, epoch,self.lr_G)
            '''train one epoch and evaluate the model'''

            output, epoch_losses = self.train_epoch(epoch)
            self.g_scheduler.step()


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

    def save_50_micron_train_example( self,model = None, epoch=20):

        # image_path = 'lr_f1_160_z_75.png'
        images = cv2.imread('lr_f1_160_z_75.png', cv2.IMREAD_UNCHANGED)
        # print("Reached here")
        # print(images)
        # quit();
        images = images/255.

        images = torch.from_numpy(images).float().unsqueeze(0).unsqueeze(0)
        print("shape of images is", images.shape)
        print("range of image is", images.min(), images.max())

        device = next(model.parameters()).device
        images = images.to(device)

        with torch.no_grad():
            out = forward_chop(model, images) #model(im_input)
            torch.cuda.synchronize()

        # input_image =  images.squeeze().cpu().numpy().astype('float')
        output_image =  out.detach().squeeze().cpu().numpy().astype('float')

        output_image = (output_image*255.).astype('uint8')


        image_name = 'image_plot_'+str(epoch)+'.png'
        image_path = os.path.join(self.plot_image_example, image_name)

        if not os.path.exists(self.plot_image_example):
            os.makedirs(self.plot_image_example)

        cv2.imwrite(image_path, output_image)