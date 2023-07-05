from train_utils.general import *
import wandb
import copy
import matplotlib.pyplot as plt
from loss.gram_loss import GramLoss
from loss.ssim_loss import SSIM
from torch.optim.lr_scheduler import StepLR
from loss.laplacian_pyramid_loss import LaplacianPyramidLoss
from loss.tv_regularizer import TVRegularizer


class Trainer(object):

    def __init__(self, args):
        print("Trainer super init model.")
        self.args = args
        self.resume = args.resume
        if self.resume:
            self.start_epoch = args.start_epoch
        else: 
            self.start_epoch = 0

        self.G = args.generator
        self.D = args.discriminator
        self.wandb_obj = args.wandb_obj

        self.plot_train_example = args.plot_train_example

        self.device = args.device
        self.wandb = args.wandb  # true of false
        self.eval = args.eval #True or false


        self.train_dataset = args.train_dataset
        self.train_dataloader =  args.train_dataloader
        self.train_batch_size = args.train_batch_size

        if self.eval:
            self.eval_dataloader = args.eval_dataloader
            self.eval_datasets = args.eval_datasets 
            self.eval_batch_size = args.eval_batch_size

        self.weight_cliping_limit = 0.01

        if self.eval:
            self.eval_freq = args.eval_freq
        else:
            self.eval_freq = None

    
        self.d_optimizer = args.d_optimizer
        self.g_optimizer = args.g_optimizer

        self.g_lr_decay_factor = args.g_lr_decay_factor
        self.g_lr_schedular_step = args.g_lr_schedular_step
        self.g_scheduler = StepLR(self.g_optimizer, step_size=self.g_lr_schedular_step, gamma=self.g_lr_decay_factor)

        self.d_lr_decay_factor = args.d_lr_decay_factor
        self.d_lr_schedular_step = args.d_lr_schedular_step
        self.d_scheduler = StepLR(self.d_optimizer, step_size=self.d_lr_schedular_step, gamma=self.d_lr_decay_factor)

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

    # def get_infinite_batches(self):
    #     while True:
    #         for idx, (data) in enumerate(self.train_dataloader):
    #             images = data['lr'].to(self.device)
    #             labels = data['hr'].to(self.device)
    #             yield images,labels

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def set_gen_loss(self):
        for loss in self.gen_loss_type:
            if loss in ['l1', 'L1']:
                print("Using L1 loss")
                self.gen_losses['l1_loss'] = nn.L1Loss()
                self.generator_loss_key_list.append('l1_loss')
            elif loss in ['l2','L2','mse','MSE']:
                print("Using MSE loss")
                self.gen_losses['mse_loss'] = nn.MSELoss()
                self.generator_loss_key_list.append('mse_loss')
            elif loss in ['gram','Gram']:
                print("Using Gram loss")
                self.gen_losses['gram_loss'] = GramLoss()
                self.generator_loss_key_list.append('gram_loss')
            elif loss in ['ssim','SSIM']:
                print("Using SSIM loss")
                self.gen_losses['ssim_loss'] = SSIM()
                self.generator_loss_key_list.append('ssim_loss')
            elif loss in ['laplacian','laplacian_pyramid', 'pyramid']:
                print("Using Laplacian Pyramid loss")
                self.gen_losses['laplacian_pyramid'] = LaplacianPyramidLoss()
                self.generator_loss_key_list.append('laplacian_pyramid')
            elif loss in ['tv_regularizer','regularizer', 'tv']:
                print("Using TV regularizer")
                self.gen_losses['tv_regularizer'] = TVRegularizer()
                self.generator_loss_key_list.append('tv_regularizer')

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
        label_image =  labels[random_image_index,:,:,:].squeeze().detach().cpu().numpy().astype('float')
        output_image = fake_images[random_image_index,:,:,:].squeeze().detach().cpu().numpy().astype('float')


        if plot_label:
            label_image =  labels[random_image_index,:,:,:].squeeze().detach().cpu().numpy().astype('float')
            i = 3
        else:
            i = 2


        fig = plt.figure()
        ax1 = fig.add_subplot(1,i,1)
        ax1.imshow(input_image, cmap='gray')
        ax1.set_title("Input image")


        ax2 = fig.add_subplot(1,i,2)
        ax2.imshow(output_image,cmap='gray')
        ax2.set_title("Output image")

        if plot_label:
            ax3 = fig.add_subplot(1,i,3)
            ax3.imshow(label_image, cmap='gray')
            ax3.set_title("label image")

        image_name = 'image_plot_'+str(epoch)+'.png'
        image_path = os.path.join(self.plot_dir, image_name)

        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Save the full figure...
        fig.savefig(image_path)

    def train_epoch():
      pass

    
      
    def train(self, train_loader):
       pass

       