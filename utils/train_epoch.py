
from tqdm import tqdm
from tensorboard_logger import configure, log_value
import torch
import os
import torch.nn as nn
# from skimage.metrics import peak_signal_noise_ratio
# from skimage.metrics import structural_similarity
# from utils.general import min_max_normalize
from utils.logging_metric import update_epoch_losses
from utils.preprocess import hfen_error



def train_epoch_srdense(opt,model,criterion,optimizer,train_dataset,train_dataloader,epoch,epoch_losses): 
    with tqdm(total=(len(train_dataset) - len(train_dataset) % opt.train_batch_size), ncols=80) as t:
        t.set_description('epoch: {}/{}'.format(epoch, opt.num_epochs - 1))

        for idx, data in enumerate(train_dataloader):
            images = data['lr'].to(opt.device)
            labels = data['hr'].to(opt.device)
            preds = model(images)

            # print("input shape", images.shape)
            # print("labels shape", labels.shape)
            # print("preds shape", preds.shape)
            # quit();
           
            loss = criterion(preds, labels)

            # epoch_losses.update(loss.item(), len(images))
            update_epoch_losses(epoch_losses, count=len(images),values=[loss.item()])
            
            optimizer.zero_grad()
            # a = list(model.parameters())[0].clone()
            loss.backward()
            optimizer.step()
            # b = list(model.parameters())[0].clone()
            # print("is parameter same after backpropagation?",torch.equal(a.data, b.data))

        t.set_postfix(loss='{:.6f}'.format(epoch_losses['train_loss'].avg))
        t.update(len(images))

        if epoch % opt.n_freq==0:
            if not os.path.exists(opt.checkpoints_dir):
                os.makedirs(opt.checkpoints_dir)
            path = os.path.join(opt.checkpoints_dir, 'epoch_{}_f_{}.pth'.format(epoch,opt.factor))
            if opt.data_parallel:
                model.module.save(model.state_dict(),opt,path,optimizer.state_dict(),epoch)
            else:
                model.save(model.state_dict(),opt,path,optimizer.state_dict(),epoch)
    

    return images,labels,preds



def validate_srdense(opt,model, dataloader,criterion=nn.MSELoss()):
    model.eval()
    l1_loss = nn.L1Loss()
    count,psnr,ssim,loss,l1,hfen = 0,0,0,0,0,0
    with torch.no_grad():
        for data in dataloader:  #batch size is always 1 to calculate psnr and ssim
            image = data['lr'].to(opt.device)
            label = data['hr'].to(opt.device)
            output = model(image)

            loss += criterion(output,label) 
            l1 += l1_loss(output,label)
            count += len(label)
            output = output.clamp(0.0,1.0)

            #psnr and ssim using tensor
            psnr += opt.psnr(output, label)
            ssim += opt.ssim(output,label)


            output = output.squeeze().detach().to('cpu').numpy()
            # image = image.squeeze().to('cpu').numpy()
            label = label.squeeze().detach().to('cpu').numpy()
            # psnr += peak_signal_noise_ratio(output, label)
            # ssim += structural_similarity(output, label)
            hfen += hfen_error(output, label)


    return loss.item()/count, l1.item()/count,psnr.item()/count, ssim.item()/count,hfen/count

