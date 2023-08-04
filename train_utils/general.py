import pickle  
import os
import glob
import torch
from PIL import Image
import numpy as np 
import wandb
import json
import torch.nn as nn
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load(path):
  if os.path.isfile(path):
    data=[path]
  elif os.path.isdir(path):
    data=list(glob.glob(path+'/*.jpg'))+list(glob.glob(path+'/*.png'))
  return data


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0)) ),0, 1) * 255.0
    return image_numpy.astype(imtype)

def model_num_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def print_layerwise_params(model):
    tensor_list = list(model.items())
    for layer_tensor_name, tensor in tensor_list:
        print('Layer {}: {} elements'.format(layer_tensor_name, torch.numel(tensor)))

    print('Total no of params', model_num_params(model))


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def save_configuration_yaml(opt,save_name='configuration.yaml'):
    if not os.path.exists(opt.loss_dir):
        os.makedirs(opt.loss_dir)
    save_path = os.path.join(opt.loss_dir,save_name)
    opt.device = 'cuda'
    opt.psnr ='psnr'
    opt.ssim = 'ssim'
    with open(save_path, 'w') as f:
        json.dump(opt.__dict__, f, indent=2)
    return save_path

def load_model(checkpoint,device,model=None):
    # model = MainModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device
        )
    )
    return model



def normalize_array(arr):
    deno = arr.max().item()-arr.min().item()
    return (arr-arr.min().item())/deno

def preprocess(image_tensor,normalize=False):
    image_tensor = image_tensor.squeeze().detach().to("cpu").float().numpy()
    if normalize:
        image_tensor=normalize_array(image_tensor)
    image_tensor = image_tensor*255.
    image_tensor = image_tensor.clip(0,255)
    image_tensor = image_tensor.astype('uint8')
    # image_tensor = Image.fromarray(image_tensor)
    return image_tensor




def min_max_normalize(arr):
    return (arr-arr.min())/(arr.max()-arr.min())


def read_pickle(path):
    import pickle

    with open(path, 'rb') as f:
        x = pickle.load(f)
    return x




'''reduce learning rate of optimizer by half on every  150 and 225 epochs'''
def adjust_learning_rate(optimizer, epoch,lr,lr_factor=0.5):
    if lr <= 0.0000001:
        return lr
    else:
        if epoch % 30 == 0 or epoch % 50 == 0:
            lr = lr * lr_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr



def create_loss_meters_gan():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_fake = AverageMeter()
    loss_G_real= AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_fake': loss_G_fake,
            'loss_G_real': loss_G_real,
            'loss_G_Gan': loss_G_GAN,
            'loss_G_pix': loss_G_L1,
            'loss_G': loss_G}

def create_loss_meters_from_keys(key_list):
    dict = {}
    for key in key_list:
        dict.update({key: AverageMeter()})
    return dict



def log_results(loss_meter_dict):
    print("\n")
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LogOutputs():
    def __init__(self):
        self.epoch_list =[]
        self.images_list =[]
        self.labels_list = []
        self.predictions_list=[]

    def append_list(self,epoch,images,labels,predictions):
        self.epoch_list.append(epoch)
        self.images_list.append(preprocess(images[0]))
        self.labels_list.append(preprocess(labels[0]))
        self.predictions_list.append(preprocess(predictions[0]))

    def append_list_classification(self,epoch,images,labels,predictions): #for appending image and their rank classification
        self.epoch_list.append(epoch)
        self.images_list.append(preprocess(images[0]))
        self.labels_list.append(labels[0])
        self.predictions_list.append(predictions[0])

    def append_list_3d(self,epoch,images,labels,predictions):
        self.epoch_list.append(epoch)
        self.images_list.append(preprocess(images))
        self.labels_list.append(preprocess(labels))
        self.predictions_list.append(preprocess(predictions))

    def log_images(self,columns=["epoch","image", "pred", "label"],wandb=None, images =True):
        table = wandb.Table(columns=columns)
        if images:
            for epoch,img, pred, targ in zip(self.epoch_list,self.images_list,self.predictions_list,self.labels_list):
                table.add_data(epoch, wandb.Image(img),wandb.Image( pred),wandb.Image(targ))
        else:
           for epoch,img, pred, targ in zip(self.epoch_list,self.images_list,self.predictions_list,self.labels_list):
                table.add_data(epoch, wandb.Image(img),pred,targ) 
        wandb.log({"outputs_table":table}, commit=False)


class LogMetric(object):
    def __init__(self,training_dict=None):
        if training_dict:
            self.training_dict = training_dict
        else:
            self.training_dict = { 'train_loss' : []}
        self.val_dict={'val_loss': [], 'val_l1': [],'val_psnr':[],'val_ssim':[],'val_hfen':[],'epoch':[]}
        self.log_dict ={}

    def update_dict_with_key(self,key,value,training=True):
        if training:
            self.training_dict[key].append(value)
        else:
            self.val_dict[key].append(value)

    def update_dict(self,value=[],training=True):
        for (k, v) in zip(self.get_dict_keys(training), value):
            if training:
                self.training_dict[k].append(v)
            else:
                self.val_dict[k].append(v)

    def get_dict_keys(self,training=True):
        if training:
            return list(self.training_dict.keys()) 
        else:
            return list(self.val_dict.keys()) 

    def update_dict_with_dictionary (self,epoch_losses):
        for key, value in epoch_losses.items():
            self.training_dict[key].append(value)


    def update_log_dict(self):
        self.log_dict = {**self.training_dict, **self.val_dict}
        return self.log_dict

    def save_dict(self,opt,save_name='loss_metric'):
        if not os.path.exists(opt.loss_dir):
            os.makedirs(opt.loss_dir)
        save_path=os.path.join(opt.loss_dir,save_name)
        
        self.update_log_dict()
        with open(save_path,"wb") as fp:
            pickle.dump(self.log_dict,fp)
        return save_path 


def hfen_error(original_arr,est_arr,sigma=3):
   original = ndimage.gaussian_laplace(original_arr,sigma=sigma)
   est = ndimage.gaussian_laplace(est_arr,sigma=sigma)
   num = np.sum(np.square(original-est))
   deno = np.sum(np.square(original))
   hfen = np.sqrt(num/deno)
   return hfen

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



def save_50_micron_train_example( model = None, epoch=20, plot_dir= 'example_plot'):

    image_path = 'lr_f1_160_z_75.png'
    images = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    images = images/255.

    images = torch.from_numpy(images).float().unsqueeze(0).unsqueeze(0)
    print("shape of images is", images.shape)

    device = next(model.parameters()).device
    images = images.to(device)

    with torch.no_grad():
        out = forward_chop(model, images) #model(im_input)
        torch.cuda.synchronize()

    # input_image =  images.squeeze().cpu().numpy().astype('float')
    output_image =  out.detach().squeeze().cpu().numpy().astype('float')

    output_image = (output_image*255).astype('uint8')


    image_name = 'image_plot_'+str(epoch)+'.png'
    image_path = os.path.join(plot_dir, image_name)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    cv2.imwrite(image_path, output_image)


if __name__== '__main__':
    save_50_micron_train_example()