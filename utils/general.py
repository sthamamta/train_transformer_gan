import pickle  
import os
import glob
import torch
from PIL import Image
import numpy as np 
import wandb
import json
import torch.nn as nn

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


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))
        

def calculate_psnr_np(img1, img2):
    import numpy as np
    SE_map = (1.*img1-img2)**2
    cur_MSE = np.mean(SE_map)
    return 20*np.log10(255./np.sqrt(cur_MSE))

def calculate_psnr_torch(img1, img2):
    SE_map = (1.*img1-img2)**2
    cur_MSE = torch.mean(SE_map)
    return 20*torch.log10(1./torch.sqrt(cur_MSE))


def save_configuration(opt,save_name='configuration'):
    if not os.path.exists(opt.loss_dir):
        os.makedirs(opt.loss_dir)
    save_path=os.path.join(opt.loss_dir,save_name)
    opt.device = 'cuda'
    opt.psnr ='psnr'
    opt.ssim = 'ssim'
    with open(save_path,"wb") as fp:
        pickle.dump(opt,fp)
    return save_path 

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



def log_output_images(images, predicted, labels):
    images = preprocess(images[0])
    predicted = preprocess(predicted[0])
    labels = preprocess(labels[0])
    "Log a wandb.Table with (img, pred, label)"
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", "pred", "label"])
    # for img, pred, targ in zip(images, predicted, labels):
    # table.add_data(images, predicted, labels)
    table.add_data(wandb.Image(images),wandb.Image(predicted),wandb.Image(labels))
    wandb.log({"outputs_table":table}, commit=False)


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

    def append_list_3d(self,epoch,images,labels,predictions):
        self.epoch_list.append(epoch)
        self.images_list.append(preprocess(images))
        self.labels_list.append(preprocess(labels))
        self.predictions_list.append(preprocess(predictions))

    def log_images(self,columns=["epoch","image", "pred", "label"],wandb=None):
        table = wandb.Table(columns=columns)
        for epoch,img, pred, targ in zip(self.epoch_list,self.images_list,self.predictions_list,self.labels_list):
            table.add_data(epoch, wandb.Image(img),wandb.Image( pred),wandb.Image(targ))
        wandb.log({"outputs_table":table}, commit=False)


class LogEdgesOutputs():
    def __init__(self):
        self.epoch_list =[]
        self.hr_list =[]
        self.final_output_list = []

        self.lr_list=[]
        self.label_edges_list=[]
        self.pred_edges_list = []
        self.input_edges_list = []
        self.mask_list =[]

    def append_list(self,output_dict):
    # def append_list(self,epoch,hr,final_output,lr,label_edges):
        print('Images appended')
        self.epoch_list.append(output_dict['epoch'])
        self.hr_list.append(preprocess(output_dict['hr'][0]))
        self.final_output_list.append(preprocess(output_dict['final_output'][0]))

        self.lr_list.append(preprocess(output_dict['lr'][0]))
        self.label_edges_list.append(preprocess(output_dict['label_edges'][0],normalize=True))
        self.pred_edges_list.append(preprocess(output_dict['pred_edges'][0],normalize=True))
        self.input_edges_list.append(preprocess(output_dict['input_edges'][0],normalize=True))
        if output_dict['mask'] is not None:
            self.mask_list.append(preprocess(output_dict['mask'][0]))

    def log_images(self,columns=["epoch","hr","final output", "lr", "label edges","pred edges","input edges"],wandb=None):
        columns=["epoch","hr","final output", "lr", "label edges","pred edges","input edges"]
        table = wandb.Table(columns=columns)
        for epoch,hr,final_output,lr,label_edges,pred_edges,input_edges in zip(self.epoch_list,self.hr_list,self.final_output_list,self.lr_list,self.label_edges_list,self.pred_edges_list,self.input_edges_list):
        # for epoch,hr,final_output,lr in zip(self.epoch_list,self.hr_list,self.final_output_list,self.lr_list,self.label_edges_list):
            table.add_data(epoch,
             wandb.Image(hr),
             wandb.Image(final_output),
             wandb.Image(lr),
             wandb.Image(label_edges),
             wandb.Image(pred_edges),
             wandb.Image(input_edges),
             )
        wandb.log({"outputs_table":table}, commit=False)

    def log_images_and_mask(self,columns=["epoch","hr","final output", "lr", "label edges","pred edges","input edges","mask"],wandb=None):
        columns=["epoch","hr","final output", "lr", "label edges","pred edges","input edges","mask"]
        table = wandb.Table(columns=columns)
        for epoch,hr,final_output,lr,label_edges,pred_edges,input_edges,mask in zip(self.epoch_list,self.hr_list,self.final_output_list,self.lr_list,self.label_edges_list,self.pred_edges_list,self.input_edges_list,self.mask_list):
        # for epoch,hr,final_output,lr in zip(self.epoch_list,self.hr_list,self.final_output_list,self.lr_list,self.label_edges_list):
            table.add_data(epoch,
             wandb.Image(hr),
             wandb.Image(final_output),
             wandb.Image(lr),
             wandb.Image(label_edges),
             wandb.Image(pred_edges),
             wandb.Image(input_edges),
             wandb.Image(mask)
             )
        wandb.log({"outputs_table":table}, commit=False)


def min_max_normalize(arr):
    return (arr-arr.min())/(arr.max()-arr.min())


def read_pickle(path):
    import pickle

    with open(path, 'rb') as f:
        x = pickle.load(f)
    return x


class NRMSELoss(nn.Module):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.eps = eps
        
    def forward(self,y,yhat):
        numerator = torch.sqrt(self.mse(yhat,y) + self.eps)
        zeros = torch.zeros(y.shape).to(y.get_device())
        denominator = torch.sqrt(self.mse(y,zeros)+self.eps)
        loss = numerator/denominator
        return loss