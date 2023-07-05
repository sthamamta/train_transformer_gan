
# https://github.com/yjn870/SRDenseNet-pytorch/blob/5cf7af791fdd200441d71de5e3b3d4e8c3941e9c/utils.py (train code and model)
# https://github.com/andreasveit/densenet-pytorch/blob/master/train.py (logger and learning rate adjusting)

import torch
import numpy as np
import os
from PIL import Image
from torchvision.transforms import ToTensor
import glob
from scipy import ndimage
from torchvision.transforms import functional as F
import cv2
from typing import Any



def calc_psnr(img1, img2):
    return 10. * np.log10(1. / np.mean((img1 - img2) ** 2))


def create_dictionary(image_dir,label_dir):
    lst = []
    for f in os.listdir(image_dir):
        if not f.startswith('.'):
            lst.append(f)
        else:
            pass
    lst.sort()
    label_lst=[]
    for f in os.listdir(label_dir):
        if not f.startswith('.'):
            label_lst.append(f)
        else:
            pass
    label_lst.sort()
   
    dir_dictionary={}
    for i in range(len(lst)):
        dir_dictionary[lst[i]]=label_lst[i]
        
    return dir_dictionary

def min_max_normalize(image):
    max_img = image.max()
    min_img = image.min()
    denom = max_img-min_img
    norm_image = (image-min_img)/denom
    return norm_image 

def normalize(image,max_value=None):
    if max_value:
        return image/max_value
    else:
        return image/image.max()   


def Average(lst):
    return sum(lst) / len(lst)


def check_image(fn):
    try:
        im = Image.open(fn)
        im.verify()
        return True
    except:
        return False
    
def check_image_dir(path):
    for fn in glob.glob(path):
        if not check_image(fn):
            print("Corrupt image: {}".format(fn))
            os.remove(fn)

def save_img(image_tensor, filename):
   image_numpy = image_tensor.squeeze().detach().to('cpu').float().numpy()
   image_numpy = min_max_normalize(image_numpy)
   image_numpy=image_numpy*255
   image_numpy = image_numpy.clip(0, 255)
   image_numpy = image_numpy.astype(np.uint8)
   image_pil = Image.fromarray(image_numpy)
   image_pil.save(filename)
   print("Image saved as {}".format(filename))


def apply_model(model,epoch,opt,addition=False):
    image= Image.open(opt.epoch_image_path)
    image_tensor = ToTensor()(image)
    image_tensor= torch.unsqueeze(image_tensor.float(),0)
    image_tensor =  image_tensor.to(opt.device)
    image_tensor = min_max_normalize(image_tensor)
    output = model(image_tensor)
    if addition:
        output=output+image_tensor
        output = min_max_normalize(output)
    if not os.path.exists(opt.epoch_images_dir):
        os.makedirs(opt.epoch_images_dir)
    path = os.path.join(opt.epoch_images_dir,'epoch_{}.png'.format(epoch))
    # file_name = opt.epoch_images_dir+'/epoch_{}.png'.format(epoch)
    save_img(output,path)
    return True

def apply_model_edges(model,epoch,opt):
    image= Image.open(opt.epoch_image_path)

    image = cv2.imread(opt.epoch_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(image = image, threshold1=1, threshold2=20)

    image = image.astype(np.float32) / 255.
    edges = edges.astype(np.float32) / 255.
    
    image_tensor = image2tensor(image, range_norm=False, half=False)
    image_tensor= torch.unsqueeze(image_tensor.float(),0)
   
    edge_tensor = image2tensor(edges, range_norm=False, half=False)
    edge_tensor= torch.unsqueeze(edge_tensor.float(),0)
    
    edge_tensor.to(opt.device)

    outputs = model(edge_tensor)
    outputs.to(opt.device)

    outputs = outputs.to(opt.device) + image_tensor.to(opt.device)
    # outputs = min_max_normalize(outputs)
    if not os.path.exists(opt.epoch_images_dir):
        os.makedirs(opt.epoch_images_dir)

    path = os.path.join(opt.epoch_images_dir,'epoch_{}.png'.format(epoch))
    image_numpy = outputs.squeeze().detach().to('cpu').float().numpy()

    image_numpy = image_numpy*255.
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    cv2.imwrite(path, image_numpy)
    print("Image saved as {}".format(path))
    return True

#incomplete need to write
def apply_model_3d(model,epoch,opt,addition=False):
    image = np.load(opt.epoch_image_path)
    image = torch.from_numpy(image)
    image = min_max_normalize(image)
    image_tensor= torch.unsqueeze(image.float(),0)
    output = model(image_tensor)
    if addition:
        output = output.to(opt.device)+image_tensor.to(opt.device)
    if not os.path.exists(opt.epoch_images_dir):
        os.makedirs(opt.epoch_images_dir)
    path = os.path.join(opt.epoch_images_dir,'epoch_{}.png'.format(epoch))
    image_numpy = output.squeeze().detach().to('cpu').float().numpy()
    image_numpy = min_max_normalize(image_numpy)
    image_numpy = image_numpy*255.
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    cv2.imwrite(path, image_numpy)
    print("Image saved as {}".format(path))
    print('apply model 3d not implemented')
    return True

def apply_model_using_cv(model,epoch,opt,addition=False):
    image= Image.open(opt.epoch_image_path)

    image = cv2.imread(opt.epoch_image_path).astype(np.float32) / 255.

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_tensor = image2tensor(image, range_norm=False, half=False)
    image_tensor= torch.unsqueeze(image_tensor.float(),0)
   
    output = model(image_tensor)

    if addition:
        output = output.to(opt.device)+image_tensor.to(opt.device)
        # output = min_max_normalize(output)
    if not os.path.exists(opt.epoch_images_dir):
        os.makedirs(opt.epoch_images_dir)

    path = os.path.join(opt.epoch_images_dir,'epoch_{}.png'.format(epoch))
    image_numpy = output.squeeze().detach().to('cpu').float().numpy()
    image_numpy = min_max_normalize(image_numpy)
    image_numpy = image_numpy*255.
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    cv2.imwrite(path, image_numpy)
    print("Image saved as {}".format(path))
    return True


def hfen_error(original_arr,est_arr,sigma=3):
   original = ndimage.gaussian_laplace(original_arr,sigma=sigma)
   est = ndimage.gaussian_laplace(est_arr,sigma=sigma)
   num = np.sum(np.square(original-est))
   deno = np.sum(np.square(original))
   hfen = np.sqrt(num/deno)
   return hfen

# https://github.com/Lornatang/ESRGAN-PyTorch/blob/main/imgproc.py
def image2tensor(image: np.ndarray, range_norm: bool=False, half: bool=False) -> torch.Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch
    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type
    Returns:
        tensor (torch.Tensor): Data types supported by PyTorch
    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image2tensor(example_image, range_norm=True, half=False)
    """
    # Convert image data type to Tensor data type
    tensor = F.to_tensor(image)

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor


def tensor2image(tensor: torch.Tensor, range_norm: bool=False, half: bool=False,color:bool=False) -> Any:
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type
    Args:
        tensor (torch.Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.
    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV
    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image2tensor(example_image, range_norm=False, half=False)
    """
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)
    if half:
        tensor = tensor.half()

    if color:
      image = tensor.detach().squeeze().permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
    else:
        image = tensor.detach().squeeze().mul(255).clamp(0, 255).cpu().numpy().astype("uint8") 

    return image




def crop_pad_kspace(data,pad=False,factor=2):  #function for cropping and/or padding the image in kspace
    F = np.fft.fft2(data)
    fshift = np.fft.fftshift(F)
    
    y,x = fshift.shape
    data_pad = np.zeros((y,x),dtype=np.complex_)
    center_y = y//2 #defining the center of image in x and y direction
    center_x = x//2
    startx = center_x-(x//(factor*2))  
    starty = center_y-(y//(factor*2))
    
    arr = fshift[starty:starty+(y//factor),startx:startx+(x//factor)]
    if pad:
        data_pad[starty:starty+(y//factor),startx:startx+(x//factor)] = arr
        img_reco_cropped = np.fft.ifft2(np.fft.ifftshift(data_pad))
    else:
        img_reco_cropped = np.fft.ifft2(np.fft.ifftshift(arr)) 
    return np.abs(img_reco_cropped )



def create_freq_mask(shape,factor=2, lower=False):

  y,x = shape
  mask_lower = np.zeros((y,x), dtype=np.complex_)

  center_y = y//2 #defining the center of image in x and y direction
  center_x = x//2
  startx = center_x-(x//(factor*2))  
  starty = center_y-(y//(factor*2))

  mask_lower[starty:starty+(y//factor),startx:startx+(x//factor)] = 1
  mask_upper = 1-mask_lower
  if lower:
    return mask_lower
  else:
    return mask_upper

def get_high_freq(image,factor=2):
  fshift = np.fft.fftshift(np.fft.fft2(image))
  mask = create_freq_mask(fshift.shape,factor=factor, lower=False)
  high_freq = fshift*mask
  return np.abs(np.fft.ifft2(np.fft.ifftshift(high_freq)))
