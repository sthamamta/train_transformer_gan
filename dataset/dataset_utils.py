
'''With the use of this dataset, we downsample the hr image by factor 2 to get lr using various methods and train model to increase the size of the image'''

import torch

import numpy as np


import numpy as np
import nibabel as nib
import torch
from PIL import Image
import math
import cv2

import warnings
warnings.filterwarnings("ignore")


def min_max_normalize(image):
    max_img = image.max()
    min_img = image.min()
    denom = (max_img-min_img) + 0.00000000001
    norm_image = (image-min_img)/denom
    return norm_image 


def load_data_nii(fname):
    img = nib.load(fname)
    data = img.get_fdata()
    data_norm = torch.from_numpy(data)
    return data_norm 


def prepare_lr_array(hr_array,factor=2,pad=True):
    # 3d fourier transform
    spectrum = np.fft.fft2(hr_array)  

    # Apply frequency shift along spatial dimentions 
    spectrum_sh = np.fft.fftshift(spectrum, axes=(0,1))  

    x,y = spectrum_sh.shape
    data_pad = np.zeros((x,y),dtype=np.complex_)

    center_y = y//2 #defining the center of image in x, y and z direction
    center_x = x//2

    startx = center_x-(x//(factor*2))  
    starty = center_y-(y//(factor*2))

    arr = spectrum_sh[startx:startx+(x//factor),starty:starty+(y//factor)]
      
    if pad:
        data_pad[startx:startx+(x//factor),starty:starty+(y//factor)] = arr
        img_reco_cropped = np.fft.ifft2(np.fft.ifftshift(data_pad))
    else:
        img_reco_cropped = np.fft.ifft2(np.fft.ifftshift(arr)) 
        
    return np.abs(img_reco_cropped)



def downsample_bicubic(hr_image,upscale_factor=2, double_downsample=False):


    # hr_image = min_max_normalize(hr_image)
    # hr_image = hr_image * 255.


    hr_image = hr_image.astype(np.uint8)
    image = Image.fromarray(hr_image)

    x,y = image.size
    
    if double_downsample:
        shape = (x//(2*upscale_factor), y//(2*upscale_factor))  #downsample by factor 4
    else:
        shape = (x//(upscale_factor), y//(upscale_factor))  #downsample by factor 2

    image = image.resize(shape,Image.BICUBIC)

    if double_downsample:
        shape = (x//upscale_factor, y//upscale_factor)  #upsample by factor 2
        image = image.resize(shape,Image.BICUBIC)

    image = np.array(image).astype(np.float32)
    # image = min_max_normalize(image)

    return image


def downsample_image_with_mode(array,factor=2, mode='Lanczos',double_downsample = False):

    # array = min_max_normalize(array)
    # array = hr_image * 255.

    array = array.astype(np.uint8)
    image = Image.fromarray(array)

    x,y = image.size

    if double_downsample:
        shape = (x//(2*factor), y//(2*factor))  #downsample by factor -> 2*factor
    else:
        shape = (x//(factor), y//(factor))  #downsample by factor -> factor
    
    if mode in ['nearest','Nearest']:
        image = image.resize(shape,Image.NEAREST)
    elif mode in ['bilinear','Bilinear']:
        image = image.resize(shape,Image.BILINEAR)
    elif mode in ['lanczos','Lanczos']:
        image = image.resize(shape,Image.LANCZOS)

    if double_downsample:
        shape = (x//factor, y//factor)  #upsample by factor -> factor
        if mode in ['nearest','Nearest']:
            image = image.resize(shape,Image.NEAREST)
        elif mode in ['bilinear','Bilinear']:
            image = image.resize(shape,Image.BILINEAR)
        elif mode in ['lanczos','Lanczos']:
            image = image.resize(shape,Image.LANCZOS)

    image = np.array(image).astype(np.float32)
    # image = min_max_normalize(image)

    return image   

    
    
def downsample_kspace(hr_image, upscale_factor= 2, gaussian = False, sigma = 75, kspace_crop=False):

    F = np.fft.fft2(hr_image)
    fshift = np.fft.fftshift(F)
    
    y,x = fshift.shape

   ###### # data_pad = np.zeros((y,x),dtype=np.complex_)

    center_y = y//2 #defining the center of image in x and y direction
    center_x = x//2
    startx = center_x-(x//(upscale_factor*2))  
    starty = center_y-(y//(upscale_factor*2))
    
    arr = fshift[starty:starty+(y//upscale_factor),startx:startx+(x//upscale_factor)]
    
    img_reco_cropped = np.fft.ifft2(np.fft.ifftshift(arr)) 
    image = np.abs(img_reco_cropped )
    
    if gaussian:
        if kspace_crop:
            image = downsample_gaussian_with_sigma(image,sigma)
        else:
            # print("no cropping kspace")
            image = downsample_gaussian_with_sigma(hr_image,sigma)

    return image

def downsample_kspace_han_ham(hr_image, upscale_factor= 2, method = 'han'):

    F = np.fft.fft2(hr_image)
    fshift = np.fft.fftshift(F)
    
    y,x = fshift.shape

    # data_pad = np.zeros((y,x),dtype=np.complex_)

    center_y = y//2 #defining the center of image in x and y direction
    center_x = x//2
    startx = center_x-(x//(upscale_factor*2))  
    starty = center_y-(y//(upscale_factor*2))
    
    arr = fshift[starty:starty+(y//upscale_factor),startx:startx+(x//upscale_factor)]

    if method in ['han', 'hanning']:
        filter = get_hanning_filter(arr.shape,factor=upscale_factor)
    else:
        filter = get_hamming_filter(arr.shape,factor=upscale_factor)
    

    FFL = filter* arr  #multiplying with gaussian filter

    image = np.abs(np.fft.ifft2(np.fft.ifftshift(FFL))) 

    return image


def get_gaussian_low_pass_filter(shape, sigma = None):
        """Computes a gaussian low pass mask
        shape: the shape of the mask to be generated
        sigma: the cutoff frequency of the gaussian filter 
        factor: downsampling factor for given image
        returns a gaussian low pass mask"""
        rows, columns = shape
        d0 = sigma
        mask = np.zeros((rows, columns), dtype=np.complex_)
        mid_R, mid_C = int(rows / 2), int(columns / 2)
        for i in range(rows):
            for j in range(columns):
                d = math.sqrt((i - mid_R) ** 2 + (j - mid_C) ** 2)
                mask[i, j] = np.exp(-(d * d) / (2 * d0 * d0)) #dont divide by 2pi(sigma)^2 as it reduces the value of mask to the order of e-6
                
        final_mask = mask
        return final_mask


def apply_filter(image_arr,filter_apply):
    F = np.fft.fft2(image_arr)  #fourier transform of image
    fshift = np.fft.fftshift(F)  #shifting 

    FFL = filter_apply* fshift  #multiplying with gaussian filter

    img_recon = np.abs(np.fft.ifft2(np.fft.ifftshift(FFL))) #inverse shift and inverse fourier transform
    
    return img_recon
    
def downsample_gaussian_with_sigma(image_arr,sigma=150):
    low_filter = get_gaussian_low_pass_filter(image_arr.shape,sigma=sigma)
    image_downsampled = apply_filter(image_arr,low_filter)

    return image_downsampled



def get_hamming_filter(shape,factor=2):
    y,x = shape
    data_pad = np.zeros((y,x),dtype=np.complex_)
    mask_lower = np.zeros((y,x), dtype=np.float32)

    window1d = np.abs(np.hamming(y/factor))
    window1yd = np.abs(np.hamming(x/factor))
    window2d = np.sqrt(np.outer(window1d,window1yd))

    center_y = y//2 #defining the center of image in x and y direction
    center_x = x//2
    startx = center_x-(x//(factor*2))  
    starty = center_y-(y//(factor*2))

    mask_lower[starty:starty+(y//factor),startx:startx+(x//factor)] = window2d
    return mask_lower



def generate_hamming_blur(image_array, factor=2):
    hamming_window_filter = get_hamming_filter(image_array.shape,factor=factor)
    image_downsampled = apply_filter(image_array,hamming_window_filter)
    return image_downsampled


def downsample_bicubic_gaussian(hr_image,sigma=50):
    lr_image = downsample_bicubic(hr_image)
    lr_image = downsample_gaussian_with_sigma(lr_image,sigma)
    return lr_image


def get_hanning_filter(shape,factor=5):
    y,x = shape
    data_pad = np.zeros((y,x),dtype=np.complex_)
    mask_lower = np.zeros((y,x), dtype=np.float32)

    window1d = np.abs(np.hanning(y/factor))
    window1yd = np.abs(np.hanning(x/factor))
    window2d = np.sqrt(np.outer(window1d,window1yd))

    center_y = y//2 #defining the center of image in x and y direction
    center_x = x//2
    startx = center_x-(x//(factor*2))  
    starty = center_y-(y//(factor*2))

    mask_lower[starty:starty+(y//factor),startx:startx+(x//factor)] = window2d
    return mask_lower


def generate_mean_blur_images(image_array, kernel_size=3, upscale_factor=2):
    '''
    prepare mean blur
    '''
    # image_array = min_max_normalize(image_array)
    # image_array = image_array * 255.

    F = np.fft.fft2(image_array)
    fshift = np.fft.fftshift(F)
    
    y,x = fshift.shape

    # data_pad = np.zeros((y,x),dtype=np.complex_)

    center_y = y//2 #defining the center of image in x and y direction
    center_x = x//2
    startx = center_x-(x//(upscale_factor*2))  
    starty = center_y-(y//(upscale_factor*2))
    
    arr = fshift[starty:starty+(y//upscale_factor),startx:startx+(x//upscale_factor)]

    img_recon = np.abs(np.fft.ifft2(np.fft.ifftshift(arr)))

    ksize = (kernel_size, kernel_size)  
    mean_blur = cv2.blur(img_recon, ksize, cv2.BORDER_DEFAULT) 


    return mean_blur

#********************************************************************************************************************

def generate_median_blur_images(image_array, kernel_size=3, upscale_factor=2):
    '''
    prepare median blur
    '''
    # image_array = (image_array - image_array.min())/ (image_array.max()-image_array.min()) 
    # image_array = image_array * 255.

    F = np.fft.fft2(image_array)
    fshift = np.fft.fftshift(F)
    
    y,x = fshift.shape

    # data_pad = np.zeros((y,x),dtype=np.complex_)

    center_y = y//2 #defining the center of image in x and y direction
    center_x = x//2
    startx = center_x-(x//(upscale_factor*2))  
    starty = center_y-(y//(upscale_factor*2))
    
    arr = fshift[starty:starty+(y//upscale_factor),startx:startx+(x//upscale_factor)]

    img_recon = np.abs(np.fft.ifft2(np.fft.ifftshift(arr)))

    median_blur = cv2.medianBlur(img_recon.astype('float32'),kernel_size)

    return median_blur


def generate_hanning_blur(image_array, factor=2):
    hanning_window_filter = get_hanning_filter(image_array.shape,factor=factor)
    image_downsampled = apply_filter(image_array,hanning_window_filter)
    return image_downsampled



def prepare_lr_image(hr_image, degradation_method,upscale_factor):

    if degradation_method == 'bicubic':
        lr_image = downsample_bicubic(hr_image, upscale_factor)

    elif degradation_method == 'nearest':
        # print("inside nearest")
        lr_image = downsample_image_with_mode(array=hr_image,mode='nearest')

    elif degradation_method == 'bilinear':
        lr_image = downsample_image_with_mode(array=hr_image,mode='bilinear')

    elif degradation_method == 'lanczos':
        lr_image = downsample_image_with_mode(array=hr_image,mode='lanczos')

    elif degradation_method in ['crop_kspace','kspace']:
        lr_image = downsample_kspace(hr_image, upscale_factor)

    elif degradation_method in ['kspace_gaussian_50','kspace_gaussian_75','kspace_gaussian_25','kspace_gaussian_100','kspace_gaussian_125','kspace_gaussian_150']:# higher signma means less
        sigma = int(degradation_method.split('_')[2])
        lr_image = downsample_kspace(hr_image=hr_image, gaussian =True, sigma=sigma, kspace_crop=True)
 
    elif degradation_method == 'mean_blur':
        lr_image = generate_mean_blur_images(image_array=hr_image, kernel_size=3,upscale_factor=upscale_factor)

    elif degradation_method == 'median_blur':
        lr_image = generate_median_blur_images(image_array=hr_image, kernel_size=3, upscale_factor=upscale_factor)

    elif degradation_method in ['han','hann','hanning']:
        lr_image = downsample_kspace_han_ham(hr_image, upscale_factor= 2, method = 'han')

    elif degradation_method in ['ham','hamming','hamm']:
        lr_image = downsample_kspace_han_ham(hr_image, upscale_factor= 2, method = 'hamm')

    else:
        print("using default")
        lr_image = downsample_bicubic(hr_image)

    return lr_image
