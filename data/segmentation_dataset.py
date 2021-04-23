  
"""Dataset class template
This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
import os
from data.image_folder import make_dataset, make_dataset_from_img_list
# from data.image_folder import make_dataset
from PIL import Image


class SegmentationDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.path_to_imgs_file_list  = os.path.join(opt.dataroot, 'ImageSets', opt.phase + '.txt')
        self.images_path, self.masks_path = make_dataset_from_img_list(self.path_to_imgs_file_list, opt.dataroot, opt.max_dataset_size) 
        self.images_size =  len(self.images_path)
        self.masks_size =  len(self.masks_path)
        # Any transformation
        self.transformImg = get_transform(self.opt, normalize=False, convert=True) # Precise no flip # no normalize
        self.transformMask = get_transform(self.opt, normalize=False, convert=True) # Precise no flip  

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index -- a random integer for data indexing
        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        
        Imgpath = self.images_path[index]  # make sure index is within then range
        Maskpath =  self.masks_path[index]
        Img_i = Image.open(Imgpath).convert('RGB')
        Mask_i = Image.open(Maskpath).convert('RGB')
        Mask_i = Mask_i.getchannel('R')
        # further downsample seg label, need to avoid seg label misalignment
        segm_rounded_width = self.round2nearest_multiple(Mask_i.size[0], self.segm_downsampling_rate)
        segm_rounded_height = self.round2nearest_multiple(Mask_i.size[1], self.segm_downsampling_rate)
        segm_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
        segm_rounded.paste(Mask_i, (0, 0))
        segm = self.imresize(
            segm_rounded,
            (segm_rounded.size[0] // self.segm_downsampling_rate, \
             segm_rounded.size[1] // self.segm_downsampling_rate), \
            interp='nearest')
        
        Img = self.transformImg(Img_i)
        Mask = self.transformMask(segm)
        #######################
        # MAsk to long tensor
        #######################
        Mask = Mask.long() 
        return {'Img': Img, 'Mask': Mask, 'Imgpath': Imgpath, 'Maskpath': Maskpath}


    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p
    
    def imresize(self, im, size, interp='bilinear'):
        if interp == 'nearest':
            resample = Image.NEAREST
        elif interp == 'bilinear':
            resample = Image.BILINEAR
        elif interp == 'bicubic':
            resample = Image.BICUBIC
        else:
            raise Exception('resample method undefined!')

        return im.resize(size, resample)
    def __len__(self):
        """Return the total number of images."""
        return max(self.images_size, self.masks_size)