"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import pandas as pd

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def make_dataset_from_img_list(path_to_files_imgs_list, dataroot, max_dataset_size=float("inf")):
    images = []
    masks = [] 
    df  = pd.read_table(path_to_files_imgs_list, header = None)
    list_picts = df.iloc[:,0]
#     print('****************************************** EMILIE ******************************* \n\n\n'
#        ' list_picts ', list_picts  )
    path_to_img_folder = os.path.join(dataroot, 'JPEGImages')
    path_to_mask_folder = os.path.join(dataroot, 'SegmentationClassRaw')
    for pict in list_picts:
        Imgname = pict+'.jpg'
        Maskname =  pict+'.png'
        full_img_path =  os.path.join(path_to_img_folder, Imgname)
        full_mask_path =  os.path.join(path_to_mask_folder, Maskname)
#         print('full_img_path ', full_img_path)
        if Imgname in os.listdir(path_to_img_folder) and Maskname in os.listdir(path_to_mask_folder):
            masks.append(full_mask_path)
            images.append(full_img_path)
        else:
            print('Picture ', Imgname, 'or', Maskname,  'not found in ', path_to_img_folder, 'or in the folder', path_to_mask_folder)
            return -1 
#     print('\n\n images[:min(max_dataset_size, len(images))] ', images[:min(max_dataset_size, len(images))], '\n\n\n\n')
    return images[:min(max_dataset_size, len(images))], masks[:min(max_dataset_size, len(images))]
    
def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
