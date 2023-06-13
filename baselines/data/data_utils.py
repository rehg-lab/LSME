from random import shuffle
import torch
import os
import numpy as np
import cv2

from torchvision import transforms
from PIL import Image


class TransformWGeometricContrastive(object):
    def __init__(self):

        self.brightness_min = 0.4
        self.brightness_max = 1.6

        self.contrast_min = 0.4
        self.contrast_max = 1.6

        self.saturation_min = 0
        self.saturation_max = 1

        self.hue_min = -0.5
        self.hue_max = 0.5

        self.kernels = [3,5]

        self.sigma_min = 0.1
        self.sigma_max = 0.2

        self.gamma_min = 0.25
        self.gamma_max = 1.75

        self.totensor = transforms.ToTensor()

    def apply_hflip(self, img):
        img = transforms.functional.hflip(img)
        return img

    def apply_blur(self, img):
        kernel_size = int(np.random.choice(self.kernels))
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = transforms.functional.gaussian_blur(img, kernel_size, sigma)

        return img

    def apply_color_jitter(self, img):

        if np.random.uniform(0,1)<0.75:
            brightness = np.random.uniform(self.brightness_min, self.brightness_max)
            img = transforms.functional.adjust_brightness(img, brightness)

        if np.random.uniform(0,1)<0.9:
            hue = np.random.uniform(self.hue_min, self.hue_max)
            saturation = np.random.uniform(1.5,2)
            img = transforms.functional.adjust_saturation(img, saturation)
            img = transforms.functional.adjust_hue(img, hue)

        if np.random.uniform(0,1)<0.75:
            gamma = np.random.uniform(self.gamma_min, self.gamma_max)
            img = transforms.functional.adjust_gamma(img, gamma)

        if np.random.choice([0,1]):
            contrast = np.random.uniform(self.contrast_min, self.contrast_max)
            img = transforms.functional.adjust_contrast(img, contrast)

        if np.random.choice([0,1]):
            saturation = np.random.uniform(self.saturation_min, self.saturation_max)
            img = transforms.functional.adjust_saturation(img, saturation)

        return img
    
    def _augment_img_only(self, img):
        if np.random.uniform(0,1)<0.9:
            img = self.apply_color_jitter(img)

        if np.random.choice([0,1]):
            img = self.apply_blur(img)

        if np.random.choice([0,1]):
            img = self.apply_hflip(img)

        img = self.totensor(img)
        return img
    
    def _augment_img_with_correspondences(self, img, uv_fg, uv_bg, seg):
        W, H = img.size
        n_pts_fg, _ = uv_fg.shape
        n_pts_bg, _ = uv_bg.shape

        if np.random.uniform(0,1)<0.9:
            img = self.apply_color_jitter(img)

        if np.random.choice([0,1]):
            img = self.apply_blur(img)

        if np.random.choice([0,1]):

            mask_fg = torch.zeros(H,W).int()
            mask_fg[
                    uv_fg[torch.arange(n_pts_fg),0], uv_fg[torch.arange(n_pts_fg),1]
                ] = (torch.arange(n_pts_fg)+1).int()

            mask_bg = torch.zeros(H,W).int()
            mask_bg[
                    uv_bg[torch.arange(n_pts_bg),0], uv_bg[torch.arange(n_pts_bg),1]
                ] = (torch.arange(n_pts_bg)+1).int()

            img = self.apply_hflip(img)
            seg = self.apply_hflip(seg)
            mask_fg = self.apply_hflip(mask_fg)
            mask_bg = self.apply_hflip(mask_bg)

            uv_fg = [torch.cat(torch.where(mask_fg==i)) for i in (torch.arange(n_pts_fg)+1).int()]
            uv_fg = [x if len(x)>0 else torch.zeros(2).long() for x in uv_fg]
            uv_fg = torch.stack(uv_fg)

            uv_bg = [torch.cat(torch.where(mask_bg==i)) for i in (torch.arange(n_pts_bg)+1).int()]
            uv_bg = [x if len(x)>0 else torch.zeros(2).long() for x in uv_bg]
            uv_bg = torch.stack(uv_bg)

        img = self.totensor(img)
        return img, uv_fg, uv_bg, seg


    def __call__(self, *args, **kw):
            
        if len(kw) > 0:
            img = args[0]
            uv_fg = kw['uv_fg']
            uv_bg = kw['uv_bg']
            seg = kw['seg']

            return self._augment_img_with_correspondences(img, uv_fg, uv_bg, seg)
        else:
            img = args[0]
            return self._augment_img_only(img)


class TransformWGeometricLowShot(object):
    def __init__(self):
        self.color_jitter = transforms.ColorJitter(brightness=(0.65,1.0),
                                        contrast=(0.7,1),
                                        saturation=(0,1))

        self.random_affine = transforms.RandomAffine(degrees=(-15.0,15.0),
                                         translate=(0.10,0.10),
                                         scale=(0.9, 1.1),
                                         shear=None,
                                         interpolation=transforms.InterpolationMode.BILINEAR,
                                         fill=(0,0,0))

        self.random_flip = transforms.RandomHorizontalFlip(p=0.5)
        self.totensor = transforms.ToTensor()
        self.resized_crop = transforms.RandomResizedCrop(224, scale=(0.8, 1.))

    def apply_random_affine(self, img):
        t_img = self.random_affine(img)
        t = self.totensor(t_img)
        t_img = transforms.ToPILImage(mode='RGB')(t)

        return t_img

    def __call__(self, img):

        if np.random.choice([0,1]):
            img = self.resized_crop(img)

        if np.random.choice([0,1]):
            img = self.random_flip(img)

        if np.random.choice([0,1]):
            img = self.apply_random_affine(img)

        if np.random.choice([0,1]):
            img = self.color_jitter(img)


        return img

def read_segmentation(fpath):

    seg = cv2.imread(fpath)[:, :, 0]
    seg[seg <= 50] = 0.0
    seg[seg >= 50] = 1.0

    return seg


def get_all_scenes_with_instance(data_path, scene_list, inst, subset=None):
    all_scenes = []
    if subset is not None:
        scene_list = np.random.choice(scene_list, subset, replace=False)
    for scene in scene_list:
        scene_path = os.path.join(data_path, scene)
        all_instances = [i[:-9] for i in os.listdir(scene_path) if i.endswith('.npz')]
        if inst in all_instances:
            inst_data = np.load(os.path.join(scene_path, inst+'_corr.npz'), allow_pickle=True)
            corr = inst_data["corr"]
            if len(corr) > 0:
                all_scenes.append(scene)
    return all_scenes

def get_padded_bbx(seg, pad=10):
    ########## Seg returned from read_segmentation
    img_size = seg.size

    seg1 = np.sum(seg, axis=0)
    seg2 = np.sum(seg, axis=1)

    cmin = np.where(seg1 > 0)[0][0]
    rmin = np.where(seg2 > 0)[0][0]

    cmax = np.where(seg1 > 0)[0][-1]
    rmax = np.where(seg2 > 0)[0][-1]

    rmin = max(rmin-pad, 0)
    rmax = min(rmax+pad, img_size[0])

    cmin = max(cmin-pad, 0)
    cmax = min(cmax+pad, img_size[1])

    return rmin, rmax, cmin, cmax

def center_crop(rmin, rmax, cmin, cmax, img, jitter=True):
    H, W = img.size
    if jitter: 
        shiftr = np.clip(np.random.randint(-10,10), -rmin, H-rmax)
        shiftc = np.clip(np.random.randint(-10,10), -cmin, W-rmax)
    else:
        shiftr = 0
        shiftc = 0

    img = np.array(img)[rmin+shiftr:rmax+shiftr, cmin+shiftc:cmax+shiftc, :]
    img = Image.fromarray(img)
    img = img.resize((H,W), resample=Image.BICUBIC)

    return img

def mask_image(image, mask):
    image = np.asarray(image)
    mask = np.asarray(mask)

    image = image * mask
    image = Image.fromarray(image)
    mask = Image.fromarray(mask)
    return image, mask





