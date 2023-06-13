import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
import os
import cv2
import copy
import torch.nn.functional as F
import colorsys
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import warnings
from data import data_utils
from torchvision.transforms import functional as TF
from torchvision import transforms
import json

warnings.filterwarnings("ignore")

# DATA_ROOT_SHOT = '/data/DevLearning/odme/data_1k/toys_rendering_output_trainshot'
# DATA_ROOT_QUERY = '/data/DevLearning/odme/data_1k/toys_rendering_output_testshot'
# DATA_ROOT_BASE = '/data/DevLearning/odme/data_1k/toys_rendering_output_base'

DATA_ROOT_PRED_SHOT = '/data/DevLearning/odme/FreeSOLO_pred/train_predictions'
DATA_ROOT_PRED_QUERY = '/data/DevLearning/odme/FreeSOLO_pred/test_predictions'

# DATA_ROOT_SHOT = '/data/DevLearning/odme/toys_rendering_output_single_trainshot'
# DATA_ROOT_QUERY = '/data/DevLearning/odme/toys_rendering_output_single_test'

# DATA_ROOT_SHOT = '/data/DevLearning/odme/toys_rendering_output_trainshot_one_obj'
# DATA_ROOT_QUERY = '/data/DevLearning/odme/toys_rendering_output_testshot_one_obj'

# DATA_ROOT_SHOT = '/data/DevLearning/odme/toys_rendering_output_trainshot_one_obj_pose_vary'
# DATA_ROOT_SHOT = '/home/ant/develop/FreeSOLO/datasets/coco/filter'

# DATA_ROOT_QUERY = '/data/DevLearning/odme/toys_rendering_output_testshot_one_obj_pose_vary'

DATA_ROOT_SHOT = '/data/DevLearning/odme/simple_data_toys/toys_rendering_output_trainshot_simple'
# DATA_ROOT_QUERY = '/data/DevLearning/odme/simple_data_toys/toys_rendering_output_testshot_simple'
DATA_ROOT_BASE = '/data/DevLearning/odme/simple_data_toys/toys_rendering_output_base_simple'

# DATA_ROOT_QUERY = '/home/ant/develop/FreeSOLO/datasets/coco/filter'
# DATA_ROOT_QUERY = '/data/DevLearning/odme/toys_rendering_output_testshot_one_obj_pose_vary'
# DATA_ROOT_QUERY = '/data/DevLearning/odme/simple_data_toys/toys_rendering_output_testshot_simple'

DATA_ROOT_QUERY = '/data/DevLearning/odme/CO3D_rearranged_test/'


LVIS = '/home/ant/develop/FreeSOLO/datasets/coco/annotations/lvis_train_overlap10k.json'

def get_co3d_test(npz_path):
    data = np.load(npz_path, allow_pickle=True)['dict'].item()
    return list(data.keys())


def get_all_dirs(root):
    return os.listdir(root)

def get_cat_lvis_ann():
    with open(LVIS, 'r') as f:
        data = json.load(f)

    categories = data['categories']
    categories = [c['name'] for c in categories]
    return categories

test_cats = get_co3d_test('/data/DevLearning/odme/CO3D_rearranged_test/cat_to_scenes.npz')
print('CO3D: ', test_cats)

# test_cats = get_cat_lvis_ann()
cat_to_num = {c: i for i, c in enumerate(test_cats)}

class LowShotDataset(Dataset):
    def __init__(self, n_shot, n_query, mode='global', permute=False, pred_seg=False, category=True, easy=False):

        self.n_shot = n_shot
        self.n_query = n_query
        self.mode = mode

        self.obj_scn_shot = np.load(os.path.join(DATA_ROOT_SHOT, 'cat_to_scenes.npz'), allow_pickle=True)['dict'].item()
        self.obj_scn_query = np.load(os.path.join(DATA_ROOT_QUERY, 'cat_to_scenes.npz'), allow_pickle=True)['dict'].item()
        self.all_cats = test_cats
        self.all_labels_shot = np.concatenate([[c for _ in self.obj_scn_shot[c]] for c in self.all_cats])
        self.scenes_shot = np.concatenate([self.obj_scn_shot[c] for c in self.all_cats])
        self.scenes_query = np.concatenate([self.obj_scn_query[c] for c in self.all_cats])

        # self.total_shots = len(self.all_labels_shot)
        self.total_shots = 0

        self.all_labels_query = np.concatenate([[c for _ in self.obj_scn_query[c]] for c in self.all_cats])

        # self.max_num_objs = 3
        self.max_num_objs = 1

        # self.labels = np.concatenate([self.all_labels_shot, self.all_labels_query])
        # self.data = np.concatenate([self.scenes_shot, self.scenes_query])
        self.labels = self.all_labels_query
        self.data = self.scenes_query

        self.permute = permute
        self.pred_seg = pred_seg

        self.category = category
        self.easy = easy



    def __len__(self):
        return len(self.data)


    def get_item_global(self, idx):
        cat = self.labels[idx]
        scene = self.data[idx]
        if idx >= self.total_shots:
            DATA_ROOT = DATA_ROOT_QUERY
            DATA_PRED_SEG = DATA_ROOT_PRED_QUERY
        else:
            DATA_ROOT = DATA_ROOT_SHOT
            DATA_PRED_SEG = DATA_ROOT_PRED_SHOT

        if idx < self.total_shots:
            obj = sorted([o[:-4] for o in os.listdir(os.path.join(DATA_ROOT, scene, "visibility")) if o.startswith(cat)])
        else:
            # obj = [o[:-4] for o in os.listdir(os.path.join(DATA_ROOT, "{:012d}".format(int(scene)), "RGB"))]
            obj = [scene]
        if len(obj) > 1:
            obj = np.random.choice(obj)
        else:
            obj = obj[0]


        all_imgs = []
        if idx < self.total_shots:
            view = get_random_view(DATA_ROOT, scene, obj, easy=self.easy)

            other_objs = [str(o[:-4]) for o in os.listdir(os.path.join(DATA_ROOT, scene, "visibility"))]
            ext = 'png'
        else:
            view = get_co3d_view(DATA_ROOT, scene)
            # view = 0
            other_objs = [obj]
            # other_objs = [o[:-4] for o in os.listdir(os.path.join(DATA_ROOT, "{:012d}".format(int(scene)), "RGB"))]
            ext = 'jpeg'
        for o in other_objs:
            if self.pred_seg:
                all_imgs.append(get_data_helper(DATA_ROOT, scene, o, view, data_root_pred_seg = DATA_PRED_SEG, pred_seg=True)[0])
            else:

                # all_imgs.append(get_data_helper(DATA_ROOT, scene, o, view, lvis=(idx>=self.total_shots))[0])
                all_imgs.append(get_data_helper(DATA_ROOT, scene, o, view, co3d=(idx>=self.total_shots), ext=ext)[0])


        all_imgs = torch.stack(all_imgs) ##### num_objects x C x H x W
        if len(all_imgs) < self.max_num_objs:
            all_imgs = all_imgs.repeat(self.max_num_objs,1,1,1)

        if self.category:
            if idx < self.total_shots:
                other_objs = np.array([cat_to_num.get(str(o[:-4]), -1) for o in other_objs])
            else:
                other_objs = np.array([cat_to_num.get(str(cat), -1) for o in other_objs])

        else:

            other_objs = np.array([cat_to_num.get(str(cat), -1) for o in other_objs])

        if len(other_objs) < self.max_num_objs:
            other_objs = np.repeat(other_objs, self.max_num_objs, axis=0)



        # other_objs = [str(o[:-4]) for o in other_objs]
        cat = cat_to_num[cat]
        # import pdb; pdb.set_trace()
        return all_imgs, cat, other_objs

    def get_item_local(self, idx):
        cat = self.labels[idx]
        scene = self.data[idx]
        if idx >= self.total_shots:
            DATA_ROOT = DATA_ROOT_QUERY
        else:
            DATA_ROOT = DATA_ROOT_SHOT
        obj = sorted([o[:-4] for o in os.listdir(os.path.join(DATA_ROOT, scene, "visibility")) if o.startswith(cat)])
        if len(obj) > 1:
            obj = np.random.choice(obj)
        else:
            obj = obj[0]

        view = get_random_view(DATA_ROOT, scene, obj, easy=self.easy)

        all_segs = []
        other_objs = [str(o[:-4]) for o in os.listdir(os.path.join(DATA_ROOT, scene, "visibility"))]

        for o in other_objs:
            other_imgs, other_segs = get_data_helper(DATA_ROOT, scene, o, view, g=False)
            all_segs.append(other_segs)
        if self.permute:
            rand_idx = np.random.permutation(len(all_segs))
            all_segs = list(np.array(all_segs)[rand_idx])
        all_segs = torch.stack(all_segs) ##### B x num_objects x C x H x W
        ##############
        if self.category:
            other_objs = np.array([cat_to_num.get(str(o[:-4]), -1) for o in other_objs])
        else:
            other_objs = np.array([cat_to_num.get(str(o), -1) for o in other_objs])

        cat = cat_to_num[cat]

        if len(all_segs) < self.max_num_objs:
            all_segs = torch.cat([all_segs, torch.tensor(np.zeros((self.max_num_objs-len(all_segs),112,112)), dtype=torch.float32)], dim=0)
        

        return other_imgs, all_segs, cat, other_objs



    def __getitem__(self, idx):
        if self.mode == 'global':
            return self.get_item_global(idx)
        return self.get_item_local(idx)


def get_co3d_view(data_root, scene):
    all_views = [int(o.split('.')[0]) for o in os.listdir(os.path.join(data_root, scene, "RGB"))]
    return np.random.choice(all_views)

def get_random_view(data_root, scene, obj, easy=False):
    all_views = np.arange(20)
    if not easy:
        not_visible_path = os.path.join(data_root, scene, "visibility", "{}.npz".format(obj))
        not_visible = np.load(not_visible_path)['not_visible']
        visible_views = [v for v in all_views if not v in not_visible]
    else:
        visible_path = os.path.join(data_root, scene, "all_visibility", "{}.npz".format(obj))
        visible_views = np.load(visible_path)['all_visible']
    view = np.random.choice(visible_views)
    return view


def mask_seg(seg, ratio=0., random=False):
    rmin1, rmax1, cmin1, cmax1 = data_utils.get_padded_bbx(seg, pad=0)

    seg = np.array(seg)
    r, c = np.where(seg[:,:,0] > 0)
    pixels = np.sum(seg[:,:,0])
    masked_pixels = int(pixels * ratio)
    if random:
        index = np.random.choice(len(r), masked_pixels, replace=False)
        seg[r[index], c[index], :] = 0

    else:
        l_square = int(np.round(np.sqrt(masked_pixels)))
        r_chosen = np.random.randint(rmin1, max(rmax1-l_square,rmin1+1))
        c_chosen = np.random.randint(cmin1, max(cmax1-l_square,cmin1+1))
        seg[r_chosen:min(224,r_chosen+l_square),c_chosen:min(224,c_chosen+l_square)] = 0


        ### choose start index


    seg = Image.fromarray(seg)
    return seg



def get_data_helper(data_root, scene, obj_inst, img_idx, \
        mask_size=224, pred_seg=False, data_root_pred_seg = '', \
        g=True, image_net_norm=True, lvis=False, co3d=False,ext='png'):
    if lvis:
        img_path = os.path.join(data_root, "{:012d}".format(int(scene)), "RGB", "{:012d}.jpg".format(int(obj_inst)))
        seg_path = os.path.join(data_root, "{:012d}".format(int(scene)), "segmentations", "{:012d}.jpg".format(int(obj_inst)))
    else:
        img_path = os.path.join(data_root, scene, "RGB", "{:04d}.{}".format(img_idx, ext))
        if not co3d:
            seg_path = os.path.join(data_root, scene, "segmentations", obj_inst, "{}_{:04d}.{}".format(obj_inst, img_idx, ext))
        else:
            seg_path = os.path.join(data_root, scene, "segmentations", "{:04d}.{}".format(img_idx, ext))

    with open(img_path, "rb") as f:
        img = Image.open(f).convert("RGB")

    img_size = img.size
    if img_size[0] != mask_size or img_size[1] != mask_size:
        img = img.resize((mask_size, mask_size))

    if not pred_seg:
        seg = data_utils.read_segmentation(seg_path)
    else:
        try:
            pred_seg_path = os.path.join(data_root_pred_seg, scene, "FreeSolo_masks", obj_inst, "{}_{:04d}.png".format(obj_inst, img_idx))
            seg = data_utils.read_segmentation(pred_seg_path)
        except:
            seg = np.zeros((224,224), dtype=np.uint8)

    seg = np.stack([seg, seg, seg], axis=-1)
    seg = Image.fromarray(seg)

    if seg.size[0] != mask_size or seg.size[1] != mask_size:
        seg = seg.resize((mask_size, mask_size))



    if g:
        ##########################
        seg = mask_seg(seg, ratio=0.7, random=True)
        ##########################
        try:

            # rmin1, rmax1, cmin1, cmax1 = data_utils.get_padded_bbx(seg, pad=20)
            # img = data_utils.center_crop(rmin1, rmax1, cmin1, cmax1, img, jitter=False)
            # seg = data_utils.center_crop(rmin1, rmax1, cmin1, cmax1, seg, jitter=False)


            img, seg = data_utils.mask_image(img, seg)
        except:
            img, seg = data_utils.mask_image(img, seg)


    if image_net_norm:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        img = prep(img)
    else:
        img = TF.to_tensor(img)

    # seg = torch.tensor(np.array(seg), dtype=torch.float32)[:,:,0] # make single channel
    seg = torch.tensor(cv2.resize(
        np.array(seg), (mask_size, mask_size), interpolation=cv2.INTER_AREA
    ), dtype=torch.float32)[:,:,0]

    # seg = TF.resize(
    #         seg.unsqueeze(0), 
    #         mask_size,
    #         interpolation=transforms.InterpolationMode.NEAREST).squeeze()

    return img, seg