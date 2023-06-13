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

DATA_ROOT_SHOT = '/data/DevLearning/odme/data_1k/toys_rendering_output_trainshot'
DATA_ROOT_QUERY = '/data/DevLearning/odme/data_1k/toys_rendering_output_testshot'
DATA_ROOT_BASE = '/data/DevLearning/odme/data_1k/toys_rendering_output_base'
# DATA_ROOT_BASE = '/data/DevLearning/odme/toys_rendering_output_base_4k'

DATA_ROOT_PRED_SHOT = '/data/DevLearning/odme/FreeSOLO_pred/train_predictions'
DATA_ROOT_PRED_QUERY = '/data/DevLearning/odme/FreeSOLO_pred/test_predictions'

# DATA_ROOT_PRED_SHOT = '/data/DevLearning/odme/FreeSOLO_pred/train_predictions_nft'
# DATA_ROOT_PRED_QUERY = '/data/DevLearning/odme/FreeSOLO_pred/test_predictions_nft'

# DATA_ROOT_PRED_SHOT = '/data/DevLearning/odme/FreeSOLO_shapenet/train_predictions'
# DATA_ROOT_PRED_QUERY = '/data/DevLearning/odme/FreeSOLO_shapenet/test_predictions'



# DATA_ROOT_SHOT = '/data/DevLearning/odme/toys_rendering_output_single_trainshot'
# DATA_ROOT_QUERY = '/data/DevLearning/odme/toys_rendering_output_single_test'

# DATA_ROOT_SHOT = '/data/DevLearning/odme/toys_rendering_output_trainshot_one_obj'
# DATA_ROOT_QUERY = '/data/DevLearning/odme/toys_rendering_output_testshot_one_obj'

# DATA_ROOT_SHOT = '/data/DevLearning/odme/toys_rendering_output_trainshot_one_obj_pose_vary'
# DATA_ROOT_QUERY = '/data/DevLearning/odme/toys_rendering_output_testshot_one_obj_pose_vary'

# DATA_ROOT_SHOT = '/data/DevLearning/odme/simple_data_toys/toys_rendering_output_trainshot_simple'
# DATA_ROOT_QUERY = '/data/DevLearning/odme/simple_data_toys/toys_rendering_output_testshot_simple'
# DATA_ROOT_BASE = '/data/DevLearning/odme/simple_data_toys/toys_rendering_output_base_simple'


# DATA_ROOT_SHOT = '/data/DevLearning/odme/shapenet_trainshot_1k/'
# DATA_ROOT_QUERY = '/data/DevLearning/odme/shapenet_testshot_1k/'
# DATA_ROOT_BASE = '/data/DevLearning/odme/shapenet_base/'

SHAPENET = False
MULTI_BASE = 0

def get_shapenet_mapper(cats):
    with open('/home/ant/develop/instance_category/source/data/lowshot/shapenetcorev2_synset.json', 'r') as f:
        data = json.load(f)
    invert_data = {q: k for k, q in data.items()}
    return [invert_data[k] for k in cats]


def get_all_dirs(root):
    return os.listdir(root)

def get_co3d_test(npz_path):
    data = np.load(npz_path, allow_pickle=True)['dict'].item()
    return list(data.keys())

base_cats = [
        'candy',
        'flower',
        'dragon',
        'apple',
        'guitar',
        'tree',
        'glass',
        'cup',
        'pig',
        'cat',
        'chair',
        'ice_cream',
        'hat',
        'deer_moose',
        'penguin',
        'ball',
        'fox',
        'dog',
        'knife',
        'laptop',
        'pen',
        'mug',
        'plate',
        'chess_piece',
        'cake',
        'frog',
        'ladder',
        'keyboard',
        'sofa',
        'trashcan',
        'dinosaur',
        'bottle',
        'elephant',
        'pencil',
        'key',
        'monitor',
        'hammer',
        'screwdriver',
        'robot',
        'bread'
        ]
# other_cats = [
#        'airplane',
#        'shark',
#        'truck',
#        'phone',
#        'giraffe',
#        'horse',
#        'fish',
#        'fan',
#        'shoe',
#        'snake'
#       ]
if SHAPENET: 
    base_cats = [
            'chair',
            'table',
            'bathtub',
            'cabinet',
            'lamp',
            'car',
            'bus',
            'cellphone',
            'guitar',
            'bench',
            'bottle',
            'laptop',
            'jar',
            'loudspeaker',
            'bookshelf',
            'faucet',
            'watercraft',
            'clock',
            'airplane',
            'flowerpot',
            'rifle',
            'display',
            'knife',
            'telephone',
            'sofa'
            ]

test_cats = [
        'boat',
        'lion',
        'whale',
        'cupcake',
        'train',
        'pizza',
        'marker',
        'cookie',
        'sandwich',
        'octopus',
        'monkey',
        'fries',
        'violin',
        'mushroom',
        'closet',
        'tractor',
        'submarine',
        'butterfly',
        'pear',
        'bicycle',
        'dolphin',
        'bunny',
        'coin',
        'radio',
        'grapes',
        'banana',
        'cow',
        'donut',
        'stove',
        'sink',
        'orange',
        'saw',
        'chicken',
        'hamburger',
        'piano',
        'light_bulb',
        'spade',
        'crab',
        'sheep',
        'toaster',
        'lizard',
        'motorcycle',
        'mouse',
        'pc_mouse',
        'bus',
        'helicopter',
        'microwave',
        'cells_battery',
        'drum',
        'panda',
        'tv',
        'car',
        'helmet',
        'fridge',
        'bowl'
       ]
if SHAPENET:
    test_cats = [
            'stove',
            'microwaves',
            'microphone',
            'cap',
            'dishwasher',
            'keyboard',
            'tower',
            'helmet',
            'birdhouse',
            'can',
            'piano',
            'train',
            'file cabinet',
            'pistol',
            'motorbike',
            'printer',
            'mug',
            'rocket',
            'skateboard',
            'bed',
            'trash bin',
            'washer',
            'bowl',
            'bag',
            'mailbox',
            'pillow',
            'earphone',
            'camera',
            'basket',
            'remote'
        ]

# test_cats = get_all_dirs('/home/ant/develop/CRIBpp_generic/common/toys200_poses_canonical')
# test_cats = get_co3d_test('/data/DevLearning/odme/CO3D_rearranged_test/cat_to_scenes.npz')
if SHAPENET:
    test_cats = get_shapenet_mapper(test_cats)
    base_cats = get_shapenet_mapper(base_cats)
print(test_cats)
cat_to_num = {c: i for i, c in enumerate(test_cats)}




class BaseDataset(Dataset):
    def __init__(self, mode='global', permute = False):

        self.data_root = DATA_ROOT_BASE
        self.mode = mode

        self.obj_scn = np.load(os.path.join(self.data_root, 'cat_to_scenes.npz'), allow_pickle=True)['dict'].item()

        if SHAPENET:
            self.all_cats = list(self.obj_scn.keys())
        else:
            self.all_cats = base_cats
        self.all_labels = np.concatenate([[c for _ in self.obj_scn[c]] for c in self.all_cats])
        self.all_data = np.concatenate([self.obj_scn[c] for c in self.all_cats])

        if MULTI_BASE > 0:
            self.all_labels = np.concatenate([self.all_labels for _ in range(MULTI_BASE)])
            self.all_data = np.concatenate([self.all_data for _ in range(MULTI_BASE)])


        self.permute = permute

    def __len__(self):
        return len(self.all_data)

    def get_item_global(self, idx):
        scene = self.all_data[idx]
        cat = self.all_labels[idx]
        obj = [o[:-4] for o in os.listdir(os.path.join(self.data_root, scene, "visibility")) if o.startswith(cat)]

        if len(obj) > 1:
            obj = np.random.choice(obj)
        else:
            obj = obj[0]

        # view = idx // (self.__len__()//20)

        all_views = np.arange(20)
        not_visible_path = os.path.join(self.data_root, scene, "visibility", "{}.npz".format(obj))
        not_visible = np.load(not_visible_path)['not_visible']
        visible_views = [v for v in all_views if not v in not_visible]
        view = np.random.choice(visible_views)

        img, seg = get_data_helper(self.data_root, scene, obj, view)

        return img, cat

    def get_item_local(self, idx):
        idx = 0
        scene = self.all_data[idx]
        cat = self.all_labels[idx]
        obj = sorted([o[:-4] for o in os.listdir(os.path.join(self.data_root, scene, "visibility")) if o.startswith(cat)])
        if len(obj) > 1:
            obj = np.random.choice(obj)
        else:
            obj = obj[0]

        view = get_random_view(self.data_root, scene, obj)

        all_segs = []
        other_objs = [str(o[:-4]) for o in os.listdir(os.path.join(self.data_root, scene, "visibility"))]

        for o in other_objs:
            other_imgs, other_segs = get_data_helper(self.data_root, scene, o, view, g=False)
            all_segs.append(other_segs)
        if self.permute:
            rand_idx = np.random.permutation(len(all_segs))
            all_segs = list(np.array(all_segs)[rand_idx])
        all_segs = torch.stack(all_segs) ##### B x num_objects x C x H x W


        if len(all_segs) < 3:
            all_segs = torch.cat([all_segs, torch.tensor(np.zeros((3-len(all_segs),28,28)), dtype=torch.float32)], dim=0)
        

        return other_imgs, all_segs, cat


    def __getitem__(self, idx):
        if self.mode == 'global':
            return self.get_item_global(idx)
        return self.get_item_local(idx)


    

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

        self.total_shots = len(self.all_labels_shot)
        self.all_labels_query = np.concatenate([[c for _ in self.obj_scn_query[c]] for c in self.all_cats])

        self.max_num_objs = 1
        self.labels = np.concatenate([self.all_labels_shot, self.all_labels_query])
        self.data = np.concatenate([self.scenes_shot, self.scenes_query])

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
        obj = sorted([o[:-4] for o in os.listdir(os.path.join(DATA_ROOT, scene, "visibility")) if o.startswith(cat)])
        
        if len(obj) > 1:
            obj = np.random.choice(obj)
        else:
            obj = obj[0]


        view = get_random_view(DATA_ROOT, scene, obj, easy=self.easy)

        all_imgs = []
        other_objs = [str(o[:-4]) for o in os.listdir(os.path.join(DATA_ROOT, scene, "visibility"))]


        for o in other_objs:
            if self.pred_seg:
                all_imgs.append(get_data_helper(DATA_ROOT, scene, o, view, data_root_pred_seg = DATA_PRED_SEG, pred_seg=True)[0])
            else:
                all_imgs.append(get_data_helper(DATA_ROOT, scene, o, view)[0])

        all_imgs = torch.stack(all_imgs) ##### B x num_objects x C x H x W
        if len(all_imgs) < self.max_num_objs:
            all_imgs = all_imgs.repeat(self.max_num_objs,1,1,1)

        if self.category:
            if not SHAPENET:
                other_objs = np.array([cat_to_num.get(str(o[:-4]), -1) for o in other_objs])
            else:
                other_objs = np.array([cat_to_num.get(str(o.split('_')[0]), -1) for o in other_objs])

        else:
            other_objs = np.array([cat_to_num.get(str(o), -1) for o in other_objs])
        # import pdb; pdb.set_trace()
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


def get_data_helper(data_root, scene, obj_inst, img_idx, mask_size=224, pred_seg=False, data_root_pred_seg = '', g=True, image_net_norm=True):
    img_path = os.path.join(data_root, scene, "RGB", "{:04d}.png".format(img_idx))
    seg_path = os.path.join(data_root, scene, "segmentations", obj_inst, "{}_{:04d}.png".format(obj_inst, img_idx))
    if SHAPENET:
        try:
            seg_path = os.path.join(data_root, scene, "segmentations", obj_inst, "{}_{:04d}.png".format(obj_inst.split('_')[1], img_idx))
        except:
            print(seg_path)


    with open(img_path, "rb") as f:
        img = Image.open(f).convert("RGB")

    img_size = img.size

    if not pred_seg:
        seg = data_utils.read_segmentation(seg_path)
    else:
        try:
            pred_seg_path = os.path.join(data_root_pred_seg, scene, "FreeSolo_masks", obj_inst, "{}_{:04d}.png".format(obj_inst, img_idx))
            if SHAPENET:
                pred_seg_path = os.path.join(data_root_pred_seg, scene, "FreeSolo_masks", obj_inst, "{}_{:04d}.png".format(obj_inst.split('_')[1], img_idx))

            seg = data_utils.read_segmentation(pred_seg_path)
        except:
            seg = np.zeros((224,224), dtype=np.uint8)

    seg = np.stack([seg, seg, seg], axis=-1)
    seg = Image.fromarray(seg)



    if g:
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