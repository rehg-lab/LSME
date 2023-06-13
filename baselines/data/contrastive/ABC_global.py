import numpy as np
import torch
import os
import json
import itertools

from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset

from data import data_utils
from data.augmentations import ContrastiveAugmentation

from PIL import Image
from pprint import pprint


# DATA_ROOT = "./dataset_directory/abc"
# DATA_ROOT = "/home/ant/data/odme/ABC_rendering_output_easier"

DATA_ROOT = "/data/DevLearning/odme/toys_rendering_output_base_4k"


# DATA_ROOT = "/data/DevLearning/odme/ABC_rendering_output_JPEG"
# DATA_ROOT = "/data/DevLearning/odme/data_1k/toys_rendering_output_base"




class ABC(Dataset):
    def __init__(self, split=None, augmentation_file=None, subset=None):

        if split != '':

            split_path = os.path.join(DATA_ROOT, "split", "split2.json")

            try:

                with open(split_path, 'r') as f:
                    data = json.load(f)
                    scenes = data[split]
            except Exception:
                print('No split file')
                split = ''
                scenes = [f for f in os.listdir(DATA_ROOT) \
                    if not f.endswith('npy') and not f.endswith('npz') and f != 'split']
        else:
            scenes = [f for f in os.listdir(DATA_ROOT) \
                if not f.endswith('npy') and not f.endswith('npz') and f != 'split']

        if subset is None:
            self.scenes = scenes
        else:
            self.scenes = scenes[:subset]

        self.split = split
        self.mask_size = 56
        self.transform = None

        if split != '':
            self.obj_scn = np.load(os.path.join(DATA_ROOT, 'instance_to_scenes_%s.npz'%(split)), allow_pickle=True)['dict'].item()
        else:
            self.obj_scn = np.load(os.path.join(DATA_ROOT, 'instance_to_scenes.npz'), allow_pickle=True)['dict'].item()


        if augmentation_file is not None and augmentation_file != "none":
            with open(f"./data/augmentation_parameters/{augmentation_file}", "r") as f:
                aug_params = json.load(f)
            pprint(aug_params)
            self.transform = ContrastiveAugmentation(aug_params)

        self.mask_size = 56
        self.random_masking = False


        print("Total {} samples: {}".format(split, len(self.scenes)))

    def __len__(self):
        return len(self.scenes)

    def get_info_positive_different_scene(self, obj_inst, scene, thres=1.):
        all_scenes = self.obj_scn[obj_inst]
        ran = np.random.rand()
        if ran < thres:
            remaining_scenes = [s for s in all_scenes if s != scene]
            if len(remaining_scenes) == 0:
                scene2 = scene
            else:
                scene2 = np.random.choice(remaining_scenes)
        else:
            scene2 = np.random.choice(all_scenes)
        return scene2

    def get_info_neg_different_scene(self, obj_inst, scene):
        all_scenes = [s for s in self.scenes if s not in self.obj_scn[obj_inst]]
        scene2 = np.random.choice(all_scenes)

        all_objs = os.listdir(os.path.join(DATA_ROOT, scene2, "visibility"))
        obj2 = np.random.choice(all_objs)
        return scene2, obj2[:-4]

    def get_views_obj(self, obj_inst, scene):
        all_views = np.arange(20)
        not_visible_path = os.path.join(DATA_ROOT, scene, "visibility", "{}.npz".format(obj_inst))
        not_visible = np.load(not_visible_path)['not_visible']
        visible_views = [v for v in all_views if not v in not_visible]
        views = np.random.choice(visible_views, 2, replace=False)
        return obj_inst, scene, views
    
    def __getitem__(self, idx):

        scene = self.scenes[idx]

        obj_path = os.path.join(DATA_ROOT, scene)
        all_objs = os.listdir(os.path.join(DATA_ROOT, scene, "visibility"))
        try:
            obj = np.random.choice(all_objs)[:-4]
        except:
            print(scene)

        _, _, views = self.get_views_obj(obj, scene)
        
        img1, seg1 = self.get_data_helper(scene, obj, views[0])

        ########################
        ran = np.random.rand()
        if ran < 0:
            scene2 = self.get_info_positive_different_scene(obj, scene)
        else:
            scene2 = scene

        #######################
        if scene2 == scene:
            img2, seg2 = self.get_data_helper(scene, obj, views[1])
        else:
            _, _, views_pos = self.get_views_obj(obj, scene2)
            img2, seg2 = self.get_data_helper(scene2, obj, views_pos[0])

        try:

            remaining_objs = [o for o in all_objs if o[:-4] != obj]
            obj_neg = np.random.choice(remaining_objs)[:-4]
            _, _, view_negs = self.get_views_obj(obj_neg, scene)
            img_neg, seg_neg = self.get_data_helper(scene, obj_neg, view_negs[0])
        except:
            neg_scene, obj_neg = self.get_info_neg_different_scene(obj, scene)
            _, _, view_negs = self.get_views_obj(obj_neg, neg_scene)
            img_neg, seg_neg = self.get_data_helper(neg_scene, obj_neg, view_negs[0])



        out_dct = dict(
                image1=img1,
                seg1=seg1,
                image2=img2,
                seg2=seg2,
                scene=scene,
                instance=obj,
                image_neg=img_neg,
                seg_neg=seg_neg,
                obj_neg=obj_neg
            )

        return out_dct

    def get_scene_segs(self, scene, all_objs, view):
        segs = []
        for o in all_objs:
            segs.append(self.get_data_helper(scene, o, view)[1])
        segs = torch.stack(segs)
        return segs
    
    def mask_seg(self, seg, ratio=0., random=True):
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


    def get_data_helper(self, scene, obj_inst, img_idx):
        img_path = os.path.join(DATA_ROOT, scene, "RGB", "{:04d}.png".format(img_idx))
        seg_path = os.path.join(DATA_ROOT, scene, "segmentations", obj_inst, "{}_{:04d}.png".format(obj_inst, img_idx))

        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        seg = data_utils.read_segmentation(seg_path)

        seg = np.stack([seg, seg, seg], axis=-1)
        seg = Image.fromarray(seg)

        ############
        # seg = self.mask_seg(seg, ratio=0.3)
        ############


        if self.transform is not None:
            choice_list = [True, False]
            bool1 = choice_list[np.random.randint(2)]
            # bool2 = choice_list[np.random.randint(2)]

            # bool1 = True #### Always do random masking

            img, seg, _ = self.transform(img, seg, torch.zeros(10,2), bool1)

            ##### Center crop if not masked
            if not bool1:
                # rmin, rmax, cmin, cmax = data_utils.get_padded_bbx(seg, pad=np.random.randint(20,30))
                # img = data_utils.center_crop(rmin, rmax, cmin, cmax, img, jitter=False)
                # seg = data_utils.center_crop(rmin, rmax, cmin, cmax, seg, jitter=False)

                img, seg = data_utils.mask_image(img, seg)


        if self.transform is None:
            #######################################
            # rmin, rmax, cmin, cmax = data_utils.get_padded_bbx(seg, pad=np.random.randint(20,30))

            # img = data_utils.center_crop(rmin, rmax, cmin, cmax, img, jitter=False)
            # seg = data_utils.center_crop(rmin, rmax, cmin, cmax, seg, jitter=False)
            #######################################

            img, seg = data_utils.mask_image(img, seg)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        img = prep(img)
        # img = TF.to_tensor(img)
        seg = torch.tensor(np.array(seg), dtype=torch.float32)[:,:,0] # make single channel

        seg = TF.resize(
                seg.unsqueeze(0), 
                self.mask_size,
                interpolation=transforms.InterpolationMode.NEAREST).squeeze()

        return img, seg



if __name__ == "__main__":
    SEED = 1234
    import random
    import matplotlib.pyplot as plt
    import matplotlib

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    seen_train = ABC(split="train", augmentation_file="local_contrastive_for_tuning.json")

    ''' 
    for i in range(len(seen_train)):
        print(i, end='\r')
        try:
            data = seen_train.__getitem__(i)
        except:
            print(seen_train.objects[i])
    train_loader = torch.utils.data.DataLoader(
            seen_train,
            batch_size=512,
            num_workers=10,
            drop_last = False
        )
    i=0 
    for batch in train_loader:
        print(i)
        i+=1

    print("mean angular distance {:.2f} degrees".format(np.mean(arc_ls)))
    ''' 

    ## viz dataloading  
    # out_dir = "data/test_outputs/ABC_global_dataset_test_output"

    out_dir = "/data/DevLearning/odme/test_viz_scene_global"


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in range(40):
        data = seen_train.__getitem__(i) 
        image1, image2 = data['image1'], data['image2']
        
        name = data["instance"]

        image1 = image1.numpy().transpose(1,2,0)
        image2 = image2.numpy().transpose(1,2,0)

        fig = matplotlib.figure.Figure()
        (ax1, ax2) = fig.subplots(1,2)
        ax1.axis("off")
        ax2.axis("off")
        
        ax1.imshow(image1)
        ax2.imshow(image2)
        
        fig.suptitle("{}".format(name))

        fig.tight_layout()
        fig.savefig(
                os.path.join(out_dir, "{:04d}.png".format(i))
            )


