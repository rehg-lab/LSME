import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib

from prettytable import PrettyTable

matplotlib.use("Agg")


class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

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


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def compute_similarity_global(q, k, k_neg):
    if len(q.shape) == 2:
        pos_idx = torch.eye(q.shape[0], q.shape[0])

        q_temp = q.reshape(-1,3,q.shape[-1])
        block_diag = torch.block_diag(*([torch.ones(3, 3)]*q_temp.shape[0]))
        same_scene_neg_idx = block_diag - pos_idx
        same_scene_neg_idx = same_scene_neg_idx.nonzero()[:,1].reshape(-1, 2).cuda()

        neg_idx = torch.ones(q.shape[0], q.shape[0]) - block_diag
        pos_idx = pos_idx.nonzero()[:,1].reshape(-1,1).cuda()
        neg_idx = neg_idx.nonzero()[:,1].reshape(-1, q.shape[0]-3).cuda()
        object_dot = torch.mm(q,k.transpose(1,0))
        sim = object_dot.gather(1, pos_idx).mean()
        not_sim_diff_scene = object_dot.gather(1, neg_idx).mean()
        not_sim_same_scene = object_dot.gather(1, same_scene_neg_idx).mean()
    elif len(q.shape) == 3:
        object_dot = torch.matmul(q.unsqueeze(1), k.transpose(2,1))
        B1, B2, _, _ = object_dot.shape
        object_dot = torch.sum(object_dot.reshape(B1, B2, -1), dim=2)
    return sim, not_sim_diff_scene, not_sim_same_scene

def compute_similarity_global_single(q, k, k_neg):
    perm = torch.roll(torch.arange(len(q)), 1)

    global_sim_dot = torch.bmm(q.unsqueeze(1), k.unsqueeze(2))
    global_sim_dot = global_sim_dot.mean().cpu().detach().numpy()

    if k_neg is not None:

        global_same_scene_not_sim = torch.bmm(q.unsqueeze(1), k_neg.unsqueeze(2))
        global_same_scene_not_sim = global_same_scene_not_sim.mean().cpu().detach().numpy()
    else:
        global_same_scene_not_sim = 0

    global_not_sim_dot = torch.bmm(q[perm].unsqueeze(1), k.unsqueeze(2))
    global_not_sim_dot = global_not_sim_dot.mean().cpu().detach().numpy()

    return global_sim_dot, global_not_sim_dot, global_same_scene_not_sim

def compute_similarity_object(q, k):
    if len(q.shape) == 2:
        object_dot = torch.mm(q,k.transpose(1,0))
    elif len(q.shape) == 3:
        object_dot = torch.matmul(q.unsqueeze(1), k.transpose(2,1))
        B1, B2, _, _ = object_dot.shape
        object_dot = torch.sum(object_dot.reshape(B1, B2, -1), dim=2)
    return object_dot

def show_mask(mask, ax=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.reshape(h,w,1)
    mask_image = mask * color.reshape(1, 1, -1)
    return mask_image[:,:,:3]
    # ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

def get_box(seg):
    ########## Seg returned from read_segmentation
    img_size = seg.shape

    seg1 = np.sum(seg, axis=0)
    seg2 = np.sum(seg, axis=1)

    cmin = np.where(seg1 > 0)[0][0]
    rmin = np.where(seg2 > 0)[0][0]

    cmax = np.where(seg1 > 0)[0][-1]
    rmax = np.where(seg2 > 0)[0][-1]

    return rmin, rmax, cmin, cmax



def compute_ME_acc(pred, gt, available):
    available = [a for a in available if a != -1]
    available = np.unique(available)
    chosen = np.array([ind for ind in range(len(gt)) if gt[ind] in available])
    pred = pred[chosen]
    gt = gt[chosen]
    return np.sum(pred == gt)/len(gt)

def maskpool(masked_descriptor, seg):
    B, C, H, W = masked_descriptor.shape
    pixels = torch.sum(seg.reshape(B,-1), dim=-1)
    masked_descriptor = masked_descriptor.reshape(B, C, -1)
    masked_descriptor = torch.sum(masked_descriptor,dim=2) / pixels.unsqueeze(1).cuda()
    return masked_descriptor


def shot_assignment_accuracy(gt_labels, inst_pred, N, n_shot, n_way):
    gt_labels = gt_labels.reshape(n_shot*n_way, N)
    inst_pred = inst_pred.cpu().numpy()
    preds = gt_labels[np.arange(n_shot*n_way), inst_pred]
    return np.sum(preds != -1)/len(preds)

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def pad_segs(gt_seg, num_queries):
    B, N, H, W = gt_seg.shape
    if N < num_queries:
        padded_segs = torch.cat([gt_seg, torch.zeros(B,num_queries-N,H,W).cuda()], dim=1)
    else:
        padded_segs = gt_seg
    return padded_segs
