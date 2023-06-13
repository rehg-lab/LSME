# From Yonglong Tian

import torch
import torch.nn as nn
import math

from nets.moco_func_utils import concat_all_gather


class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""

    def __init__(self, inputSize, K, B_size=128, T=0.07, use_queue=True):
        super(MemoryMoCo, self).__init__()

        self.use_queue = use_queue
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.B_size = B_size
        self.pos_idx = torch.eye(self.B_size, self.B_size)
        neg_idx = torch.ones(self.B_size, self.B_size) - self.pos_idx
        self.neg_idx = neg_idx.nonzero()[:,1].reshape(-1, self.B_size-1).cuda()


        if self.use_queue:
            self.register_buffer("params", torch.tensor([-1]))
            stdv = 1.0 / math.sqrt(inputSize / 3)
            self.register_buffer(
                "memory",
                torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv),
            )
            print("using queue shape: ({},{})".format(self.queueSize, inputSize))

    def nearest_neighbor(self, q):
        # neg logit
        queue = self.memory.clone()
        support_similarities = torch.mm(q, queue.detach().transpose(1,0)) ### B_size x queue size
        most_similar, _ = torch.max(support_similarities, dim=1, keepdim=True)
        return most_similar

    def forward(self, q, k, k_1=None, k_neg=None):
        batchSize = q.shape[0]  # B x Q
        B = batchSize

        C = q.shape[-1]
        ####################
        k = k.detach()
        ####################
        if k_neg is not None:
            k_neg = k_neg.detach()
        if k_1 is not None:
            k_1 = k_1.detach()

        if len(q.shape) > 2:
            q = q.reshape(-1, C)
            k = k.reshape(-1, C)
            batchSize = q.shape[0]

            if k_neg is not None:
                k_neg = k_neg.reshape(-1, C)
            if k_1 is not None:
                k_1 = k_1.reshape(-1,C)

        ########### neg logit
        queue = self.memory.clone()

        # pos logit
        l_pos = torch.bmm(q[:B].reshape(B, 1, -1), k[:B].reshape(B, -1, 1))

        l_pos = l_pos.view(B, 1)
        # l_pos = torch.cat([l_pos, torch.ones(batchSize-B,1)], dim=0)


        l_pos_2 = self.nearest_neighbor(q).view(batchSize, 1)

        if k_1 is not None:
            l_pos_2 = torch.bmm(q.reshape(batchSize, 1, -1), k_1.reshape(batchSize,-1,1))
            l_pos_2 = l_pos_2.view(batchSize, 1)




        if k_1 is not None:
            l_neg_2 = torch.mm(queue.detach(), q.transpose(1, 0))
            l_neg_2 = l_neg_2.transpose(0, 1)

            l_neg_1 = torch.mm(k.detach(), q.transpose(1, 0)) ### B_size x B_size
            l_neg_1 = l_neg_1.gather(1, self.neg_idx)
        else:
            l_neg_2 = torch.mm(queue.detach(), q.transpose(1,0))
            l_neg_2 = l_neg_2.transpose(0, 1)

            # l_neg_1 = torch.cat(torch.split(l_neg_1, B), dim=1)

            # l_neg_1 = torch.mm(k.detach(), q.transpose(1, 0)) ### B_size x B_size
            # l_neg_1 = l_neg_1.gather(1, self.neg_idx)

        if k_neg is not None:
            l_neg_1 = torch.bmm(q.view(batchSize, 1, -1), k_neg.view(batchSize, -1, 1))
            l_neg_1 = l_neg_1.view(batchSize, 1)

        # logging = {
        #     "pos": l_pos.detach().mean().item(),
        #     "pos_NN": l_pos_2.detach().mean().item(),
        #     "neg_diff_scene": l_neg_1.detach().mean().item(),
        #     "neg_same_scene": l_neg_2.detach().mean().item()
        # }

        logging = {
            "pos": l_pos.detach().mean().item(),
            "neg_diff_scene": l_neg_1.detach().mean().item()
        }


        out1 = torch.cat((l_pos, l_neg_1, l_neg_2), dim=1)
        if k_1 is not None:
            out2 = torch.cat((l_pos_2, l_neg_2), dim=1)
            out2 = torch.div(out2, self.T)
            out2 = out2.squeeze().contiguous()
        else: 
            out2 = 0
            # out2 = torch.cat((l_pos_2, l_neg_2), dim=1)
            # out2 = torch.div(out2, self.T)
            # out2 = out2.squeeze().contiguous()

        out1 = torch.div(out1, self.T)
        out1 = out1.squeeze().contiguous()


        # out2 = 0


        if self.use_queue:
            # # update memory
            with torch.no_grad():
                # k = concat_all_gather(k.contiguous())


                ################################
                key = torch.cat([k, k_neg], dim=0)
                k = concat_all_gather(key)

                batchSize = k.shape[0]

                out_ids = torch.arange(batchSize).cuda()
                out_ids += self.index
                out_ids = torch.fmod(out_ids, self.queueSize)
                out_ids = out_ids.long()
                self.memory.index_copy_(0, out_ids, k)
                self.index = (self.index + batchSize) % self.queueSize

        return out1, out2, logging


class NetworkMoCo(nn.Module):
    """ResNet-based backbone for momentum contrast"""

    def __init__(self, proj_dim):
        super(NetworkMoCo, self).__init__()

        # self.backbone = resnet18()
        # self.backbone = vit_small(patch_size=8)
        # self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16') #### 45.9
        
        # self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')    ##### 46.5
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        # self.backbone = torch.hub.load('facebookresearch/vicregl:main', 'convnext_small_alpha0p9')

    def forward(self, x):

        backbone_out = self.backbone(x)
        
        global_feat = backbone_out.contiguous()
        projected_feat = nn.functional.normalize(global_feat, dim=1)

        output = {"global_feat": global_feat, "global_projected_feat": projected_feat}


        return output
