import torch
import time
import numpy as np
import os

from models.utils import (
    AverageMeter,
    count_parameters,
    compute_similarity_global_single,
    compute_similarity_object,
    compute_ME_acc,
    shot_assignment_accuracy,
    compute_confidence_interval
)

from models.contrastive.contrastive_model_base import ContrastiveBase

from nets.moco_utils import MemoryMoCo, NetworkMoCo

from nets.moco_func_utils import (
    moment_update,
    NCESoftmaxLoss,
    batch_shuffle_ddp,
    batch_unshuffle_ddp,
)

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn.functional as F
from data.lowshot.toys4k import BaseDataset, LowShotDataset
from data.lowshot.sampler import CategoriesSamplerME


torch.multiprocessing.set_sharing_strategy('file_system')


BATCH_LOG_STEP = 10
MODEL_LOG_STEP = 50
NUM_SHOTS = 1
NUM_QUERIES = 15
NUM_WAYS = 5
NUM_EPISODE = 500

class VISPE_MoCo(ContrastiveBase):
    def __init__(self, model_args, optim_args, data_args):
        self.build_net(model_args, data_args)
        self.build_optimizer(optim_args)

        self.batch_iter = 0
        self.loss = NCESoftmaxLoss()

    def build_net(self, model_args, data_args):
        # proj_dim = 384
        proj_dim = 768


        self.encoder = NetworkMoCo(proj_dim)
        self.encoder_ema = NetworkMoCo(proj_dim)

        moment_update(self.encoder, self.encoder_ema, 0)

        for name, p in self.encoder_ema.named_parameters():
            p.requires_grad = False

        print("Encoder")
        count_parameters(self.encoder)
        print("Momentum Encoder")
        count_parameters(self.encoder_ema)

        if dist.is_initialized():
            local_rank = int(os.environ["LOCAL_RANK"])


            self.encoder = self.encoder.to(local_rank)
            self.encoder_ema = self.encoder_ema.to(local_rank)

            print("encoder ema check")
            print(local_rank, next(self.encoder_ema.parameters()).device)

            self.encoder = DDP(
                self.encoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
            )

        self.global_contrast = MemoryMoCo(
            proj_dim, K=model_args.moco_K, B_size=data_args.batch_size, T=model_args.moco_T
        )

        if dist.is_initialized():
            self.global_contrast.to(local_rank)

    def train_epoch(self, loader, writer, epoch):

        local_rank = int(os.environ["LOCAL_RANK"])

        # defining meters
        global_loss_meter = AverageMeter()
        sim_dot_meter_global = AverageMeter()  # global encoding obj1 im1 with obj1 im2
        not_sim_dot_meter_global = (
            AverageMeter()
        )  # global encoding obj1 im with obj2 im2
        same_scene_not_sim_dot_meter_global = (
            AverageMeter()
        )  # global encoding obj1 im with obj2 im1

        self.encoder.train()
        # loader.sampler.set_epoch(epoch)

        epoch_start = time.time()

        for idx, batch in enumerate(loader):
            image1 = batch["image1"].cuda()
            image2 = batch["image2"].cuda()

            image_neg = batch["image_neg"].cuda()

            self.optimizer.zero_grad()


            # Shuffle and reverse so that batch norm shortcut doesn't happen
            output_q = self.encoder(image1)


            with torch.no_grad():
                # shuffle for making use of BN
                image2, idx_unshuffle = batch_shuffle_ddp(image2)

                output_k = self.encoder_ema(image2)


                # undo shuffle
                output_k = {
                    k: batch_unshuffle_ddp(v, idx_unshuffle)
                    for k, v in output_k.items()
                }

                # shuffle for making use of BN
                image_neg, idx_unshuffle_neg = batch_shuffle_ddp(image_neg)

                output_k_neg = self.encoder_ema(image_neg)

                # undo shuffle
                output_k_neg = {
                    k: batch_unshuffle_ddp(v, idx_unshuffle_neg)
                    for k, v in output_k_neg.items()
                }

            global_feat_q = output_q["global_projected_feat"]
            global_feat_k = output_k["global_projected_feat"]
            global_feat_k_neg = output_k_neg["global_projected_feat"]


            out1, _, _ = self.global_contrast(global_feat_q, global_feat_k, k_neg=global_feat_k_neg)

            global_loss1 = self.loss(out1)
            global_loss = global_loss1


            # updating the model
            global_loss.backward()
            self.optimizer.step()

            moment_update(self.encoder, self.encoder_ema, 0.999)

            # progress bookkeeping

            global_loss = global_loss.cpu().item()

            with torch.no_grad():
                global_sim_dot, global_not_sim_dot, \
                    global_same_scene_not_sim = compute_similarity_global_single(
                    global_feat_q, global_feat_k, global_feat_k_neg
                )

            if local_rank == 0:

                writer.add_scalar(
                    "iter/train-global-loss", global_loss, self.batch_iter
                )
                writer.add_scalar(
                    "iter/train-global-sim-dot", global_sim_dot, self.batch_iter
                )
                writer.add_scalar(
                    "iter/train-global-not-sim-dot", global_not_sim_dot, self.batch_iter
                )
                writer.add_scalar(
                    "iter/train-global-same-scene-not-sim-dot", global_same_scene_not_sim, self.batch_iter
                )

                global_loss_meter.update(global_loss, len(batch))
                sim_dot_meter_global.update(global_sim_dot, len(batch))
                not_sim_dot_meter_global.update(global_not_sim_dot, len(batch))
                same_scene_not_sim_dot_meter_global.update(global_same_scene_not_sim, len(batch))


                if idx % BATCH_LOG_STEP == 0:
                    print(
                        f"E:{epoch:05d}   "
                        f"B: {idx:03d}|{len(loader):03d}   "
                        f"G Loss: {global_loss_meter.val:.3f}|{global_loss_meter.avg:.3f}  \n"
                        f"G Sim Dot: {sim_dot_meter_global.val:.3f}|{sim_dot_meter_global.avg:.3f}   "
                        f"G Not Sim Dot: {not_sim_dot_meter_global.val:.3f}|{not_sim_dot_meter_global.avg:.3f}   "
                        f"G Same Scene Not Sim Dot: {same_scene_not_sim_dot_meter_global.val:.3f}|{same_scene_not_sim_dot_meter_global.avg:.3f}   "
                    )

            self.batch_iter += 1

        epoch_end = time.time()
        self.lr_scheduler.step()

        print("Time per epoch: {:.2f}".format((epoch_end - epoch_start) / 60))

    @torch.no_grad()
    def eval(self, loader, writer, epoch, tag):
        sim_dot_meter_global = AverageMeter()
        not_sim_dot_meter_global = AverageMeter()
        same_scene_not_sim_dot_meter_global = AverageMeter()

        self.encoder.eval()

        for idx, batch in enumerate(loader):

            image1 = batch["image1"].cuda()
            image2 = batch["image2"].cuda()
            image_neg = batch["image_neg"].cuda()
            B, C, H, W = image1.shape

            output_q = self.encoder.module(image1)
            output_k = self.encoder.module(image2)
            output_k_neg = self.encoder.module(image_neg)


            global_feat_q = output_q["global_projected_feat"]
            global_feat_k = output_k["global_projected_feat"]
            global_feat_k_neg = output_k_neg["global_projected_feat"]


            global_sim_dot, global_not_sim_dot, \
                global_same_scene_not_sim = compute_similarity_global_single(
                global_feat_q, global_feat_k, global_feat_k_neg
            )

            sim_dot_meter_global.update(global_sim_dot, len(batch))
            not_sim_dot_meter_global.update(global_not_sim_dot, len(batch))
            same_scene_not_sim_dot_meter_global.update(global_same_scene_not_sim, len(batch))

            if idx % BATCH_LOG_STEP == 0:
                print(
                    f"E:{epoch:05d} "
                    f"B: {idx:03d}|{len(loader):03d} "
                    f"Sim DotG {tag}: {sim_dot_meter_global.val:.3f}|{sim_dot_meter_global.avg:.3f} "
                    f"Not Sim DotG {tag}: {not_sim_dot_meter_global.val:.3f}|{not_sim_dot_meter_global.avg:.3f} "
                    f"Same Scene Not Sim DotG {tag}: {same_scene_not_sim_dot_meter_global.val:.3f}|{same_scene_not_sim_dot_meter_global.avg:.3f} "
                )

        writer.add_scalar(
            f"epoch/{tag}-global-sim-dot", sim_dot_meter_global.avg, epoch
        )
        writer.add_scalar(
            f"epoch/{tag}-global-not-sim-dot", not_sim_dot_meter_global.avg, epoch
        )
        writer.add_scalar(
            f"epoch/{tag}-global-same-scene-not-sim-dot", same_scene_not_sim_dot_meter_global.avg, epoch
        )

        return sim_dot_meter_global.avg

    def save(self, p):
        save_dct = dict(
            encoder_dict=self.encoder.state_dict(),
            encoder_ema_dict=self.encoder_ema.state_dict(),
            optim_dict=self.optimizer.state_dict(),
            scheduler_dict=self.lr_scheduler.state_dict(),
            batch_iter=self.batch_iter,
        )

        torch.save(save_dct, p)

    def load(self, p):
        dct = torch.load(p, map_location="cpu")
        encoder_dict = dct["encoder_dict"]
        encoder_ema_dict = dct["encoder_ema_dict"]
        optim_dict = dct["optim_dict"]
        scheduler_dict = dct["scheduler_dict"]

        self.encoder.load_state_dict(encoder_dict)
        self.encoder_ema.load_state_dict(encoder_ema_dict)
        self.optimizer.load_state_dict(optim_dict)
        self.lr_scheduler.load_state_dict(scheduler_dict)
        self.batch_iter = dct["batch_iter"]


    @torch.no_grad()
    def LS_eval_me(self, base_loader, loader, writer=None, epoch=-1, shot_given=False):
        self.encoder.eval()
        n_shot = NUM_SHOTS
        n_way = NUM_WAYS
        n_episode = NUM_EPISODE
        per_episode_accuracy = []
        per_epi_shot_assignment_accuracy = []

        #### Extract features from the base classes
        base_classes = self.extract_base_classes(base_loader)

        with torch.no_grad():
            for idx, batch in enumerate(loader):
                ###### images: masked images of all objects in the scene. array of size ((n_shot + n_query)*n_way x N x C x H x W)
                ###### labels: gt labels of the new classes ((n_shot + n_query)*n_way)
                ###### other_labels: all labels of all objects in the scene (B x N)
                images, labels, other_labels = batch
                # import pdb; pdb.set_trace()
                B, N, C, H, W = images.shape
                images = images.reshape(-1, C, H, W)
                ###### masked_batch: objects that are totally occluded
                masked_batch = ~torch.any(images.reshape(B*N, -1), dim=1).numpy()
                # import pdb; pdb.set_trace()

                other_labels = np.array(other_labels).reshape(-1)
                embeds = torch.randn(0)
                images = torch.split(images, 10)
                for image in images:
                    # embed = self.backbone(image.cuda())
                    embed = self.encoder(image.cuda())['global_projected_feat']
                    if len(embeds) == 0:
                        embeds = embed
                    else:
                        embeds = torch.cat([embeds, embed], dim=0)
                embeds = F.normalize(embeds, dim=1)
                # embeds = encoder_output = self.backbone(images.cuda())
                # encoder_output = self.encoder.module(encoder_output)
                # embeds = encoder_output['global_projected_feat']

                shot_labels = np.array(labels[:n_shot * n_way])
                query_labels = np.array(labels[n_shot * n_way:])
                shot_embeds = embeds[:N * n_shot * n_way]
                _, C = shot_embeds.shape
                query_embeds = embeds[N * n_shot * n_way:]


                ############ given shot, only works for 1 shot now
                if shot_given:
                    other_labels_shot = other_labels[:N*n_shot*n_way].reshape(n_way*n_shot, N)
                    shot_inst_ind = torch.argmax(torch.tensor(other_labels_shot).to(torch.int), dim=-1).reshape(-1)
                else:
                    ########## predict shot
                    sim_score = compute_similarity_object(shot_embeds, base_classes)
                    ######### Get the most similarity score and index for each object in shot image
                    best_sim_score, best_sim_ind = torch.max(sim_score, dim=1)
                    ######### We don't count objects that are occluded
                    best_sim_score[masked_batch[:N*n_shot*n_way]] = 9999

                    best_sim_score = best_sim_score.reshape(n_way, n_shot, N)
                    ######### For each scene, the new class is determined as the object with the least similarity with the base classes
                    shot_inst_ind = torch.argmin(best_sim_score, dim=-1).reshape(-1) ##### n_shot * n_way
                shot_embeds = shot_embeds.reshape(n_shot*n_way, N, C) ### n_shot * n_way x N x C

                shot_assignment_acc = shot_assignment_accuracy(other_labels[:N*n_shot*n_way], shot_inst_ind, N, n_shot, n_way)
                per_epi_shot_assignment_accuracy.append(shot_assignment_acc)
                sa = np.mean(per_epi_shot_assignment_accuracy)

                shot_embeds = shot_embeds[torch.arange(n_shot*n_way), shot_inst_ind, :] ##### n_shot * n_way x C


                sim_score_query = compute_similarity_object(query_embeds, shot_embeds)
                best_sim_score_query, best_sim_ind_query = torch.max(sim_score_query, dim=1)
                pred_labels = shot_labels[best_sim_ind_query.cpu().numpy()]

                ######### We don't care about labels of occluded objects
                pred_labels[masked_batch[N*n_shot*n_way:]] = -111
                other_labels_query = other_labels[N * n_shot * n_way:]
                ######### Accuracy computed only on the new classes for all objects
                accuracy = compute_ME_acc(pred_labels, other_labels_query, labels)
                per_episode_accuracy.append(accuracy)
                m = np.mean(per_episode_accuracy)

                p_str = f"{idx:04d}/{len(loader):04d} - curr epi:{accuracy:.4f}  avg:{m:.4f} - curr epi sa:{shot_assignment_acc:.4f}  avg:{sa:.4f}"

                print(p_str, end="\r")


        m, pm = compute_confidence_interval(per_episode_accuracy)
        sa, psa = compute_confidence_interval(per_epi_shot_assignment_accuracy)
        print("Ways {:03d} Shots {:03d} Acc {:.4f} CI {:.4f} SA {:.4f} CI {:.4f}".format(n_way, n_shot, m, pm, sa, psa))

        # writer.add_scalar("epoch/LS-val-acc", m, epoch)

        return m

    @torch.no_grad()
    def extract_base_classes(self, base_loader):
        self.encoder.eval()
        base_class_features = []
        with torch.no_grad():
            for idx, batch in enumerate(base_loader):
                images, cats = batch
                encoder_output = self.encoder(images.cuda())
                embeds = F.normalize(encoder_output['global_projected_feat'], dim=1)

                base_class_features.append(embeds)
            base_class_features = torch.cat(base_class_features, dim=0)
        return base_class_features


    @torch.no_grad()
    def LS_eval_me_single(self, base_loader, loader, writer=None, epoch=-1):
        self.encoder.eval()
        n_shot = NUM_SHOTS
        n_way = NUM_WAYS
        n_episode = NUM_EPISODE
        per_episode_accuracy = []
        per_epi_shot_assignment_accuracy = []
        per_epi_seg_iou = []


        with torch.no_grad():
            for idx, batch in enumerate(loader):
                ###### images: N image per scene ((n_shot + n_query)*n_way x N x C x H x W)
                ###### labels: gt labels of the new classes ((n_shot + n_query)*n_way)
                ###### other_labels: all labels of all objects in the scene (B x N)
                images, labels, other_labels = batch
                # import pdb; pdb.set_trace()
                B, N, C, H, W = images.shape
                images = images.reshape(-1, C, H, W)
                ###### masked_batch: objects that are totally occluded
                masked_batch = ~torch.any(images.reshape(B*N, -1), dim=1).numpy()
                # import pdb; pdb.set_trace()

                other_labels = np.array(other_labels).reshape(-1)
                embeds = torch.randn(0)
                images = torch.split(images, 10)
                for image in images:
                    # embed = self.backbone(image.cuda())
                    embed = self.encoder(image.cuda())['global_projected_feat']
                    if len(embeds) == 0:
                        embeds = embed
                    else:
                        embeds = torch.cat([embeds, embed], dim=0)
                embeds = F.normalize(embeds, dim=1)

                shot_labels = np.array(labels[:n_shot * n_way])
                query_labels = np.array(labels[n_shot * n_way:])
                shot_embeds = embeds[:N * n_shot * n_way]
                _, C = shot_embeds.shape
                query_embeds = embeds[N * n_shot * n_way:]

                shot_embeds = shot_embeds.reshape(-1, N, C) ### n_shot * n_way x N x C


                ######### For each scene, the new class is determined as the object with the least similarity with the base classes
                shot_inst_ind = torch.zeros(shot_embeds.shape[:2]).to(torch.long).reshape(-1) ##### n_shot * n_way

                shot_assignment_acc = shot_assignment_accuracy(other_labels[:N*n_shot*n_way], shot_inst_ind, N, n_shot, n_way)
                per_epi_shot_assignment_accuracy.append(shot_assignment_acc)
                sa = np.mean(per_epi_shot_assignment_accuracy)

                shot_embeds = shot_embeds[torch.arange(n_shot*n_way), shot_inst_ind, :] ##### n_shot * n_way x C


                sim_score_query = compute_similarity_object(query_embeds, shot_embeds)
                best_sim_score_query, best_sim_ind_query = torch.max(sim_score_query, dim=1)
                pred_labels = shot_labels[best_sim_ind_query.cpu().numpy()]
                ######### We don't care about labels of occluded objects
                pred_labels[masked_batch[N*n_shot*n_way:]] = -111
                other_labels_query = other_labels[N * n_shot * n_way:]
                ######### Accuracy computed only on the new classes for all objects
                accuracy = compute_ME_acc(pred_labels, other_labels_query, labels)
                per_episode_accuracy.append(accuracy)
                m = np.mean(per_episode_accuracy)

                p_str = f"{idx:04d}/{len(loader):04d} - curr epi:{accuracy:.4f}  avg:{m:.4f} - curr epi sa:{shot_assignment_acc:.4f}  avg:{sa:.4f}"
                print(p_str, end="\r")


        m, pm = compute_confidence_interval(per_episode_accuracy)
        print("Ways {:03d} Shots {:03d} Acc {:.4f} CI {:.4f}".format(n_way, n_shot, m, pm))
        if writer != None:
            writer.add_scalar("epoch/LS-val-acc", m, epoch)

        return m

    def train_full_ddp(self, loaders, writer, epoch_start, epoch_end, val_freq):

        train_loader = loaders["train_loader"]
        train_loader_eval = loaders["train_loader_eval"]
        train_loader_eval_viz = loaders["train_loader_eval_viz"]
        val_loader = loaders["val_loader"]
        val_loader_viz = loaders["val_loader_viz"]
        ls_val_loader = loaders["ls_val_loader"]

        ls_dataset = LowShotDataset(NUM_SHOTS, NUM_WAYS)
        base_dataset = BaseDataset()

        base_loader = torch.utils.data.DataLoader(
            base_dataset,
            num_workers=12,
            batch_size=16, 
            pin_memory=True
            )
        sampler_params = [ls_dataset.all_labels_shot, ls_dataset.all_labels_query, NUM_EPISODE, NUM_WAYS, NUM_SHOTS, NUM_QUERIES] ## num way, num shot
        ls_sampler = CategoriesSamplerME(*sampler_params)
        loader = torch.utils.data.DataLoader(
            ls_dataset,
            num_workers=12,
            batch_sampler=ls_sampler, 
            pin_memory=True,
            shuffle=False
            )

        local_rank = int(os.environ["LOCAL_RANK"])

        if local_rank == 0:
            ckpt_log_dir = os.path.join(writer.log_dir, "ckpts")

            if not os.path.exists(ckpt_log_dir):
                os.makedirs(ckpt_log_dir)

        best_metric_train = 1e-5
        best_metric_val = 1e-5
        best_ls_val_acc = 1e-5

        eval_metric_train = 1e-5
        eval_metric_val = 1e-5
        ls_val_acc = 1e-5

        for epoch in range(epoch_start, epoch_end):

            self.train_epoch(train_loader, writer, epoch)

            if local_rank == 0:

                if epoch % val_freq == 0:

                    ls_val_acc = self.LS_eval_me(base_loader, loader, writer, epoch)



                    # eval_metric_train = self.eval(
                    #     train_loader_eval, writer, epoch, "eval_train"
                    # )
                    # eval_metric_val = self.eval(val_loader, writer, epoch, "eval_val")

                    print(
                        f"Epoch {epoch:05d}\t"
                        f"Current val seen {eval_metric_train:.3f}\t"
                        f"Current val unseen {eval_metric_val:.3f}\t"
                        f"Current LS val acc {ls_val_acc:.3f}"
                    )

                    if eval_metric_train > best_metric_train and epoch > 20:
                        best_metric_train = eval_metric_train
                        p = os.path.join(writer.log_dir, "best_eval_train_ckpt.pt")
                        self.save(p)

                    if eval_metric_val > best_metric_val and epoch > 20:
                        best_metric_val = eval_metric_val
                        p = os.path.join(writer.log_dir, "best_eval_val_ckpt.pt")
                        self.save(p)

                    if ls_val_acc > best_ls_val_acc:
                        best_ls_val_acc = ls_val_acc
                        p = os.path.join(writer.log_dir, "best_ls_val_ckpt.pt")
                        self.save(p)

                if epoch % MODEL_LOG_STEP == 0:
                    self.save(os.path.join(ckpt_log_dir, "{:04d}.pt".format(epoch)))

                p = os.path.join(writer.log_dir, "last.pt")
                self.save(p)

                print(
                    f"Epoch {epoch:05d}\t"
                    f"Best val seen {best_metric_train:.3f}\t"
                    f"Best val unseen {best_metric_val:.3f}\t"
                    f"Best LS val acc {best_ls_val_acc:.3f}"
                )

            print("{} PRE BARRIER".format(local_rank))
            dist.barrier()
            print("{} POST BARRIER".format(local_rank))
