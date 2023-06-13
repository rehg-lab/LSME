import numpy as np
import torch
from torch.utils.data import Sampler
from collections import defaultdict

'''
adopted from 
https://github.com/mileyan/simple_shot/blob/5d38fc83e698f11fea56bdfa3e1a8fdde9935e1a/src/datasets/sampler.py
'''

class CategoriesSampler(Sampler):

    def __init__(self, label, n_iter, n_way, n_shot, n_query, mode):

        self.n_iter = n_iter
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        
        assert mode in ["single_instance_shots", "multi_instance_shots"]
        
        self.mode = mode

        if isinstance(label, torch.Tensor):
            if label.device.type == 'cuda':
                label = label.cpu()
            label = [x.item() for x in label]

        
        label = np.array(label)
        # this contains the indices in the data 
        # list corresponding to each of the classes
        self.m_ind = [] 
        unique = np.unique(label)
        unique = np.sort(unique)
        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        
    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for i in range(self.n_iter):
            # get the indices for the shots
            batch_gallery = []
            # get the indices for the queries
            batch_query = []
            
            # pick which classes we're gonna use

            classes = torch.randperm(len(self.m_ind))[:self.n_way]
            
            if self.mode == "multi_instance_shots":
                for c in classes:
                    # l -> the instance indices of the class we have chosen (c)
                    l = self.m_ind[c.item()]
                    # permutation to index over the instance indices of class c
                    pos = torch.randperm(l.size()[0])

                    # take the first n_shot of the permuted instance indices to be the shots
                    batch_gallery.append(l[pos[:self.n_shot]])
                    # take the n_shot to n_shot+n_query to be the queries
                    batch_query.append(l[pos[self.n_shot : self.n_shot + self.n_query]])

            if self.mode == "single_instance_shots":
                for c in classes:
                    # l -> the instance indices of the class we have chosen (c)
                    l = self.m_ind[c.item()]
                    # permutation to index over the instance indices of class c
                    pos = torch.randperm(l.size()[0])
                    shot_idx = l[pos[0]]

                    # take the first n_shot of the permuted instance indices to be the shots
                    batch_gallery.append(torch.tensor([shot_idx] * self.n_shot))
                    # take the n_shot to n_shot+n_query to be the queries
                    batch_query.append(l[pos[1 : 1 + self.n_query]])

            # concatenate the shots and the queries and yield them

            batch = torch.cat(batch_gallery + batch_query)


            yield iter(batch.tolist())


class CategoriesSamplerME(Sampler):
    def __init__(self, labels_shot, labels_query, n_iter, n_way, n_shot, n_query):
        self.n_iter = n_iter
        self.n_shot = n_shot
        self.n_way = n_way
        self.n_query = n_query

        if isinstance(labels_shot, torch.Tensor):
            if labels_shot.device.type == 'cuda':
                labels_shot = labels_shot.cpu()
            labels_shot = [x.item() for x in labels_shot]

        if isinstance(labels_query, torch.Tensor):
            if labels_query.device.type == 'cuda':
                labels_query = labels_query.cpu()
            labels_query = [x.item() for x in labels_query]

        labels_shot = np.array(labels_shot)
        labels_query = np.array(labels_query)

        self.total_shots = len(labels_shot)

        self.m_ind_shot = []
        self.m_ind_query = []

        unique = np.unique(labels_shot)
        unique = np.sort(unique)
        for i in unique:
            ind_shot = np.argwhere(labels_shot == i).reshape(-1)
            ind_query = np.argwhere(labels_query == i).reshape(-1)

            ind_shot = torch.from_numpy(ind_shot)
            ind_query = torch.from_numpy(ind_query)

            self.m_ind_shot.append(ind_shot)
            self.m_ind_query.append(ind_query)

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for i in range(self.n_iter):
            batch_gallery = []
            batch_query = []

            classes = torch.randperm(len(self.m_ind_shot))[:self.n_way]

            for c in classes:
                l_shot = self.m_ind_shot[c.item()]
                l_query = self.m_ind_query[c.item()]

                pos_shot = torch.randperm(l_shot.size()[0])
                pos_query = torch.randperm(l_query.size()[0])

                batch_gallery.append(l_shot[pos_shot[:self.n_shot]])
                batch_query.append(l_query[pos_query[:self.n_query]]+self.total_shots)

            batch = torch.cat(batch_gallery+batch_query)

            yield iter(batch.tolist())

