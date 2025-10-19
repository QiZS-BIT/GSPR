import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from plyfile import PlyData
from sklearn.neighbors import NearestNeighbors
import math


class BaseDataset(Dataset):
    def __init__(self, info_root, gs_dataset_root, max_sh_degree=3):
        super(BaseDataset, self).__init__()
        self.info_root = info_root
        self.gs_dataset_root = gs_dataset_root

        self.infos = self.read_info()

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def read_info(self):
        with open(self.info_root, 'rb') as f:
            infos = pickle.load(f)
        return infos

    def load_gaussian_scene_vfe(self, index):
        cur_info = self.infos[index]
        sample_token = cur_info['sample_token']
        input_filename = sample_token + '.npy'

        input_filepath = os.path.join(self.gs_dataset_root, input_filename)
        gaussian_scene_after_vfe_array = np.load(input_filepath)
        gaussian_scene_after_vfe = torch.from_numpy(gaussian_scene_after_vfe_array).float()
        # for 8192 voxels
        gaussian_scene_after_vfe = torch.cat([gaussian_scene_after_vfe, gaussian_scene_after_vfe], dim=0)
        return gaussian_scene_after_vfe, sample_token


class TripletDataset(BaseDataset):
    def __init__(self, info_root, gs_dataset_root, database_root_list, query_root_list,
                 n_pos, n_neg, n_neg_sample, neg_dist_thres, pos_dist_thres):
        super().__init__(info_root, gs_dataset_root)
        # same elements may exist both in database and query
        self.database_root_list = database_root_list
        self.query_root_list = query_root_list
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.n_neg_sample = n_neg_sample
        self.neg_dist_thres = neg_dist_thres
        self.pos_dist_thres = pos_dist_thres

        assert len(self.database_root_list) != 0, print("Database root list is empty!")
        self.database_index_n_pos = np.load(self.database_root_list[0])
        for i in range(1, len(self.database_root_list)):
            database_index_n_pos = np.load(self.database_root_list[i])
            self.database_index_n_pos = np.concatenate([self.database_index_n_pos, database_index_n_pos], axis=0)
        assert len(self.query_root_list) != 0, print("Query root list is empty!")
        self.query_index_n_pos = np.load(self.query_root_list[0])
        for i in range(1, len(query_root_list)):
            query_index_n_pos = np.load(self.query_root_list[i])
            self.query_index_n_pos = np.concatenate([self.query_index_n_pos, query_index_n_pos], axis=0)

        self.latent_vectors = np.zeros([len(self.database_index_n_pos) + len(self.query_index_n_pos), 256])
        self.is_batch_hard_mining = False

        knn = NearestNeighbors()
        knn.fit(self.database_index_n_pos[:, 1:])
        dist, pos_index_array = knn.radius_neighbors(self.query_index_n_pos[:, 1:],
                                                     radius=self.pos_dist_thres,
                                                     return_distance=True)
        self.pos_index = list(pos_index_array)
        for i, posi in enumerate(self.pos_index):
            pos_index = np.sort(posi)
            self.pos_index[i] = pos_index

        potential_positives = list(knn.radius_neighbors(self.query_index_n_pos[:, 1:],
                                                        radius=self.neg_dist_thres,
                                                        return_distance=False))
        self.neg_index = list()
        for pos in potential_positives:
            self.neg_index.append(np.setdiff1d(np.arange(self.database_index_n_pos.shape[0]), pos, assume_unique=True))

    def __len__(self):
        return len(self.query_index_n_pos)

    def __getitem__(self, index):
        query_index_in_info = self.query_index_n_pos[index, 0].astype(int)
        query_gaussian, sample = self.load_gaussian_scene_vfe(query_index_in_info)
        sample_list = list()
        sample_list.append(sample)
        gaussian_list = list()
        gaussian_list.append(query_gaussian)

        # index-query idx, pos_index-database idx, neg_index-database idx
        pos_sample = np.random.choice(self.pos_index[index], self.n_pos).astype(int)
        pos_index_in_info = self.database_index_n_pos[pos_sample, 0].astype(int)
        for i in range(len(pos_index_in_info)):
            gaussian, sample = self.load_gaussian_scene_vfe(pos_index_in_info[i])
            sample_list.append(sample)
            gaussian_list.append(gaussian)

        if self.is_batch_hard_mining:
            query_offset = len(self.database_root_list)
            query_index = index + query_offset
            query_des = torch.tensor(self.latent_vectors[query_index, :])
            neg_sample_index = np.random.permutation(np.arange(0, len(self.database_index_n_pos)))[:self.n_neg_sample]
            neg_sample = self.latent_vectors[neg_sample_index, :]
            neg_des = torch.tensor(neg_sample)
            dist = torch.norm(query_des[None, :] - neg_des, dim=1)
            result = dist.topk(self.n_neg * 10, largest=False)
            neg_dist, neg_idx_in_sample = result.values, result.indices
            neg_idx_whole = neg_sample_index[neg_idx_in_sample]
            neg_idx = torch.tensor(list(set(neg_idx_whole.tolist()) & set(self.neg_index[index].tolist())))
            neg_idx = neg_idx[:self.n_neg]
        else:
            neg_idx = np.random.choice(self.neg_index[index], self.n_neg).astype(int)
        neg_index_in_info = self.database_index_n_pos[neg_idx, 0].astype(int)
        for i in range(len(neg_index_in_info)):
            gaussian, sample = self.load_gaussian_scene_vfe(neg_index_in_info[i])
            sample_list.append(sample)
            gaussian_list.append(gaussian)

        gaussian_tensor = torch.stack(gaussian_list, dim=0)

        res_dict = dict({'gaussian_tuple': gaussian_tensor})

        return res_dict

    def update_latent_vectors(self, vecs_filepath):
        latent_vectors = pickle.load(open(vecs_filepath, 'rb'))
        latent_vectors_full = np.zeros([len(self.database_index_n_pos) + len(self.query_index_n_pos), 256])
        for i in range(len(latent_vectors)):
            latent_vectors_full[i, :] = latent_vectors[i]
        self.latent_vectors = latent_vectors_full
        self.is_batch_hard_mining = True


class QueryDataset(BaseDataset):
    def __init__(self, info_root, gs_dataset_root, database_root_list, query_root_list, non_triv_pos_dist_thres):
        # database and query are separated
        super().__init__(info_root, gs_dataset_root)
        self.database_root_list = database_root_list
        self.query_root_list = query_root_list
        self.positives = None
        self.non_triv_pos_dist_thres = non_triv_pos_dist_thres

        assert len(self.database_root_list) != 0, print("Database root list is empty!")
        self.database_index_n_pos = np.load(self.database_root_list[0])
        for i in range(1, len(self.database_root_list)):
            database_index_n_pos = np.load(self.database_root_list[i])
            self.database_index_n_pos = np.concatenate([self.database_index_n_pos, database_index_n_pos], axis=0)
        assert len(self.query_root_list) != 0, print("Query root list is empty!")
        self.query_index_n_pos = np.load(self.query_root_list[0])
        for i in range(1, len(query_root_list)):
            query_index_n_pos = np.load(self.query_root_list[i])
            self.query_index_n_pos = np.concatenate([self.query_index_n_pos, query_index_n_pos], axis=0)

        self.dataset_index_n_pos = np.concatenate([self.database_index_n_pos, self.query_index_n_pos], axis=0)
        self.num_db = self.database_index_n_pos.shape[0]
        self.num_query = self.query_index_n_pos.shape[0]

    def __len__(self):
        return len(self.dataset_index_n_pos)

    def __getitem__(self, index):
        index_in_info = self.dataset_index_n_pos[index, 0].astype(int)
        gaussian, sample = self.load_gaussian_scene_vfe(index_in_info)
        gaussian_list = list()
        gaussian_list.append(gaussian)
        gaussian_tensor = torch.stack(gaussian_list, dim=0)
        res_dict = dict({'gaussian_tuple': gaussian_tensor, 'sample_list': [sample]})
        return res_dict

    def get_positives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors()
            dataset = np.ascontiguousarray(self.dataset_index_n_pos[:self.num_db, 1:])
            knn.fit(dataset)
            self.positives = list(knn.radius_neighbors(self.dataset_index_n_pos[self.num_db:, 1:],
                                                       radius=self.non_triv_pos_dist_thres,
                                                       return_distance=False))
        return self.positives
