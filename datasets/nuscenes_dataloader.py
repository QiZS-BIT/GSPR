from torch.utils.data import DataLoader, default_collate
from config.params import ModelParams
from datasets.nuscenes_dataset import TripletDataset, QueryDataset


def make_dataloader(working_mode, params: ModelParams):
    if working_mode == 'train':
        train_dataset = TripletDataset(
            params.info_root,
            params.gaussian_dataset_root,
            params.database_index_list,
            params.train_query_index_list,
            params.pos_num,
            params.neg_num,
            1000,
            params.neg_dist_threshold,
            params.pos_dist_threshold
        )
        train_collate_fn = make_collate_fn()
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=train_collate_fn,
            num_workers=params.num_workers,
            pin_memory=True
        )

        val_dataset = TripletDataset(
            params.info_root,
            params.gaussian_dataset_root,
            params.database_index_list,
            params.val_query_index_list,
            params.pos_num,
            params.neg_num,
            1000,
            params.neg_dist_threshold,
            params.pos_dist_threshold
        )
        val_collate_fn = make_collate_fn()
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=val_collate_fn,
            num_workers=params.num_workers,
            pin_memory=True
        )

        train_query_dataset = QueryDataset(
            params.info_root,
            params.gaussian_dataset_root,
            params.database_index_list,
            params.train_query_index_list,
            params.gth_dist_threshold
        )
        train_query_loader = DataLoader(
            dataset=train_query_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True
        )

        val_query_dataset = QueryDataset(
            params.info_root,
            params.gaussian_dataset_root,
            params.database_index_list,
            params.val_query_index_list,
            params.gth_dist_threshold
        )
        val_query_loader = DataLoader(
            dataset=val_query_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader, train_query_loader, val_query_loader, val_query_dataset

    elif working_mode == 'test':
        test_dataset = QueryDataset(
            params.info_root,
            params.gaussian_dataset_root,
            params.database_index_list,
            params.test_query_index_list,
            params.gth_dist_threshold
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True
        )

        return test_loader, test_dataset


def make_collate_fn():
    def collate_fn(batch):
        return default_collate(batch)
    return collate_fn
