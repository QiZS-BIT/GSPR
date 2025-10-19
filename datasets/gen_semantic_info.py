import os
import csv
import pickle
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    # when generating the semantic infos of SON and SQ splits, there's some modifications needed.
    # as the 'train_query.npy' and 'val_query.npy' of SON and SQ doesn't exist,
    # the corresponding lines should be commented out.
    start_frame = 0
    dataset_root = '/media/octane17/T7ShieldNus/GSPR/data'
    db_ind_root = os.path.join(dataset_root, 'bs_db.npy')
    train_query_ind_root = os.path.join(dataset_root, 'bs_train_query.npy')
    val_query_ind_root = os.path.join(dataset_root, 'bs_val_query.npy')
    test_query_ind_root = os.path.join(dataset_root, 'bs_test_query.npy')

    infos_bs_path = os.path.join(dataset_root, 'nuscenes_infos-bs.pkl')
    with open(infos_bs_path, 'rb') as f:
        infos_bs = pickle.load(f)
    bs_db = np.load(db_ind_root)
    bs_train_query = np.load(train_query_ind_root)
    bs_val_query = np.load(val_query_ind_root)
    bs_test_query = np.load(test_query_ind_root)
    bs_whole = np.concatenate([bs_db, bs_train_query, bs_val_query, bs_test_query], axis=0)

    img_filepath_list = []
    for i in tqdm(range(start_frame, bs_whole.shape[0])):
        sample_info_seq = []
        cur_info_idx = bs_whole[i][0]
        cur_info = infos_bs[int(cur_info_idx)]
        sample_info_seq.append(cur_info)
        prev_info_idx = -1
        prev_info_token = cur_info['prev']
        for info_idx in range(len(infos_bs)):
            if infos_bs[info_idx]['sample_token'] == prev_info_token:
                prev_info_idx = info_idx
        sample_info_seq.append(infos_bs[prev_info_idx])
        next_info_idx = -1
        next_info_token = cur_info['next']
        for info_idx in range(len(infos_bs)):
            if infos_bs[info_idx]['sample_token'] == next_info_token:
                next_info_idx = info_idx
        sample_info_seq.append(infos_bs[next_info_idx])

        for info in sample_info_seq:
            cam_infos = info['cam_infos']
            for cam_info_key in cam_infos:
                cam_info = cam_infos[cam_info_key]
                img_filepath = os.path.join(dataset_root, cam_info['filename'])
                img_filepath_list.append(img_filepath)

    with open(os.path.join(dataset_root, 'semantic_info_bs.pkl'), 'wb') as f:
        pickle.dump(img_filepath_list, f)
