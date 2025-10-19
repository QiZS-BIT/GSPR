import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import random
import pickle
import matplotlib.pyplot as plt


def main():
    random.seed(1)
    dataroot = '/media/octane17/T7ShieldNus/GSPR/data'
    infos_bs_path = os.path.join(dataroot, 'nuscenes_infos-bs.pkl')
    infos_son_path = os.path.join(dataroot, 'nuscenes_infos-son.pkl')
    infos_sq_path = os.path.join(dataroot, 'nuscenes_infos-sq.pkl')

    with open(infos_bs_path, 'rb') as f:
        infos_bs = pickle.load(f)

    with open(infos_son_path, 'rb') as f:
        infos_son = pickle.load(f)

    with open(infos_sq_path, 'rb') as f:
        infos_sq = pickle.load(f)

    pos_whole_bs = []
    timestamps_bs = []
    available_ind = []
    pos_whole_son = []
    timestamps_son = []
    available_ind_son = []
    pos_whole_sq = []
    timestamps_sq = []
    available_ind_sq = []

    for i, info in enumerate(infos_bs):
        pos = info['lidar_infos']['LIDAR_TOP']['ego_pose']['translation']
        pos_whole_bs.append(pos[:2])
        timestamp = info['timestamp']
        timestamps_bs.append(timestamp)
        if info['prev'] != '' and info['next'] != '':
            available_ind.append(i)
    print("non-edge frames for boston seaport: ", len(available_ind))

    for i, info in enumerate(infos_son):
        pos = info['lidar_infos']['LIDAR_TOP']['ego_pose']['translation']
        pos_whole_son.append(pos[:2])
        timestamp = info['timestamp']
        timestamps_son.append(timestamp)
        if info['prev'] != '' and info['next'] != '':
            available_ind_son.append(i)
    print("non-edge frames for singapore one north: ", len(available_ind_son))

    for i, info in enumerate(infos_sq):
        pos = info['lidar_infos']['LIDAR_TOP']['ego_pose']['translation']
        pos_whole_sq.append(pos[:2])
        timestamp = info['timestamp']
        timestamps_sq.append(timestamp)
        if info['prev'] != '' and info['next'] != '':
            available_ind_sq.append(i)
    print("non-edge frames for singapore queenstown: ", len(available_ind_sq))

    pos_whole_bs = np.array(pos_whole_bs, dtype=np.float32)
    timestamps_bs = np.array(timestamps_bs, dtype=np.float32).reshape(-1, 1)
    pos_whole_son = np.array(pos_whole_son, dtype=np.float32)
    timestamps_son = np.array(timestamps_son, dtype=np.float32).reshape(-1, 1)
    pos_whole_sq = np.array(pos_whole_sq, dtype=np.float32)
    timestamps_sq = np.array(timestamps_sq, dtype=np.float32).reshape(-1, 1)

    print('total frames for boston seaport: ', pos_whole_bs.shape[0])
    print('total frames for singapore one north: ', pos_whole_son.shape[0])
    print('total frames for singapore queenstown: ', pos_whole_sq.shape[0])

    # ==================================================================
    #                    generate database indices
    # ==================================================================
    print('==> generating database...')
    DIS_TH_DB = 3  # Map Point Distance (m)

    pos_whole_bs = np.concatenate(
        (np.arange(len(pos_whole_bs), dtype=np.int32).reshape(-1, 1), np.array(pos_whole_bs)),
        axis=1).astype(np.float32)

    pos_whole_son = np.concatenate(
        (np.arange(len(pos_whole_son), dtype=np.int32).reshape(-1, 1), np.array(pos_whole_son)),
        axis=1).astype(np.float32)

    pos_whole_sq = np.concatenate(
        (np.arange(len(pos_whole_sq), dtype=np.int32).reshape(-1, 1), np.array(pos_whole_sq)),
        axis=1).astype(np.float32)

    pos_bs_db = pos_whole_bs[available_ind[0], :].reshape(1, -1)
    for i in range(available_ind[0], pos_whole_bs.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(pos_bs_db[:, 1:3])
        dis, index = knn.kneighbors(pos_whole_bs[i, 1:3].reshape(1, -1), 1, return_distance=True)

        if dis > DIS_TH_DB and i in available_ind:
            pos_bs_db = np.concatenate((pos_bs_db, pos_whole_bs[i, :].reshape(1, -1)), axis=0)

    pos_son_db = pos_whole_son[0, :].reshape(1, -1)  # add the first frame
    for i in range(1, pos_whole_son.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(pos_son_db[:, 1:3])
        dis, index = knn.kneighbors(pos_whole_son[i, 1:4].reshape(1, -1), 1, return_distance=True)

        if dis > DIS_TH_DB and i in available_ind_son:
            pos_son_db = np.concatenate((pos_son_db, pos_whole_son[i, :].reshape(1, -1)), axis=0)

    pos_sq_db = pos_whole_sq[0, :].reshape(1, -1)  # add the first frame
    for i in range(1, pos_whole_sq.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(pos_sq_db[:, 1:3])
        dis, index = knn.kneighbors(pos_whole_sq[i, 1:4].reshape(1, -1), 1, return_distance=True)

        if dis > DIS_TH_DB and i in available_ind_sq:
            pos_sq_db = np.concatenate((pos_sq_db, pos_whole_sq[i, :].reshape(1, -1)), axis=0)

    print('boston seaport database frames: ', pos_bs_db.shape[0])
    print('singapore one north database frames: ', pos_son_db.shape[0])
    print('singapore queenstown database frames: ', pos_sq_db.shape[0])

    # ==================================================================
    # generate train_query, val_query, test_query indices
    # ==================================================================
    print('==> generating query indices...')
    DIS_TH_QUERY = 9
    SEPERATE_TH_BS = 105
    SEPERATE_TH_SON = 15
    SEPERATE_TH_SHV = 15
    SEPERATE_TH_SQ = 15

    # boston-seaport
    timestamps = np.array(timestamps_bs - min(timestamps_bs)) / (3600 * 24 * 1e6)
    fi_bs_train, _ = np.where(timestamps < SEPERATE_TH_BS)
    fi_bs_testval, _ = np.where(timestamps >= SEPERATE_TH_BS)

    unavailable_ind = list(set(np.arange(len(pos_whole_bs))) - set(available_ind))

    fi_bs_db = pos_bs_db[:, 0].astype(int)
    fi_bs_train_query = list(set(fi_bs_train) - set(fi_bs_db) - set(unavailable_ind))
    fi_bs_testval_query = list(set(fi_bs_testval) - set(fi_bs_db) - set(unavailable_ind))
    fi_bs_val_query = random.sample(fi_bs_testval_query, int(len(fi_bs_testval_query) * 0.25))
    fi_bs_test_query = list(set(fi_bs_testval_query) - set(fi_bs_val_query))
    pos_bs_train_query = pos_whole_bs[fi_bs_train_query]
    pos_bs_test_query = pos_whole_bs[fi_bs_test_query]
    pos_bs_val_query = pos_whole_bs[fi_bs_val_query]

    pos_bs_train_query_new = pos_bs_train_query[0, :].reshape(1, -1)
    for i in range(1, pos_bs_train_query.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(pos_bs_train_query_new[:, 1:3])
        dis, index = knn.kneighbors(pos_bs_train_query[i, 1:3].reshape(1, -1), 1, return_distance=True)

        if dis > DIS_TH_QUERY and i in available_ind:
            pos_bs_train_query_new = np.concatenate((pos_bs_train_query_new, pos_bs_train_query[i, :].reshape(1, -1)), axis=0)
    pos_bs_train_query = pos_bs_train_query_new

    # query downsample
    pos_bs_test_query_new = pos_bs_test_query[0, :].reshape(1, -1)
    for i in range(1, pos_bs_test_query.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(pos_bs_test_query_new[:, 1:3])
        dis, index = knn.kneighbors(pos_bs_test_query[i, 1:3].reshape(1, -1), 1, return_distance=True)

        if dis > DIS_TH_QUERY and i in available_ind:
            pos_bs_test_query_new = np.concatenate((pos_bs_test_query_new, pos_bs_test_query[i, :].reshape(1, -1)), axis=0)
    pos_bs_test_query = pos_bs_test_query_new

    pos_bs_val_query_new = pos_bs_val_query[0, :].reshape(1, -1)
    for i in range(1, pos_bs_val_query.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(pos_bs_val_query_new[:, 1:3])
        dis, index = knn.kneighbors(pos_bs_val_query[i, 1:3].reshape(1, -1), 1, return_distance=True)

        if dis > DIS_TH_QUERY and i in available_ind:
            pos_bs_val_query_new = np.concatenate((pos_bs_val_query_new, pos_bs_val_query[i, :].reshape(1, -1)), axis=0)
    pos_bs_val_query = pos_bs_val_query_new

    # singapore one north
    timestamps_son = np.array(timestamps_son - min(timestamps_son)) / (3600 * 24 * 1e6)
    fi_son_train, _ = np.where(timestamps_son < SEPERATE_TH_SON)
    fi_son_testval, _ = np.where(timestamps_son >= SEPERATE_TH_SON)

    unavailable_ind_son = list(set(np.arange(len(pos_whole_son))) - set(available_ind_son))

    fi_son_db = pos_son_db[:, 0].astype(int)
    fi_son_train_query = list(set(fi_son_train) - set(fi_son_db) - set(unavailable_ind_son))
    fi_son_test_query = list(set(fi_son_testval) - set(fi_son_db) - set(unavailable_ind_son))
    pos_son_train_query = pos_whole_son[fi_son_train_query]
    pos_son_test_query = pos_whole_son[fi_son_test_query]

    pos_son_train_query_new = pos_son_train_query[0, :].reshape(1, -1)
    for i in range(1, pos_son_train_query.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(pos_son_train_query_new[:, 1:3])
        dis, index = knn.kneighbors(pos_son_train_query[i, 1:3].reshape(1, -1), 1, return_distance=True)

        if dis > DIS_TH_QUERY and i in available_ind_son:
            pos_son_train_query_new = np.concatenate((pos_son_train_query_new, pos_son_train_query[i, :].reshape(1, -1)), axis=0)
    pos_son_train_query = pos_son_train_query_new

    pos_son_test_query_new = pos_son_test_query[0, :].reshape(1, -1)
    for i in range(1, pos_son_test_query.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(pos_son_test_query_new[:, 1:3])
        dis, index = knn.kneighbors(pos_son_test_query[i, 1:3].reshape(1, -1), 1, return_distance=True)

        if dis > DIS_TH_QUERY and i in available_ind_son:
            pos_son_test_query_new = np.concatenate((pos_son_test_query_new, pos_son_test_query[i, :].reshape(1, -1)), axis=0)
    pos_son_test_query = pos_son_test_query_new

    # singapore queenstown
    timestamps_sq = np.array(timestamps_sq - min(timestamps_sq)) / (3600 * 24 * 1e6)
    fi_sq_train, _ = np.where(timestamps_sq < SEPERATE_TH_SQ)
    fi_sq_testval, _ = np.where(timestamps_sq >= SEPERATE_TH_SQ)

    unavailable_ind_sq = list(set(np.arange(len(pos_whole_sq))) - set(available_ind_sq))

    fi_sq_db = pos_sq_db[:, 0].astype(int)
    fi_sq_train_query = list(set(fi_sq_train) - set(fi_sq_db) - set(unavailable_ind_sq))
    fi_sq_test_query = list(set(fi_sq_testval) - set(fi_sq_db) - set(unavailable_ind_sq))
    pos_sq_train_query = pos_whole_sq[fi_sq_train_query]
    pos_sq_test_query = pos_whole_sq[fi_sq_test_query]

    pos_sq_train_query_new = pos_sq_train_query[0, :].reshape(1, -1)
    for i in range(1, pos_sq_train_query.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(pos_sq_train_query_new[:, 1:3])
        dis, index = knn.kneighbors(pos_sq_train_query[i, 1:3].reshape(1, -1), 1, return_distance=True)

        if dis > DIS_TH_QUERY and i in available_ind_sq:
            pos_sq_train_query_new = np.concatenate(
                (pos_sq_train_query_new, pos_sq_train_query[i, :].reshape(1, -1)), axis=0)
    pos_sq_train_query = pos_sq_train_query_new

    pos_sq_test_query_new = pos_sq_test_query[0, :].reshape(1, -1)
    for i in range(1, pos_sq_test_query.shape[0]):
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(pos_sq_test_query_new[:, 1:3])
        dis, index = knn.kneighbors(pos_sq_test_query[i, 1:3].reshape(1, -1), 1, return_distance=True)

        if dis > DIS_TH_QUERY and i in available_ind_sq:
            pos_sq_test_query_new = np.concatenate((pos_sq_test_query_new, pos_sq_test_query[i, :].reshape(1, -1)), axis=0)
    pos_sq_test_query = pos_sq_test_query_new

    # ==================================================================
    # delete train/val/test query who has no GT positive in the database
    # ==================================================================
    DIS_TH = 9
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(pos_bs_db[:, 1:3])

    pos_bs_train_query_new = list()
    for i in range(len(pos_bs_train_query)):
        dis, index = knn.kneighbors(pos_bs_train_query[i, 1:3].reshape(1, -1), 1, return_distance=True)
        if dis < DIS_TH:
            pos_bs_train_query_new.append(pos_bs_train_query[i, :])
    pos_bs_train_query = np.array(pos_bs_train_query_new)

    pos_bs_test_query_new = list()
    for i in range(len(pos_bs_test_query)):
        dis, index = knn.kneighbors(pos_bs_test_query[i, 1:3].reshape(1, -1), 1, return_distance=True)
        if dis < DIS_TH:
            pos_bs_test_query_new.append(pos_bs_test_query[i, :])
    pos_bs_test_query = np.array(pos_bs_test_query_new)

    pos_bs_val_query_new = list()
    for i in range(len(pos_bs_val_query)):
        dis, index = knn.kneighbors(pos_bs_val_query[i, 1:3].reshape(1, -1), 1, return_distance=True)
        if dis < DIS_TH:
            pos_bs_val_query_new.append(pos_bs_val_query[i, :])
    pos_bs_val_query = np.array(pos_bs_val_query_new)

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(pos_son_db[:, 1:3])

    pos_son_query_new = list()
    for i in range(len(pos_son_train_query)):
        dis, index = knn.kneighbors(pos_son_train_query[i, 1:4].reshape(1, -1), 1, return_distance=True)
        if dis < DIS_TH:
            pos_son_query_new.append(pos_son_train_query[i, :])
    pos_son_train_query = np.array(pos_son_train_query)

    pos_son_query_new = list()
    for i in range(len(pos_son_test_query)):
        dis, index = knn.kneighbors(pos_son_test_query[i, 1:4].reshape(1, -1), 1, return_distance=True)
        if dis < DIS_TH:
            pos_son_query_new.append(pos_son_test_query[i, :])
    pos_son_test_query = np.array(pos_son_test_query)

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(pos_sq_db[:, 1:3])

    pos_sq_train_query_new = list()
    for i in range(len(pos_sq_train_query)):
        dis, index = knn.kneighbors(pos_sq_train_query[i, 1:4].reshape(1, -1), 1, return_distance=True)
        if dis < DIS_TH:
            pos_sq_train_query_new.append(pos_sq_train_query[i, :])
    pos_sq_train_query = np.array(pos_sq_train_query_new)

    pos_sq_test_query_new = list()
    for i in range(len(pos_sq_test_query)):
        dis, index = knn.kneighbors(pos_sq_test_query[i, 1:4].reshape(1, -1), 1, return_distance=True)
        if dis < DIS_TH:
            pos_sq_test_query_new.append(pos_sq_test_query[i, :])
    pos_sq_test_query = np.array(pos_sq_test_query_new)

    print('boston seaport train query: ', pos_bs_train_query.shape[0])
    print('boston seaport test query: ', pos_bs_test_query.shape[0])
    print('boston seaport val query: ', pos_bs_val_query.shape[0])
    print('singapore one north train query: ', pos_son_train_query.shape[0])
    print('singapore one north test query: ', pos_son_test_query.shape[0])
    print('singapore queenstown train query: ', pos_sq_train_query.shape[0])
    print('singapore queenstown test query: ', pos_sq_test_query.shape[0])

    # plt.scatter(pos_whole_bs[:, 1], pos_whole_bs[:, 2], c='green', s=120)
    # plt.scatter(pos_bs_db[:, 1], pos_bs_db[:, 2], c='blue', s=40)
    # plt.scatter(pos_bs_train_query[:, 1], pos_bs_train_query[:, 2], c='red', s=10)
    # plt.scatter(pos_bs_test_query[:, 1], pos_bs_test_query[:, 2], c='black', s=10)
    # plt.scatter(pos_bs_val_query[:, 1], pos_bs_val_query[:, 2], c='grey', s=10)
    # plt.show()

    # ============================================================
    # save database, train queries, test queries, validate queries
    # ============================================================
    print('===> saving database and queries..')
    np.save(os.path.join(dataroot, 'bs_whole.npy'), pos_whole_bs)
    np.save(os.path.join(dataroot, 'bs_db.npy'), pos_bs_db)
    np.save(os.path.join(dataroot, 'bs_train_query.npy'), pos_bs_train_query)
    np.save(os.path.join(dataroot, 'bs_val_query.npy'), pos_bs_val_query)
    np.save(os.path.join(dataroot, 'bs_test_query.npy'), pos_bs_test_query)
    np.save(os.path.join(dataroot, 'son_whole.npy'), pos_whole_son)
    np.save(os.path.join(dataroot, 'son_db.npy'), pos_son_db)
    # np.save(os.path.join(dataroot, 'son_train_query.npy'), pos_son_train_query)
    np.save(os.path.join(dataroot, 'son_test_query.npy'), pos_son_test_query)
    np.save(os.path.join(dataroot, 'sq_whole.npy'), pos_whole_sq)
    # np.save(os.path.join(dataroot, 'sq_train_query.npy'), pos_sq_train_query)
    np.save(os.path.join(dataroot, 'sq_test_query.npy'), pos_sq_test_query)
    np.save(os.path.join(dataroot, 'sq_db.npy'), pos_sq_db)


if __name__ == '__main__':
    main()
