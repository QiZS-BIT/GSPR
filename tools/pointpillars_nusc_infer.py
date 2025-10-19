from mmdet3d.apis import init_model, inference_detector
import os
import pickle
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
import open3d as o3d


if __name__ == '__main__':
    start_frame = 0
    dataset_root = '/media/octane17/T7ShieldNus/GSPR/data'
    db_ind_root = os.path.join(dataset_root, 'bs_db.npy')
    train_query_ind_root = os.path.join(dataset_root, 'bs_train_query.npy')
    val_query_ind_root = os.path.join(dataset_root, 'bs_val_query.npy')
    test_query_ind_root = os.path.join(dataset_root, 'bs_test_query.npy')
    save_annotation_root = os.path.join(dataset_root, 'pp_annotation')
    model_config_file = 'pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py'
    checkpoint_file = 'hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth'

    class_names = [
        'car',
        'truck',
        'trailer',
        'bus',
        'construction_vehicle',
        'bicycle',
        'motorcycle',
        'pedestrian',
        'traffic_cone',
        'barrier',
    ]

    model = init_model(model_config_file, checkpoint_file)

    infos_bs_path = os.path.join(dataset_root, 'nuscenes_infos-son.pkl')
    with open(infos_bs_path, 'rb') as f:
        infos_bs = pickle.load(f)
    bs_db = np.load(db_ind_root)
    bs_train_query = np.load(train_query_ind_root)
    bs_val_query = np.load(val_query_ind_root)
    bs_test_query = np.load(test_query_ind_root)
    bs_whole = np.concatenate([bs_db, bs_train_query, bs_val_query, bs_test_query], axis=0)

    for i in tqdm(range(start_frame, len(infos_bs))):
        sample_info_seq = []
        cur_info = infos_bs[i]

        token = cur_info['sample_token']
        pcd_path = cur_info['lidar_infos']['LIDAR_TOP']['filename']
        pcd_abs_path = os.path.join(dataset_root, pcd_path)

        result, data = inference_detector(model, pcd_abs_path)

        bbox_corners = result.pred_instances_3d.bboxes_3d.corners.cpu().numpy()
        bbox_labels = result.pred_instances_3d.labels_3d.cpu().numpy()

        # save pred result
        annotations = list()
        for bbox_corner, bbox_label in zip(bbox_corners, bbox_labels):
            class_name = ''
            if bbox_label == 4:
                class_name = 'vehicle.' + 'construction'
            elif bbox_label == 7:
                class_name = 'human.' + class_names[bbox_label]
            if bbox_label <= 6:
                class_name = 'vehicle.' + class_names[bbox_label]
            annotation_metadata = {
                'token': '',
                'sample_token': token,
                'instance_token': '',
                'visibility_token': '',
                'attribute_tokens': [],
                'translation': [],
                'size': [],
                'rotation': [],
                'corner': bbox_corner,
                'prev': '',
                'next': '',
                'num_lidar_pts': 0,
                'num_radar_pts': 0,
                'category_name': class_name
            }
            annotations.append(annotation_metadata)

        annotation_save_path = os.path.join(save_annotation_root, token + '.pkl')
        with open(annotation_save_path, 'wb') as f:
            pickle.dump(annotations, f)
