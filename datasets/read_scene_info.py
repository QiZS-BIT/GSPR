import os
import pickle
import numpy as np
import open3d
import cv2
import shutil
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes, transform_matrix, Quaternion
from utils import load_lidar_data, matrix_to_quaternion, get_bbox_vertice


def gen_info(nusc, sample_tokens, use_gth_bbox, annotation_folder):
    idx = 0
    infos = list()
    valid_object_categories = ['vehicle', 'human', 'cycle']
    for sample_token in tqdm(sample_tokens):
        # each info corresponds to a sample
        sample = nusc.get('sample', sample_token)
        # store scene info
        info = dict()
        info['sample_token'] = sample_token
        info['prev'] = sample['prev']
        info['next'] = sample['next']
        info['timestamp'] = sample['timestamp']
        info['scene_token'] = sample['scene_token']
        cam_names = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_RIGHT', 'CAM_BACK',
            'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
        ]
        lidar_names = ['LIDAR_TOP']
        cam_infos = dict()
        lidar_infos = dict()

        # -------------------------------------------------
        mask_object_metadata = []
        object_status = []
        if use_gth_bbox:
            annotation_tokens = sample['anns']
            for annotation_token in annotation_tokens:
                annotation_metadata = nusc.get('sample_annotation', annotation_token)
                object_class = annotation_metadata['category_name'].split('.')[0]
                if object_class in valid_object_categories:
                    mask_object_metadata.append(annotation_metadata)
                    attribute_token = annotation_metadata['attribute_tokens']
                    if len(attribute_token) == 0:
                        if object_class == 'vehicle':
                            attribute = 'vehicle.moving'
                        elif object_class == 'human':
                            attribute = 'pedestrian.moving'
                        elif object_class == 'cycle':
                            attribute = 'cycle.with_rider'
                        else:
                            raise ValueError("Invalid object attribute, check valid_object_categories")
                    else:
                        attribute_metadata = nusc.get('attribute', attribute_token[0])
                        attribute = attribute_metadata['name']
                    object_status.append(attribute)
                    # if attribute == 'vehicle.moving':
                    #     vehicle_moving_metadata.append(annotation_metadata)
        else:
            annotation_filepath = os.path.join(annotation_folder, sample_token + '.pkl')
            with open(annotation_filepath, 'rb') as f:
                annotation_metadatas = pickle.load(f)
            for annotation_metadata in annotation_metadatas:
                object_class = annotation_metadata['category_name'].split('.')[0]
                if object_class in valid_object_categories:
                    mask_object_metadata.append(annotation_metadata)
                    if object_class == 'vehicle':
                        attribute = 'vehicle.moving'
                    elif object_class == 'human':
                        attribute = 'pedestrian.moving'
                    elif object_class == 'cycle':
                        attribute = 'cycle.with_rider'
                    else:
                        raise ValueError("Invalid object attribute, check valid_object_categories")
                    object_status.append(attribute)
        info['annotation'] = mask_object_metadata
        info['object_status'] = object_status

        for cam_name in cam_names:
            cam_data = nusc.get('sample_data', sample['data'][cam_name])
            cam_info = dict()
            cam_info['sample_token'] = cam_data['sample_token']
            cam_info['ego_pose'] = nusc.get('ego_pose', cam_data['ego_pose_token'])
            cam_info['filename'] = cam_data['filename']
            cam_info['calibrated_sensor'] = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            cam_infos[cam_name] = cam_info
        for lidar_name in lidar_names:
            lidar_data = nusc.get('sample_data', sample['data'][lidar_name])
            lidar_info = dict()
            lidar_info['sample_token'] = lidar_data['sample_token']
            lidar_info['ego_pose'] = nusc.get('ego_pose', lidar_data['ego_pose_token'])
            lidar_info['filename'] = lidar_data['filename']
            lidar_info['calibrated_sensor'] = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            lidar_infos[lidar_name] = lidar_info
        info['cam_infos'] = cam_infos
        info['lidar_infos'] = lidar_infos
        infos.append(info)
        idx = idx + 1
    return infos


def get_location_sample_tokens(nusc, location):
    # select scenes sampled in specific locations
    location_indices = get_location_indices(nusc, location)
    sample_token_list = []
    # get sequential sample tokens
    for scene_index in location_indices:
        scene = nusc.scene[scene_index]
        sample_token = scene['first_sample_token']
        while not sample_token == '':
            sample = nusc.get('sample', sample_token)
            sample_token_list.append(sample_token)
            sample_token = sample['next']
    return sample_token_list


def get_location_indices(nusc, location):
    location_indices = []
    for scene_index in range(len(nusc.scene)):
        scene = nusc.scene[scene_index]
        sample = nusc.get('sample', scene['last_sample_token'])
        if nusc.get('log', scene['log_token'])['location'] != location:
            continue
        location_indices.append(scene_index)
    return np.array(location_indices)


def coord_aligner(sample_info_seq, base_dataset_path, use_gth_bbox_3d):
    cam_names = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
    ]
    mid_frame = int((len(sample_info_seq) - 1) / 2)
    W2G = transform_matrix(
        sample_info_seq[mid_frame]['lidar_infos']['LIDAR_TOP']['ego_pose']['translation'],
        Quaternion(sample_info_seq[mid_frame]['lidar_infos']['LIDAR_TOP']['ego_pose']['rotation']),
        inverse=True
    )

    for cam_name in cam_names:
        for i in range(len(sample_info_seq)):
            cam_pose_in_w = dict()
            cur_info = sample_info_seq[i]
            G2E = transform_matrix(
                cur_info['cam_infos'][cam_name]['ego_pose']['translation'],
                Quaternion(cur_info['cam_infos'][cam_name]['ego_pose']['rotation']),
                inverse=False
            )
            E2C = transform_matrix(
                cur_info['cam_infos'][cam_name]['calibrated_sensor']['translation'],
                Quaternion(cur_info['cam_infos'][cam_name]['calibrated_sensor']['rotation']),
                inverse=False
            )
            G2C = np.dot(G2E, E2C)
            W2C = np.dot(W2G, G2C)
            C2W = np.linalg.inv(W2C)
            C2W_translation = C2W[:3, 3]
            C2W_quartenion = Quaternion(matrix_to_quaternion(C2W[:3, :3]))
            cam_pose_in_w['translation'] = C2W_translation
            cam_pose_in_w['rotation'] = C2W_quartenion
            cur_info['cam_infos'][cam_name]['world_cam_pose'] = cam_pose_in_w

    W2L_list = list()
    for i in range(len(sample_info_seq)):
        cur_info = sample_info_seq[i]
        cur_pcd = load_lidar_data(base_dataset_path, cur_info)
        ones_array = np.ones([cur_pcd.shape[0], 1])
        cur_pcd_ones = np.concatenate((cur_pcd, ones_array), axis=1)
        G2E = transform_matrix(
            cur_info['lidar_infos']['LIDAR_TOP']['ego_pose']['translation'],
            Quaternion(cur_info['lidar_infos']['LIDAR_TOP']['ego_pose']['rotation']),
            inverse=False
        )
        E2L = transform_matrix(
            cur_info['lidar_infos']['LIDAR_TOP']['calibrated_sensor']['translation'],
            Quaternion(cur_info['lidar_infos']['LIDAR_TOP']['calibrated_sensor']['rotation']),
            inverse=False
        )
        G2L = np.dot(G2E, E2L)
        W2L = np.dot(W2G, G2L)
        W2L_list.append(W2L)
        cur_pcd_in_w = W2L.dot(cur_pcd_ones.T).T[:, :3]
        cur_info['lidar_infos']['LIDAR_TOP']['world_pcd'] = cur_pcd_in_w

    # extract bbox vertices in world coordinate
    for i in range(len(sample_info_seq)):
        vertices_list = []
        vertices_crop_list = []
        cur_info = sample_info_seq[i]
        for annotation_metadata in cur_info['annotation']:
            bbox_size = annotation_metadata['size']
            if use_gth_bbox_3d:
                G2B = transform_matrix(
                    np.array(annotation_metadata['translation']),
                    Quaternion(annotation_metadata['rotation']),
                    inverse=False
                )

                vertices = get_bbox_vertice(bbox_size)
                ones_array = np.ones([vertices.shape[0], 1])
                vertices_ones = np.concatenate((vertices, ones_array), axis=1)
                vertices_in_g_ones = G2B.dot(vertices_ones.T).T
                vertices_in_w = W2G.dot(vertices_in_g_ones.T).T[:, :3]
                vertices_list.append(vertices_in_w)

                vertices_crop = get_bbox_vertice(np.array(bbox_size) * 1.3)
                ones_array = np.ones([vertices_crop.shape[0], 1])
                vertices_crop_ones = np.concatenate((vertices_crop, ones_array), axis=1)
                vertices_crop_in_g_ones = G2B.dot(vertices_crop_ones.T).T
                vertices_crop_in_w = W2G.dot(vertices_crop_in_g_ones.T).T[:, :3]
                vertices_crop_list.append(vertices_crop_in_w)
            else:
                vertices_in_l = annotation_metadata['corner']
                ones_array = np.ones([vertices_in_l.shape[0], 1])
                vertices_in_l_ones = np.concatenate((vertices_in_l, ones_array), axis=1)
                vertices_in_w = W2L_list[i].dot(vertices_in_l_ones.T).T[:, :3]
                vertices_list.append(vertices_in_w)

                center = np.mean(vertices_in_l, axis=0)
                vertices_crop_in_l = (vertices_in_l - center) * 1.3 + center
                ones_array = np.ones([vertices_crop_in_l.shape[0], 1])
                vertices_crop_in_l_ones = np.concatenate((vertices_crop_in_l, ones_array), axis=1)
                vertices_crop_in_w = W2L_list[i].dot(vertices_crop_in_l_ones.T).T[:, :3]
                vertices_crop_list.append(vertices_crop_in_w)

        cur_info['bbox_attribute'] = vertices_list
        cur_info['bbox_attribute_crop'] = vertices_crop_list

    return sample_info_seq
