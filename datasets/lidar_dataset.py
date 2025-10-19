import os
import csv
import pickle
import numpy as np
import open3d
import cv2
import shutil
from tqdm import tqdm
from PIL import Image
from nuscenes.nuscenes import NuScenes, transform_matrix, Quaternion
from read_scene_info import coord_aligner
from utils import read_image, gen_distant_hemisphere_points, get_cloud_radius


def from_nusc_to_lidar_color(sample_info_seq, base_dataset_root, target_base_dataset_root, semantic_label_folder, use_gth_bbox_3d):
    sample_filename = os.path.basename(cur_info['sample_token'])
    target_dataset_root = os.path.join(target_base_dataset_root, sample_filename)

    if not os.path.exists(target_dataset_root):
        os.makedirs(target_dataset_root)
    img_root = os.path.join(str(target_dataset_root), "images")
    if not os.path.exists(img_root):
        os.makedirs(img_root)
    mask_root = os.path.join(str(target_dataset_root), "masks")
    if not os.path.exists(mask_root):
        os.makedirs(mask_root)
    sparse_root = os.path.join(str(target_dataset_root), "sparse", "0")
    if not os.path.exists(sparse_root):
        os.makedirs(sparse_root)
    intrinsics_path = os.path.join(sparse_root, "cameras.txt")
    extrinsics_path = os.path.join(sparse_root, "images.txt")
    pcd_path = os.path.join(sparse_root, "points3D.txt")

    if not os.path.exists(semantic_label_folder):
        raise FileNotFoundError("Semantic label folder does not exist.")

    sample_info_seq = coord_aligner(sample_info_seq, base_dataset_root, use_gth_bbox_3d)

    # intrinsics
    write_cameras_text(sample_info_seq[0], intrinsics_path)

    # extrinsics
    write_images_text(sample_info_seq, extrinsics_path)

    # pointcloud
    write_points3D_text(sample_info_seq, base_dataset_root, pcd_path)

    # imgs and masks
    write_masked_imgs_and_dyn_mask(sample_info_seq, base_dataset_root, img_root, mask_root, semantic_label_folder)
    # copy_imgs(sample_info_seq, base_dataset_root, img_root)


def copy_imgs(sample_infos, base_img_root, target_img_root):
    cam_names = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
    ]
    for i in range(len(sample_infos)):
        cur_info = sample_infos[i]
        for cam_name in cam_names:
            img_filepath = os.path.join(base_img_root, cur_info['cam_infos'][cam_name]['filename'])
            img_basename = os.path.basename(cur_info['cam_infos'][cam_name]['filename'])
            img_tgt_filepath = os.path.join(target_img_root, img_basename)
            shutil.copy(str(img_filepath), str(img_tgt_filepath))


def write_masked_imgs_and_dyn_mask(sample_infos, base_img_root, target_img_root, target_mask_root, semantic_mask_folder):
    cam_names = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
    ]
    semantic_label = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain',
        'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
        'motorcycle', 'bicycle'
    ]
    vehicle_mask = ['car', 'truck', 'bus', 'train']
    ped_mask = ['person']
    cycle_mask = ['motorcycle', 'bicycle']
    cycle_with_rider_mask = ['motorcycle', 'bicycle', 'rider', 'person']
    static_mask_status = [
        'vehicle.parked', 'vehicle.stopped',
        'pedestrian.sitting_lying_down', 'pedestrian.standing',
        'cycle.without_rider'
    ]
    dynamic_mask_status = [
        'vehicle.moving', 'pedestrian.moving', 'cycle.with_rider'
    ]
    env_mask = ['sky', 'road']
    img_width = 1600
    img_height = 900

    cam_intrinsics = []
    for cam_name in cam_names:
        cur_intrinsic = sample_infos[0]['cam_infos'][cam_name]['calibrated_sensor']['camera_intrinsic']
        cam_intrinsics.append(np.array(cur_intrinsic))

    for i in range(len(sample_infos)):
        cam_idx = 0
        cur_info = sample_infos[i]
        for cam_name in cam_names:
            sem_seg_basename = os.path.basename(cur_info['cam_infos'][cam_name]['filename']) + '.npy'
            sem_seg_filepath = os.path.join(semantic_mask_folder, sem_seg_basename)
            if not os.path.exists(sem_seg_filepath):
                raise FileNotFoundError("Semantic label file does not exist.", sem_seg_filepath)
            # (900, 1600) to (1600, 900)
            sem_seg = np.load(sem_seg_filepath)
            sem_seg = sem_seg.transpose()

            # initialize static mask
            valid_static_binary_mask = np.ones_like(sem_seg, dtype=int)
            # initialize dynamic mask
            mask_basename = os.path.basename(cur_info['cam_infos'][cam_name]['filename'])
            mask_filepath = os.path.join(target_mask_root, mask_basename)
            mask = np.ones((1600, 900), dtype=int)

            # environment mask
            env_masked_label_list = list()
            for class_name_idx in range(len(semantic_label)):
                if semantic_label[class_name_idx] in env_mask:
                    env_masked_label_list.append(class_name_idx)
            for env_masked_label_idx in env_masked_label_list:
                current_valid_mask = (sem_seg != env_masked_label_idx)
                valid_static_binary_mask = np.multiply(valid_static_binary_mask, current_valid_mask)

            # object mask
            C2W = transform_matrix(
                cur_info['cam_infos'][cam_name]['world_cam_pose']['translation'],
                Quaternion(cur_info['cam_infos'][cam_name]['world_cam_pose']['rotation']),
                inverse=False
            )
            for bbox_vertices_in_w, bbox_status in zip(cur_info['bbox_attribute'], cur_info['object_status']):
                # static mask
                if bbox_status not in static_mask_status and bbox_status not in dynamic_mask_status and bbox_status != []:
                    continue
                # bbox visibility judgement
                bbox_vertices_in_w = np.array(bbox_vertices_in_w)
                ones_array = np.ones([bbox_vertices_in_w.shape[0], 1])
                bbox_vertices_in_w_ones = np.concatenate((bbox_vertices_in_w, ones_array), axis=1)
                bbox_vertices_in_c = C2W.dot(bbox_vertices_in_w_ones.T).T[:, :3]
                bbox_vertice_angle = np.arctan2(bbox_vertices_in_c[:, 0], bbox_vertices_in_c[:, 2]) * 180. / np.pi
                if cam_name == 'CAM_BACK':
                    visible_flag = np.sum((bbox_vertice_angle < 55.0) & (bbox_vertice_angle > -55.0))
                else:
                    visible_flag = np.sum((bbox_vertice_angle < 35.0) & (bbox_vertice_angle > -35.0))
                if not visible_flag:
                    continue

                # 2d bbox vertices in pixel coordinate
                uv_points = cam_intrinsics[cam_idx].dot(bbox_vertices_in_c.T).T
                uv_points_norm = uv_points[:, :2] / uv_points[:, 2].reshape(-1, 1)
                upper_left_w = min(uv_points_norm[:, 0])
                upper_left_h = min(uv_points_norm[:, 1])
                lower_right_w = max(uv_points_norm[:, 0])
                lower_right_h = max(uv_points_norm[:, 1])
                if upper_left_w > img_width or upper_left_h > img_height or lower_right_w < 0 or lower_right_h < 0:
                    continue
                upper_left_w = int(np.floor(max(0, upper_left_w)))
                upper_left_h = int(np.floor(max(0, upper_left_h)))
                lower_right_w = int(np.floor(min(img_width, lower_right_w)))
                lower_right_h = int(np.floor(min(img_height, lower_right_h)))

                # define valid semantic categories
                if bbox_status in static_mask_status[0:2]:
                    valid_sem_category = vehicle_mask
                elif bbox_status in static_mask_status[2:4]:
                    valid_sem_category = ped_mask
                elif bbox_status in static_mask_status[4:]:
                    valid_sem_category = cycle_mask
                elif bbox_status in dynamic_mask_status[0]:
                    valid_sem_category = vehicle_mask
                elif bbox_status in dynamic_mask_status[1]:
                    valid_sem_category = ped_mask
                elif bbox_status in dynamic_mask_status[2]:
                    valid_sem_category = cycle_with_rider_mask
                else:
                    valid_sem_category = None
                valid_sem_idx = []
                for class_name_idx in range(len(semantic_label)):
                    if semantic_label[class_name_idx] in valid_sem_category:
                        valid_sem_idx.append(class_name_idx)

                # mask valid semantic pixel positions
                semantic_label_in_bbox = sem_seg[upper_left_w:lower_right_w, upper_left_h:lower_right_h]
                valid_mask_in_bbox = np.ones_like(semantic_label_in_bbox, dtype=int)
                for valid_idx in valid_sem_idx:
                    valid_mask_in_bbox[semantic_label_in_bbox == valid_idx] = 0
                valid_mask_full_img = np.ones_like(valid_static_binary_mask)
                valid_mask_full_img[upper_left_w:lower_right_w, upper_left_h:lower_right_h] = valid_mask_in_bbox
                if bbox_status in static_mask_status or bbox_status == []:
                    valid_static_binary_mask = np.multiply(valid_static_binary_mask, valid_mask_full_img)
                elif bbox_status in dynamic_mask_status:
                    mask = np.multiply(mask, valid_mask_full_img)

            # load and save img
            img_filepath = os.path.join(base_img_root, cur_info['cam_infos'][cam_name]['filename'])
            img_basename = os.path.basename(cur_info['cam_infos'][cam_name]['filename'])
            img_tgt_filepath = os.path.join(target_img_root, img_basename)
            img = read_image(img_filepath, format="BGR")
            valid_static_binary_mask = valid_static_binary_mask.transpose()
            static_binary_mask = np.expand_dims(valid_static_binary_mask.astype(img.dtype), axis=2)
            rgb_mask = np.concatenate([static_binary_mask, static_binary_mask, static_binary_mask], axis=2)
            masked_img = np.multiply(img, rgb_mask)
            masked_img = masked_img[:, :, ::-1]
            masked_image_PIL = Image.fromarray(masked_img)
            # masked_image_PIL.save(img_tgt_filepath, quality=100)
            masked_image_PIL.save(img_tgt_filepath)

            # save dynamic mask
            if cam_name == 'CAM_BACK':
                identity_mask_filepath = '/home/octane17/3dgs/scripts/datasets/CAM_BACK.png'
                identity_mask = Image.open(identity_mask_filepath)
                identity_mask = np.array(identity_mask, dtype=mask.dtype)[:, :, 0].transpose() / 255
                mask = np.multiply(mask, identity_mask)
            mask = mask.transpose()
            mask = np.array(mask, dtype=np.uint8) * 255
            mask_PIL = Image.fromarray(mask)
            # mask_PIL.save(mask_filepath, quality=100)
            mask_PIL.save(mask_filepath)

            cam_idx = cam_idx + 1


def write_cameras_text(sample_info, path):
    cam_names = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
    ]
    HEADER = (
        "# Camera list with one line of data per camera:\n"
        + "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        + "# Number of cameras: {}\n".format(len(cam_names))
    )
    with open(path, "w") as fid:
        fid.write(HEADER)
        cam_id = 1
        for cam_name in cam_names:
            cur_intrinsic = sample_info['cam_infos'][cam_name]['calibrated_sensor']['camera_intrinsic']
            to_write = [cam_id, 'PINHOLE', 1600, 900,
                        cur_intrinsic[0][0], cur_intrinsic[1][1],
                        cur_intrinsic[0][2], cur_intrinsic[1][2]]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")
            cam_id = cam_id + 1


def write_images_text(sample_infos, txt_path):
    scaling_coeff = 1.0
    cam_names = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
    ]
    HEADER = (
        "# Image list with two lines of data per image:\n"
        + "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        + "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        + "# Number of images: {}, mean observations per image: {}\n".format(
            len(sample_infos) * len(cam_names), 0
        )
    )
    with open(txt_path, "w") as fid:
        fid.write(HEADER)
        img_id = 1
        cam_id = 1
        for cam_name in cam_names:
            for i in range(len(sample_infos)):
                cur_info = sample_infos[i]
                C2W_translation = cur_info['cam_infos'][cam_name]['world_cam_pose']['translation']
                C2W_quartenion = cur_info['cam_infos'][cam_name]['world_cam_pose']['rotation']
                img_name = str(cam_id) + '/' + str(os.path.basename(cur_info['cam_infos'][cam_name]['filename']))
                image_header = [
                    img_id,
                    C2W_quartenion[0], C2W_quartenion[1], C2W_quartenion[2], C2W_quartenion[3],
                    C2W_translation[0] / scaling_coeff, C2W_translation[1] / scaling_coeff, C2W_translation[2] / scaling_coeff,
                    cam_id,
                    img_name,
                ]
                img_id = img_id + 1
                first_line = " ".join(map(str, image_header))
                fid.write(first_line + "\n")
                fid.write(" " + "\n")
            cam_id = cam_id + 1


def write_points3D_text(sample_infos, base_dataset_path, path):
    cam_names = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
    ]
    HEADER = (
        "# 3D point list with one line of data per point:\n"
        + "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        + "# Number of points: {}, mean track length: {}\n".format(
            0, 0
        )
    )
    with open(path, "w") as fid:
        fid.write(HEADER)
        pcd_list = []
        for i in range(len(sample_infos)):
            cur_info = sample_infos[i]
            cur_pcd_in_w = cur_info['lidar_infos']['LIDAR_TOP']['world_pcd']

            if len(cur_info['bbox_attribute_crop']) != 0:
                static_bg_idx = inside_test(cur_pcd_in_w, cur_info['bbox_attribute_crop'])
                cur_pcd_in_w = cur_pcd_in_w[static_bg_idx]

            pcd_list.append(cur_pcd_in_w)
        full_pcd = np.vstack(pcd_list)
        ground_mask = full_pcd[:, 2] > 0.3
        full_pcd = full_pcd[ground_mask]

        # gen distant hemisphere
        cloud_radius = get_cloud_radius(full_pcd)
        hemisphere_points = gen_distant_hemisphere_points(cloud_radius)

        # get corresponding color
        corresponding_rgb_pcd, corresponding_rgb_hemisphere, valid_sphere_mask = find_corresponding_color(
            cur_info,
            full_pcd,
            hemisphere_points,
            cam_names,
            base_dataset_path
        )
        full_pcd = np.hstack([full_pcd, corresponding_rgb_pcd])
        hemisphere_points = np.hstack([hemisphere_points, corresponding_rgb_hemisphere])
        hemisphere_points = hemisphere_points[valid_sphere_mask]
        full_pcd = np.concatenate([full_pcd, hemisphere_points], axis=0)
        np.random.shuffle(full_pcd)

        full_o3d_pcd = open3d.geometry.PointCloud()
        full_o3d_pcd.points = open3d.utility.Vector3dVector(full_pcd[:, :3])
        full_o3d_pcd.colors = open3d.utility.Vector3dVector(full_pcd[:, 3:])
        full_o3d_pcd_dsp = full_o3d_pcd.voxel_down_sample(voxel_size=0.1)
        dsp_points = np.array(full_o3d_pcd_dsp.points)
        dsp_colors = np.array(full_o3d_pcd_dsp.colors)
        full_pcd_dsp = np.hstack([dsp_points, dsp_colors])

        # point_cloud0 = open3d.geometry.PointCloud()
        # point_cloud0.points = open3d.utility.Vector3dVector(full_pcd_dsp[:, :3])
        # # point_cloud0.colors = open3d.utility.Vector3dVector(full_pcd_dsp[:, 3:] / 255.0)
        # vis = open3d.visualization.Visualizer()
        # vis.create_window()
        # opt = vis.get_render_option()
        # opt.point_size = 3
        #
        # edges = np.array([
        #     [0, 1], [1, 2], [2, 3], [3, 0],
        #     [4, 5], [5, 6], [6, 7], [7, 4],
        #     [0, 4], [1, 5], [2, 6], [3, 7]
        # ])
        # for vertices in cur_info['bbox_attribute_crop']:
        #     line_set = open3d.geometry.LineSet()
        #     line_set.points = open3d.utility.Vector3dVector(vertices)
        #     line_set.lines = open3d.utility.Vector2iVector(edges)
        #     line_set.paint_uniform_color([1, 0, 0])
        #     vis.add_geometry(line_set)
        #
        # vis.add_geometry(point_cloud0)
        # vis.run()
        # vis.destroy_window()

        for i in range(full_pcd_dsp.shape[0]):
            # set RGB and error to 0
            point_header = [i+1, full_pcd_dsp[i, 0], full_pcd_dsp[i, 1], full_pcd_dsp[i, 2],
                            int(full_pcd_dsp[i, 3]), int(full_pcd_dsp[i, 4]), int(full_pcd_dsp[i, 5]), 0]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            fid.write(" ".join(track_strings) + "\n")


def find_corresponding_color(scene_info, pcd_in_w, sphere_in_w, cam_names, base_dataset_path):
    img_width = 1600
    img_height = 900

    ones_array = np.ones([pcd_in_w.shape[0], 1])
    pcd_in_w_ones = np.concatenate((pcd_in_w, ones_array), axis=1)
    ones_array = np.ones([sphere_in_w.shape[0], 1])
    sphere_in_w_ones = np.concatenate((sphere_in_w, ones_array), axis=1)
    C2Ws = []
    for cam_name in cam_names:
        C2W = transform_matrix(
            scene_info['cam_infos'][cam_name]['world_cam_pose']['translation'],
            Quaternion(scene_info['cam_infos'][cam_name]['world_cam_pose']['rotation']),
            inverse=False
        )
        C2Ws.append(C2W)
    cam_intrinsics = []
    for cam_name in cam_names:
        cur_intrinsic = scene_info['cam_infos'][cam_name]['calibrated_sensor']['camera_intrinsic']
        cam_intrinsics.append(np.array(cur_intrinsic))

    cam_idx = 0
    rgb_colors_pcd = np.zeros_like(pcd_in_w_ones[:, :3], dtype=pcd_in_w_ones.dtype)
    projected_cam_pixel_list = []
    for cam_name in cam_names:
        projected_cam_pixel = np.zeros((img_width, img_height))

        # transform point cloud to pixel coordinate
        pcd_ones_in_cam = C2Ws[cam_idx].dot(pcd_in_w_ones.T).T
        pcd_in_cam = pcd_ones_in_cam[:, :3]
        uv_points = cam_intrinsics[cam_idx].dot(pcd_in_cam.T).T
        uv_points_norm = uv_points[:, :2] / uv_points[:, 2].reshape(-1, 1)

        point_angle = np.arctan2(pcd_in_cam[:, 0], pcd_in_cam[:, 2]) * 180. / np.pi
        if cam_name == 'CAM_BACK':
            visible_flag = (point_angle < 55.0) & (point_angle > -55.0)
        else:
            visible_flag = (point_angle < 35.0) & (point_angle > -35.0)

        img_filepath = os.path.join(base_dataset_path, scene_info['cam_infos'][cam_name]['filename'])
        image = cv2.imread(str(img_filepath))
        h, w, _ = image.shape
        for i in range(uv_points_norm.shape[0]):
            if not visible_flag[i]:
                continue
            x = uv_points_norm[i][0]
            y = uv_points_norm[i][1]
            if 0 <= x < w and 0 <= y < h:
                x = int(x)
                y = int(y)
                bgr_color = image[y, x]
                rgb_colors_pcd[i][0] = bgr_color[2]
                rgb_colors_pcd[i][1] = bgr_color[1]
                rgb_colors_pcd[i][2] = bgr_color[0]

                mask_box_len_w = 20
                mask_box_len_h = 20
                upper_left_w = int(np.floor(max(0, x - mask_box_len_w)))
                upper_left_h = int(np.floor(max(0, y - mask_box_len_h)))
                lower_right_w = int(np.floor(min(img_width, x + mask_box_len_w)))
                lower_right_h = int(np.floor(min(img_height, y + mask_box_len_h)))
                projected_cam_pixel[upper_left_w:lower_right_w, upper_left_h:lower_right_h] = 1

        projected_cam_pixel_list.append(projected_cam_pixel)
        cam_idx = cam_idx + 1

    cam_idx = 0
    rgb_colors_sphere = np.zeros_like(sphere_in_w_ones[:, :3], dtype=sphere_in_w_ones.dtype)
    invalid_sphere_points_mask = np.zeros_like(sphere_in_w_ones[:, 0], dtype=int)
    for cam_name in cam_names:
        # transform point cloud to pixel coordinate
        sphere_ones_in_cam = C2Ws[cam_idx].dot(sphere_in_w_ones.T).T
        sphere_in_cam = sphere_ones_in_cam[:, :3]
        uv_points = cam_intrinsics[cam_idx].dot(sphere_in_cam.T).T
        uv_points_norm = uv_points[:, :2] / uv_points[:, 2].reshape(-1, 1)

        point_angle = np.arctan2(sphere_in_cam[:, 0], sphere_in_cam[:, 2]) * 180. / np.pi
        if cam_name == 'CAM_BACK':
            visible_flag = (point_angle < 55.0) & (point_angle > -55.0)
        else:
            visible_flag = (point_angle < 35.0) & (point_angle > -35.0)

        img_filepath = os.path.join(base_dataset_path, scene_info['cam_infos'][cam_name]['filename'])
        image = cv2.imread(str(img_filepath))
        h, w, _ = image.shape
        for i in range(uv_points_norm.shape[0]):
            if not visible_flag[i]:
                continue
            x = uv_points_norm[i][0]
            y = uv_points_norm[i][1]
            if 0 <= x < w and 0 <= y < h:
                x = int(x)
                y = int(y)
                if projected_cam_pixel_list[cam_idx][x][y] == 1:
                    invalid_sphere_points_mask[i] = 1
                    continue
                bgr_color = image[int(y), int(x)]
                rgb_colors_sphere[i][0] = bgr_color[2]
                rgb_colors_sphere[i][1] = bgr_color[1]
                rgb_colors_sphere[i][2] = bgr_color[0]

        cam_idx = cam_idx + 1

    valid_sphere_points_mask = (invalid_sphere_points_mask == 0)

    return rgb_colors_pcd, rgb_colors_sphere, valid_sphere_points_mask


# https://stackoverflow.com/questions/21037241/how-to-determine-a-point-is-inside-or-outside-a-cube
def inside_test(points, cube3d_list):
    """
    cube3d  =  numpy array of the shape (8,3) with coordinates in the clockwise order. first the bottom plane is considered then the top one.
    points = array of points with shape (N, 3).

    Returns the indices of the points array which are outside the cube3d
    """
    outside_idx_list = []

    for cube3d in cube3d_list:
        b1, b2, b3, b4, t1, t2, t3, t4 = cube3d

        dir1 = (t1 - b1)
        size1 = np.linalg.norm(dir1)
        dir1 = dir1 / size1

        dir2 = (b2 - b1)
        size2 = np.linalg.norm(dir2)
        dir2 = dir2 / size2

        dir3 = (b4 - b1)
        size3 = np.linalg.norm(dir3)
        dir3 = dir3 / size3

        cube3d_center = (b1 + t3) / 2.0

        dir_vec = points - cube3d_center

        res1 = np.where((np.absolute(np.dot(dir_vec, dir1)) * 2) > size1)[0]
        res2 = np.where((np.absolute(np.dot(dir_vec, dir2)) * 2) > size2)[0]
        res3 = np.where((np.absolute(np.dot(dir_vec, dir3)) * 2) > size3)[0]

        outside_idx = list(set().union(res1, res2, res3))
        outside_idx_list.append(outside_idx)

    common_elements_set = set(outside_idx_list[0])
    for lst in outside_idx_list[1:]:
        common_elements_set = common_elements_set.intersection(lst)
    common_elements = list(common_elements_set)

    return common_elements


if __name__ == '__main__':
    start_frame = 0
    use_gth_bbox_3d = True
    dataset_root = '/media/octane17/T7ShieldNus/GSPR/data'
    nusc_root = '/media/octane17/T7ShieldNus/NuScenes'
    db_ind_root = os.path.join(dataset_root, 'bs_db.npy')
    train_query_ind_root = os.path.join(dataset_root, 'bs_train_query.npy')
    val_query_ind_root = os.path.join(dataset_root, 'bs_val_query.npy')
    test_query_ind_root = os.path.join(dataset_root, 'bs_test_query.npy')
    gs_dataset_root = os.path.join(dataset_root, 'NuScenesGaussianDataset/BS')
    semantic_label_root = os.path.join(dataset_root, 'semantic_label_folder/semantic_labels_bs')

    infos_bs_path = os.path.join(dataset_root, 'nuscenes_infos-bs.pkl')
    with open(infos_bs_path, 'rb') as f:
        infos_bs = pickle.load(f)
    bs_db = np.load(db_ind_root)
    bs_train_query = np.load(train_query_ind_root)
    bs_val_query = np.load(val_query_ind_root)
    bs_test_query = np.load(test_query_ind_root)
    bs_whole = np.concatenate([bs_db, bs_train_query, bs_val_query, bs_test_query], axis=0)

    for i in range(start_frame, bs_whole.shape[0]):
        sample_info_seq = []
        cur_ind = bs_whole[i][0]
        cur_info = infos_bs[int(cur_ind)]

        token = cur_info['sample_token']
        print(f'---------------> Generating dataset of {i} th sequence: {token}')

        sample_info_seq.append(infos_bs[int(cur_ind) - 1])
        sample_info_seq.append(cur_info)
        sample_info_seq.append(infos_bs[int(cur_ind) + 1])

        from_nusc_to_lidar_color(sample_info_seq, nusc_root, gs_dataset_root, semantic_label_root, use_gth_bbox_3d)
