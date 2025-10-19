import os
import copy
from tqdm import tqdm
import numpy as np
from plyfile import PlyData
from datasets.voxel_generator import VoxelGenerator


def mean_vfe_numpy(voxel_features, num_per_voxel):
    points_sum = np.sum(voxel_features, axis=1)
    points_mean = points_sum / num_per_voxel[:, np.newaxis]
    return points_mean


def load_ply(path):
    input_ply_path = os.path.join(path, "point_cloud/iteration_400/point_cloud.ply")

    max_sh_degree = 3
    plydata = PlyData.read(input_ply_path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return xyz, features_dc, features_extra, opacities, scales, rots


def concatenate_gaussian(gaussian_tuple):
    xyz = gaussian_tuple[0]
    sh_coeffs = np.concatenate((gaussian_tuple[1], gaussian_tuple[2]), axis=2)
    sh_coeffs = sh_coeffs.reshape(sh_coeffs.shape[0], 1, -1).squeeze(1)
    opacities = gaussian_tuple[3]
    scales = gaussian_tuple[4]
    rots = gaussian_tuple[5]
    points = np.concatenate((xyz, scales, rots, sh_coeffs, opacities), axis=1)
    return points


def polar_quantizer(xyz):
    theta = 180. + np.arctan2(xyz[:, 1], xyz[:, 0]) * 180. / np.pi
    theta = np.where(theta == 360., theta - 0.0001, theta)
    dist = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    z = xyz[:, 2]
    coord = np.stack([theta, dist, z], axis=1)
    return coord


def from_polar_to_cartesian(tdz):
    x = np.cos((tdz[:, 0] - 180) * np.pi / 180) * tdz[:, 1]
    y = np.sin((tdz[:, 0] - 180) * np.pi / 180) * tdz[:, 1]
    z = tdz[:, 2]
    xyz = np.stack([x, y, z], axis=1)
    return xyz


def load_sub_directories(folder):
    sub_directories = []
    for sub_folder in os.listdir(folder):
        sub_folder_name = sub_folder
        sub_directories.append(sub_folder_name)
    return sub_directories


if __name__ == '__main__':
    gaussian_scene_root = "/media/octane17/T7ShieldNus/NuscenesGaussianModel/SON_Model"
    gaussian_dsp_root = "/media/octane17/T7ShieldNus/NuscenesGaussianModel/SON_4096_VFE"
    is_in_cartesian = True
    point_cloud_range = [0.0, 0.0, 0.0, 360.0, 120.0, 25.0]
    max_num_points = 20
    target_voxel_size = 8192
    max_voxel_size = 10000

    start_index = 0
    gaussian_scene_filename = load_sub_directories(gaussian_scene_root)
    for i in tqdm(range(start_index, len(gaussian_scene_filename))):
        # -------------------------- load gaussian scene --------------------------
        gaussian_scene_filepath = os.path.join(gaussian_scene_root, gaussian_scene_filename[i])
        gaussian_model = load_ply(gaussian_scene_filepath)
        points_with_feature = concatenate_gaussian(gaussian_model)
        points_polar = polar_quantizer(points_with_feature[:, :3])
        points_with_feature[:, :3] = points_polar

        # -------------------------- dynamic downsample --------------------------
        voxel_size = 1.0
        voxel_gen = VoxelGenerator(
            voxel_size=[voxel_size, voxel_size, voxel_size],
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxel_size,
            full_mean=False
        )
        voxels, coors, num_points_per_voxel = voxel_gen.generate(points_with_feature)
        original_voxels = copy.deepcopy(voxels)
        original_num_points_per_voxel = copy.deepcopy(num_points_per_voxel)
        while voxels.shape[0] < target_voxel_size:
            voxel_size = voxel_size - 0.05
            if voxel_size <= 0.05:
                print('Voxel Num Too Low, Invalid Voxel Size!')
                voxel_size = 1.0
                voxel_gen = VoxelGenerator(
                    voxel_size=[voxel_size, voxel_size, voxel_size],
                    point_cloud_range=point_cloud_range,
                    max_num_points=max_num_points,
                    max_voxels=max_voxel_size,
                    full_mean=False
                )
                voxels, coors, num_points_per_voxel = voxel_gen.generate(points_with_feature)
                break
            voxel_gen = VoxelGenerator(
                voxel_size=[voxel_size, voxel_size, voxel_size],
                point_cloud_range=point_cloud_range,
                max_num_points=max_num_points,
                max_voxels=max_voxel_size,
                full_mean=False
            )
            voxels, coors, num_points_per_voxel = voxel_gen.generate(points_with_feature)
        while voxels.shape[0] > target_voxel_size:
            voxel_size = voxel_size + 0.05
            if voxel_size >= 1.95:
                print('Voxel Num Too High!', voxels.shape[0])
                voxel_size = 1.95
                voxel_gen = VoxelGenerator(
                    voxel_size=[voxel_size, voxel_size, voxel_size],
                    point_cloud_range=point_cloud_range,
                    max_num_points=max_num_points,
                    max_voxels=max_voxel_size,
                    full_mean=False
                )
                voxels, coors, num_points_per_voxel = voxel_gen.generate(points_with_feature)
                voxels_ind = np.random.permutation(np.arange(0, voxels.shape[0]))[:target_voxel_size]
                voxels = voxels[voxels_ind]
                coors = coors[voxels_ind]
                num_points_per_voxel = num_points_per_voxel[voxels_ind]
                break
            voxel_gen = VoxelGenerator(
                voxel_size=[voxel_size, voxel_size, voxel_size],
                point_cloud_range=point_cloud_range,
                max_num_points=max_num_points,
                max_voxels=max_voxel_size,
                full_mean=False
            )
            voxels, coors, num_points_per_voxel = voxel_gen.generate(points_with_feature)

        # -------------------------- full voxel nums to target --------------------------
        print(float(np.sum(num_points_per_voxel)) / float(target_voxel_size), round(voxel_size, 3))
        num_extra_voxels = target_voxel_size - voxels.shape[0]
        if num_extra_voxels < original_voxels.shape[0]:
            extra_voxels_ind = np.random.permutation(np.arange(0, original_voxels.shape[0]))[:num_extra_voxels]
        else:
            extra_voxels_ind = np.random.choice(np.arange(0, original_voxels.shape[0]), size=num_extra_voxels, replace=True)
        extra_voxels = original_voxels[extra_voxels_ind, :, :]
        total_voxels = np.concatenate([voxels, extra_voxels], axis=0)
        extra_num_points_per_voxel = original_num_points_per_voxel[extra_voxels_ind]
        total_num_points_per_voxel = np.concatenate([num_points_per_voxel, extra_num_points_per_voxel], axis=0)
        shuffled_ind = np.random.permutation(np.arange(0, total_voxels.shape[0]))
        total_voxels = total_voxels[shuffled_ind, :, :]
        total_num_points_per_voxel = total_num_points_per_voxel[shuffled_ind]

        # select N voxels randomly
        select_ind = np.random.choice(np.arange(0, total_voxels.shape[0]), size=4096, replace=False)
        total_voxels = total_voxels[select_ind]
        total_num_points_per_voxel = total_num_points_per_voxel[select_ind]

        if is_in_cartesian:
            points = total_voxels[:, :, :3].reshape(-1, 3)
            points_cart = from_polar_to_cartesian(points)
            # points_cart = points_cart.reshape(target_voxel_size, max_num_points, 3)
            points_cart = points_cart.reshape(4096, max_num_points, 3)
            total_voxels[:, :, :3] = points_cart

        total_voxels_after_vfe = mean_vfe_numpy(total_voxels, total_num_points_per_voxel)

        # -------------------------- save downsampled gaussians --------------------------
        gaussian_dsp_filename = gaussian_scene_filename[i] + ".npy"
        gaussian_dsp_filepath = os.path.join(gaussian_dsp_root, gaussian_dsp_filename)
        np.save(gaussian_dsp_filepath, total_voxels_after_vfe)
