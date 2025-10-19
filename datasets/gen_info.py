import os
import pickle
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from read_scene_info import get_location_sample_tokens, gen_info


def main():
    nuscroot = '/media/octane17/T7ShieldNus/NuScenes'
    dataroot = '/media/octane17/T7ShieldNus/GSPR/data'
    annotation_root = os.path.join(dataroot, 'pp_annotation')
    # True: use the ground truth annotations provided by nuscenes
    # False: use the pointpillars-based 3D annotations
    use_gth_bbox_3d = True
    nusc_trainval = NuScenes(version='v1.0-trainval', dataroot=nuscroot, verbose=True)

    # ====================generate infos====================
    sample_tokens_trainval = get_location_sample_tokens(nusc_trainval, location='boston-seaport')
    infos = gen_info(nusc_trainval, sample_tokens_trainval, use_gth_bbox_3d, annotation_root)
    with open(os.path.join(dataroot, 'nuscenes_infos-bs.pkl'), 'wb') as f:
        pickle.dump(infos, f)

    sample_tokens_trainval = get_location_sample_tokens(nusc_trainval, location='singapore-onenorth')
    infos = gen_info(nusc_trainval, sample_tokens_trainval, use_gth_bbox_3d, annotation_root)
    with open(os.path.join(dataroot, 'nuscenes_infos-son.pkl'), 'wb') as f:
        pickle.dump(infos, f)

    sample_tokens_trainval = get_location_sample_tokens(nusc_trainval, location='singapore-queenstown')
    infos = gen_info(nusc_trainval, sample_tokens_trainval, use_gth_bbox_3d, annotation_root)
    with open(os.path.join(dataroot, 'nuscenes_infos-sq.pkl'), 'wb') as f:
        pickle.dump(infos, f)


if __name__ == '__main__':
    main()
