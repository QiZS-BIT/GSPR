# GSPR
This repository is the official implementation of our paper accepted by IEEE IROS 2025. [[arxiv]](https://arxiv.org/abs/2410.00299).

**GSPR: Multimodal Place Recognition Using 3D Gaussian Splatting for Autonomous Driving**

[Zhangshuo Qi<sup>†</sup>](https://github.com/QiZS-BIT), [Junyi Ma<sup>†</sup>](https://github.com/BIT-MJY), [Jingyi Xu](https://github.com/BIT-XJY),
[Zijie Zhou](https://github.com/ZhouZijie77), [Luqi Cheng](https://github.com/ChengLuqi), [Guangming Xiong<sup>*</sup><be>](https://ieeexplore.ieee.org/author/37286205000)

![image](https://github.com/QiZS-BIT/GSPR/blob/main/pics/motivation.png)

GSPR is a novel multimodal place recognition network based on 3D Gaussian Splatting. It harmonizes multi-view camera and LiDAR data into a unified 
explicit scene representation through the proposed Multimodal Gaussian Splatting, and extracts spatio-temporally discriminative global descriptors
for query-database matching.<br>

## Table of Contents
- [GSPR](#GSPR)
    - [Installation](#installation)
    - [Data Preparation](#data-preparation)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Download](#download)
    - [Citation](#citation)
    - [Acknowledgement](#acknowledgement)

## Installation
* Ubuntu 20.04 + Python 3.7
* CUDA 11.3 + Pytorch 1.12.1
```
git clone https://github.com/QiZS-BIT/GSPR.git
cd GSPR
conda create -n gspr python=3.7
conda activate gspr
pip install -r requirements.txt
```

Additionally, a [spconv](https://github.com/traveller59/spconv)-based CPP plugin needs to be compiled. The command is as follows:
```
cd GSPR/cpp_plugins
mkdir build
cd build
cmake ..
make
```
Then, move the compiled file ``point2voxel.cpython-37m-x86_64-linux-gnu.so`` to the ``GSPR/datasets`` directory.



## Data Preparation
We use the nuScenes dataset as an example to introduce the data preparation process, 
which can be downloaded [here](https://www.nuscenes.org/nuscenes). In the following steps, 
we will assume ``/GSPR/data`` as the root directory for data storage.

### Semantic Labels
We first need to generate the semantic segmentation results for the nuScenes images.
* Configure [Mask2Former](https://github.com/facebookresearch/Mask2Former) as per the official instructions and download the corresponding weights (we suggest using ``cityscape_semantic_swin_s.pkl``).
* Copy file ``/GSPR/tools/gen_semantic_mask.py`` to the Mask2Former project directory.
* Run the ``/GSPR/datasets/gen_semantic_info.py`` script to generate the semantic segmentation metadata for BS, SON, SQ splits.
* Under the ``/GSPR/data`` directory, create a folder named ``semantic_label_folder``. Then, within this folder, create three subfolders: ``semantic_labels_bs``, ``semantic_labels_son``, and ``semantic_labels_sq``.
* Run the following commands to generate semantic labels for BS, SON, SQ splits, respectively.
```
python semantic_mask_demo.py --config-file ../configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_small_bs16_90k.yaml \
  --input /GSPR/data/semantic_info_bs.pkl \
  --output /GSPR/data/semantic_label_folder/semantic_labels_bs \
  --opts MODEL.WEIGHTS ../models/cityscape_semantic_swin_s.pkl
  
python semantic_mask_demo.py --config-file ../configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_small_bs16_90k.yaml \
  --input /GSPR/data/semantic_info_son.pkl \
  --output /GSPR/data/semantic_label_folder/semantic_labels_son \
  --opts MODEL.WEIGHTS ../models/cityscape_semantic_swin_s.pkl
  
python semantic_mask_demo.py --config-file ../configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_small_bs16_90k.yaml \
  --input /GSPR/data/semantic_info_sq.pkl \
  --output /GSPR/data/semantic_label_folder/semantic_labels_sq \
  --opts MODEL.WEIGHTS ../models/cityscape_semantic_swin_s.pkl
```

### 3D Annotations (Optional)
Although the ground-truth 3D bounding boxes from nuScenes can be used directly, 
we also provide a method for generating 3D annotations using [PointPillars](https://arxiv.org/abs/1812.05784).
* We use a lightweight implementation of PointPillars with a simple configuration. The code is available at [here](https://github.com/zhulf0804/PointPillars).
* Copy file ``/GSPR/tools/pointpillars_nusc_infer.py`` to the PointPillars project directory.
* Under the ``GSPR/data`` directory, create a folder named ``pp_annotations``.
* Run the script to generate 3D annotations.

### nuScenes Data Organizations
* Generate the infos and indexes using the following commands.
```
cd /GSPR/datasets
python gen_info.py
python gen_index.py
```

### Gaussian Scenes
* Under the ``GSPR/data`` directory, create a folder named ``NuScenesGaussianDataset``. Then, within this folder, create three subfolders: ``BS``, ``SON``, and ``SQ``.
* Run the ``/GSPR/datasets/lidar_dataset.py`` script to prepare training data for 3D-GS.
* Download [our modified version of 3D-GS](), and configure it following the [official instructions](https://github.com/graphdeco-inria/gaussian-splatting).
* Copy file ``/GSPR/tools/train_batch.py`` to the 3D-GS project directory.
* Under the ``/GSPR/data`` directory, create a folder named ``NuScenesGaussianModel``. Then, within this folder, create three subfolders: ``BS``, ``SON``, and ``SQ``.
* Run the ``/GSPR/tools/train_batch.py`` script to generate Gaussian scenes for BS, SON, SQ splits, respectively.
* Under the ``/GSPR/data/NuScenesGaussianModel`` folder, create three subfolders: ``BS_4096_VFE``, ``SON_4096_VFE``, and ``SQ_4096_VFE``.
* Run the ``/GSPR/datasets/gaussian_downsample.py`` script to generate downsampled Gaussian scenes  for BS, SON, SQ splits, respectively.

## Training
To train the model from scratch, you first need to modify ``config/params.py``. There are three main changes to pay attention to:
* Ensure that the file paths used for training are kept, while other file paths are commented out.
* Set ``self.checkpoint_path = ""`` and configure ``self.resume_checkpoint = False``.
* Set ``self.training_root`` to the desired path for storing training files.
Then, run the following script to train the model:
```
python train.py
```

## Evaluation
You can either evaluate our provided pre-trained weights, or evaluate results obtained from training locally.

First, ensure that in ``config/params.py``, the file paths 
used for evaluating specific sequences are kept, while other file paths are commented out.

Then proceed with one of the following options:

**Option 1: Evaluate Pre-trained Weights**
* Set ``self.checkpoint_path`` to the path of our provided pre-trained weights.

**Option 2: Evaluate Locally Trained Results**
* Set ``self.checkpoint_path = ""``
* Set ``self.training_root`` to the folder where training results are stored.
* Set ``self.resume_epoch`` to the epoch corresponding to the weights you want to evaluate.

After setting ``self.resume_checkpoint = True``, run the following scripts depending on which dataset you want to evaluate:
```
python test.py
```
You can switch between GSPR (better performance) and GSPR-L (faster inference speed) by commenting out line 39 in ``/GSPR/datasets/nuscenes_dataset.py``.

## Download
* Our pre-trained weights are available at this [link]().

## Citation
If you find this project useful for your research, please consider citing:
```
@article{qi2024gspr,
  title={GSPR: Multimodal Place Recognition using 3D Gaussian Splatting for Autonomous Driving},
  author={Qi, Zhangshuo and Ma, Junyi and Xu, Jingyi and Zhou, Zijie and Cheng, Luqi and Xiong, Guangming},
  journal={arXiv preprint arXiv:2410.00299},
  year={2024}
}
```

## Acknowledgement
Many thanks to these excellent projects:
* [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
* [LCPR](https://github.com/ZhouZijie77/LCPR)
* [AutoPlace](https://github.com/ramdrop/autoplace)
* [3D-GCN](https://github.com/chih-hao-lin/3dgcn)
* [Mask2Former](https://github.com/facebookresearch/Mask2Former)

