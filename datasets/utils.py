import os
import numpy as np
import torch
from PIL import Image


# https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_M_YUV2RGB = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]

# https://www.exiv2.org/tags.html
_EXIF_ORIENT = 274  # exif 'Orientation' tag


def convert_PIL_to_numpy(image, format):
    """
    Convert PIL image to numpy array of target format.

    Args:
        image (PIL.Image): a PIL image
        format (str): the format of output image

    Returns:
        (np.ndarray): also see `read_image`
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)

    # handle formats not supported by PIL
    elif format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    elif format == "YUV-BT.601":
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)

    return image


def convert_image_to_rgb(image, format):
    """
    Convert an image from given format to RGB.

    Args:
        image (np.ndarray or Tensor): an HWC image
        format (str): the format of input image, also see `read_image`

    Returns:
        (np.ndarray): (H,W,3) RGB image in 0-255 range, can be either float or uint8
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if format == "BGR":
        image = image[:, :, [2, 1, 0]]
    elif format == "YUV-BT.601":
        image = np.dot(image, np.array(_M_YUV2RGB).T)
        image = image * 255.0
    else:
        if format == "L":
            image = image[:, :, 0]
        image = image.astype(np.uint8)
        image = np.asarray(Image.fromarray(image, mode=format).convert("RGB"))
    return image


def _apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`

    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image


def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR" or "YUV-BT.601".

    Returns:
        image (np.ndarray):
            an HWC image in the given format, which is 0-255, uint8 for
            supported image modes in PIL or "BGR"; float (0-1 for Y) for YUV-BT.601.
    """
    with open(file_name, "rb") as f:
        image = Image.open(f)
        return convert_PIL_to_numpy(image, format)


def matrix_to_quaternion(matrix):
    qw = np.sqrt(1 + matrix[0, 0] + matrix[1, 1] + matrix[2, 2]) / 2
    qx = (matrix[2, 1] - matrix[1, 2]) / (4 * qw)
    qy = (matrix[0, 2] - matrix[2, 0]) / (4 * qw)
    qz = (matrix[1, 0] - matrix[0, 1]) / (4 * qw)
    return [qw, qx, qy, qz]


def load_lidar_data(base_dataset_path, cur_info):
    lidar_data = cur_info['lidar_infos']['LIDAR_TOP']
    fname = os.path.join(base_dataset_path, lidar_data['filename'])
    pc = np.frombuffer(open(fname, "rb").read(), dtype=np.float32)
    pc = np.array(pc.reshape(-1, 5)[:, :3])
    dist = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2)
    dist_mask = dist > 2.0
    pc = pc[dist_mask]
    return pc


def gen_distant_hemisphere_points(scene_radius, delta_theta=0.5, delta_phi=0.5):
    # NuScenes KITTI
    theta_values_deg = np.arange(80, 90, delta_theta)
    # KITTI-360
    # theta_values_deg = np.arange(-100, -90, delta_theta)
    # NuScenes
    phi_values_deg = np.arange(0, 360, delta_phi)
    # KITTI
    # phi_values_deg = np.arange(-50, 50, delta_phi)
    # KITTI-360
    # phi_values_deg = np.arange(130, 230, delta_phi)

    theta_mesh, phi_mesh = np.meshgrid(np.radians(theta_values_deg), np.radians(phi_values_deg))

    x = np.sin(theta_mesh) * np.cos(phi_mesh)
    y = np.sin(theta_mesh) * np.sin(phi_mesh)
    z = np.cos(theta_mesh)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)

    hemisphere_points = np.column_stack((x, y, z))
    hemisphere_points = hemisphere_points * scene_radius
    return hemisphere_points


def get_cloud_radius(lidar_points):
    center = np.mean(lidar_points, axis=0)
    max_diag_dists = -np.sort(-np.sqrt(np.sum(np.square(lidar_points - center), axis=1)))
    far_points_sample_num = int(max_diag_dists.shape[0] * 0.01)
    diagonal = np.mean(max_diag_dists[:far_points_sample_num])
    # scaling factor, prevent the hemisphere from pruning
    radius = diagonal * 1.2
    return radius


def get_bbox_vertice(bbox_size):
    # bbox_size: (,3) array or list
    length = bbox_size[1] * 1.0
    width = bbox_size[0] * 1.0
    height = bbox_size[2] * 1.0
    vertices = [
        (-length / 2, -width / 2, -height / 2),
        (length / 2, -width / 2, -height / 2),
        (length / 2, width / 2, -height / 2),
        (-length / 2, width / 2, -height / 2),
        (-length / 2, -width / 2, height / 2),
        (length / 2, -width / 2, height / 2),
        (length / 2, width / 2, height / 2),
        (-length / 2, width / 2, height / 2)
    ]
    vertices = np.array(vertices)
    return vertices
