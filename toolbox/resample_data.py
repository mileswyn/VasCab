# 对初始数据进行裁剪和重采样

from collections import OrderedDict
from copy import deepcopy

import shutil
import SimpleITK as sitk
import logging

from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.augmentations.utils import resize_segmentation
from nnunet.configuration import default_num_threads, RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD
from nnunet.preprocessing.cropping import get_case_identifier_from_npz, ImageCropper
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing.pool import Pool

def get_bbox_from_mask(mask):
    """
    从3D mask提取包含mask的最小bounding box的坐标和大小。
    
    参数:
    - mask: 3D numpy数组，表示mask图像。非零值表示感兴趣的区域。
    
    返回:
    - bbox_coords: 包含mask的bounding box的起始坐标 (z_min, y_min, x_min)。
    - bbox_size: bounding box的大小 (depth, height, width)。
    """
    # 找到mask中的非零元素的坐标
    non_zero_coords = np.array(np.nonzero(mask))

    # 获取每个维度的最小和最大坐标
    z_min, y_min, x_min = np.min(non_zero_coords, axis=1)
    z_max, y_max, x_max = np.max(non_zero_coords, axis=1)

    # bbox大小（注意max坐标需要加1，因为range是[min, max]）
    bbox_size = (z_max - z_min + 1, y_max - y_min + 1, x_max - x_min + 1)

    # bbox起始坐标
    bbox_coords = (z_min, y_min, x_min)

    return bbox_coords, bbox_size

def get_do_separate_z(spacing, anisotropy_threshold=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis

def resample_patient(data, seg, original_spacing, target_spacing, order_data=3, order_seg=1, force_separate_z=False,
                     order_z_data=0, order_z_seg=0,
                     separate_z_anisotropy_threshold=3):
    """
    :param data:
    :param seg:
    :param original_spacing:
    :param target_spacing:
    :param order_data:
    :param order_seg:
    :param force_separate_z: if None then we dynamically decide how to le along z, if True/False then always
    /never resample along z separately
    :param order_z_seg: only applies if do_separate_z is True
    :param order_z_data: only applies if do_separate_z is True
    :param separate_z_anisotropy_threshold: if max_spacing > separate_z_anisotropy_threshold * min_spacing (per axis)
    then resample along lowres axis with order_z_data/order_z_seg instead of order_data/order_seg

    :return:
    """
    assert not ((data is None) and (seg is None))
    if data is not None:
        assert len(data.shape) == 4, "data must be c x y z"
    if seg is not None:
        assert len(seg.shape) == 4, "seg must be c x y z"

    if data is not None:
        shape = np.array(data[0].shape)
    else:
        shape = np.array(seg[0].shape)
    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)

    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(original_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(original_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(original_spacing)
        elif get_do_separate_z(target_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(target_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            # every axis has the spacing, this should never happen, why is this code here?
            do_separate_z = False
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
        else:
            pass

    if data is not None:
        data_reshaped = resample_data_or_seg(data, new_shape, False, axis, order_data, do_separate_z,
                                             order_z=order_z_data)
    else:
        data_reshaped = None
    if seg is not None:
        seg_reshaped = resample_data_or_seg(seg, new_shape, True, axis, order_seg, do_separate_z, order_z=order_z_seg)
    else:
        seg_reshaped = None
    return data_reshaped, seg_reshaped


def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, order_z=0):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == len(data.shape) - 1
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            print("separate z, order in z is", order_z, "order inplane is", order)
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_data.append(resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                    elif axis == 1:
                        reshaped_data.append(resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                    else:
                        reshaped_data.append(resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                reshaped_data = np.stack(reshaped_data, axis)
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z,
                                                                   mode='nearest')[None].astype(dtype_data))
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(
                                map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                                                mode='nearest'))
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_data.append(reshaped[None].astype(dtype_data))
                else:
                    reshaped_final_data.append(reshaped_data[None].astype(dtype_data))
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            print("no separate z, order", order)
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(resize_fn(data[c], new_shape, order, **kwargs)[None].astype(dtype_data))
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no ling necessary")
        return data
    
if __name__ == "__main__":
    
    sdir = '/path/to/nnUNet_raw_data_base/nnUNet_raw_data/'
    task = 'Task407_Parse22_pulmonary_artery'
    targetSpacing = np.array([1.0, 0.651, 0.651])

    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.INFO)

    # file_handler = logging.FileHandler('/path/to/nnUNet_raw_data_base/nnUNet_raw_data/' + task + '/le.txt')
    # file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    # logger.addHandler(file_handler)
    log_file = '/path/to/nnUNetFrame/nnUNet_raw_data_base/nnUNet_raw_data/' + task + '/resample.txt'

    imagesTrdir = sdir + task + '/imagesTr'
    labelsTrdir = sdir + task + '/labelsTr'
    imagesTsdir = sdir + task + '/imagesTs'
    labelsTsdir = sdir + task + '/labelsTs'
    # maybe_mkdir_p(join(sdir + task, "imagesTr_Resamp"))
    # maybe_mkdir_p(join(sdir + task, "labelsTr_Resamp"))
    # maybe_mkdir_p(join(sdir + task, "imagesTs_Resamp"))
    # maybe_mkdir_p(join(sdir + task, "labelsTs_Resamp"))
    # 将之前的预测也重采样至相同尺度
    predTsdir = sdir + task + '/pred_mt'
    # maybe_mkdir_p(join(sdir + task, "predTs_resample"))
    predInputdir = sdir + task + '/pred_input'
    # maybe_mkdir_p(join(sdir + task, "predTs_resample"))

    with open(log_file, 'a+') as f:
        f.write('PA cut 200 from down to top')

    # for imgn in os.listdir(imagesTrdir):
    #     print(imgn)
    #     imgP = join(imagesTrdir, imgn)
    #     labelP = join(labelsTrdir, imgn.split('_0000')[0] + '.nii.gz')
    #     img, label = sitk.ReadImage(imgP), sitk.ReadImage(labelP)
    #     imgarr, labelarr = sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(label)

    #     # info record
    #     itkOriginalSpacing = img.GetSpacing()
    #     itkOriginalOrigin = img.GetOrigin()
    #     itkOriginalDirection = img.GetDirection()
    #     itkOriginalSize = img.GetSize()
    #     OriginalSpacing = np.array(itkOriginalSpacing)[[2,1,0]]
    #     OriginalSize = np.array(itkOriginalSize)[[2,1,0]]

    #     new_shapes = OriginalSpacing / targetSpacing * OriginalSize

    #     before = {
    #             'spacing': OriginalSpacing,
    #             # 'spacing_transposed': original_spacing_transposed,
    #             'data.shape (data is transposed)': OriginalSize
    #         }

    #     imgarr, labelarr = np.expand_dims(imgarr, 0), np.expand_dims(labelarr, 0)
    #     imgarr_, labelarr_ = resample_patient(imgarr, labelarr, np.array(OriginalSpacing), targetSpacing,
    #                                     3, 1,
    #                                     None, 0, 0)
        
    #     bbox_coords, bbox_size = get_bbox_from_mask(labelarr_[0])

    #     print(f"BBox Coordinates: {bbox_coords}")
    #     print(f"BBox Size: {bbox_size}")
        
    #     after = {
    #             'spacing': targetSpacing,
    #             'data.shape (data is resampled)': imgarr_.shape
    #         }
        
    #     # with open(log_file, 'a+') as f:
    #     #     f.write(imgn)
    #     #     f.write('\t before data shape: ')
    #     #     f.write(str(tuple(OriginalSize)))
    #     #     f.write('\t data.shape (data is led): ')
    #     #     f.write(str(tuple(imgarr_.shape)))
    #     #     f.write('\t BBox Coordinates: ')
    #     #     f.write(str(bbox_coords))
    #     #     f.write('\t BBox Size: ')
    #     #     f.write(str(bbox_size))
    #     #     f.write('\n')
        
    #     img_, label_ = sitk.GetImageFromArray(imgarr_[0][bbox_coords[0]:bbox_coords[0]+200, :, :]), sitk.GetImageFromArray(labelarr_[0][bbox_coords[0]:bbox_coords[0]+200, :, :])

    #     img_.SetSpacing(tuple(targetSpacing[[2,1,0]]))
    #     img_.SetOrigin(itkOriginalOrigin)
    #     img_.SetDirection(itkOriginalDirection)
    #     label_.SetSpacing(tuple(targetSpacing[[2,1,0]]))
    #     label_.SetOrigin(itkOriginalOrigin)
    #     label_.SetDirection(itkOriginalDirection)

    #     sitk.WriteImage(img_, join(sdir + task, "imagesTr_resample", imgn))
    #     sitk.WriteImage(label_, join(sdir + task, "labelsTr_Resamp", imgn.split('_0000')[0] + '.nii.gz'))

    for imgn in os.listdir(predInputdir):
        print(imgn)
        imgP = join(predInputdir, imgn)
        # labelP = join(labelsTsdir, imgn.split('_0000')[0] + '.nii.gz')
        labelP = join(predTsdir, imgn.split('_0000')[0] + '.nii.gz')
        img, label = sitk.ReadImage(imgP), sitk.ReadImage(labelP)
        imgarr, labelarr = sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(label)

        # info record
        itkOriginalSpacing = img.GetSpacing()
        itkOriginalOrigin = img.GetOrigin()
        itkOriginalDirection = img.GetDirection()
        itkOriginalSize = img.GetSize()
        OriginalSpacing = np.array(itkOriginalSpacing)[[2,1,0]]
        OriginalSize = np.array(itkOriginalSize)[[2,1,0]]

        new_shapes = OriginalSpacing / targetSpacing * OriginalSize

        before = {
                'spacing': OriginalSpacing,
                # 'spacing_transposed': original_spacing_transposed,
                'data.shape (data is transposed)': OriginalSize
            }

        imgarr, labelarr = np.expand_dims(imgarr, 0), np.expand_dims(labelarr, 0)
        imgarr_, labelarr_ = resample_patient(imgarr, labelarr, np.array(OriginalSpacing), targetSpacing,
                                        3, 1,
                                        None, 0, 0)
        
        bbox_coords, bbox_size = get_bbox_from_mask(labelarr_[0])

        print(f"BBox Coordinates: {bbox_coords}")
        print(f"BBox Size: {bbox_size}")
        
        after = {
                'spacing': targetSpacing,
                'data.shape (data is resampled)': imgarr_.shape
            }
        
        # with open(log_file, 'a+') as f:
        #         f.write(imgn)
        #         f.write('\t before data shape: ')
        #         f.write(str(tuple(OriginalSize)))
        #         f.write('\t data.shape (data is led): ')
        #         f.write(str(tuple(imgarr_.shape)))
        #         f.write('\t BBox Coordinates: ')
        #         f.write(str(bbox_coords))
        #         f.write('\t BBox Size: ')
        #         f.write(str(bbox_size))
        #         f.write('\n')
        
        
        # img_, label_ = sitk.GetImageFromArray(imgarr_[0][bbox_coords[0]:bbox_coords[0]+200, :, :]), sitk.GetImageFromArray(labelarr_[0][bbox_coords[0]:bbox_coords[0]+200, :, :])
        img_, label_ = sitk.GetImageFromArray(imgarr_[0][:, :, :]), sitk.GetImageFromArray(labelarr_[0][:, :, :])

        img_.SetSpacing(tuple(targetSpacing[[2,1,0]]))
        img_.SetOrigin(itkOriginalOrigin)
        img_.SetDirection(itkOriginalDirection)
        label_.SetSpacing(tuple(targetSpacing[[2,1,0]]))
        label_.SetOrigin(itkOriginalOrigin)
        label_.SetDirection(itkOriginalDirection)

        sitk.WriteImage(img_, join(sdir + task, "pred_input_resamp", imgn))
        sitk.WriteImage(label_, join(sdir + task, "pred_mt_resamp", imgn.split('_0000')[0] + '.nii.gz'))

