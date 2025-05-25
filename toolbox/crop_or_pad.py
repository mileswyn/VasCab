# 将数据统一裁剪到指定大小bbox
import pickle
import SimpleITK as sitk
import numpy as np

from collections import OrderedDict
from copy import deepcopy

import shutil
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
from tqdm import tqdm

def load_from_pkl(pkl_file):
    """从pkl文件中加载数据"""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

def load_nifti_file(nifti_file):
    """读取nifti文件，返回numpy数组和SimpleITK Image对象"""
    image = sitk.ReadImage(nifti_file)
    array = sitk.GetArrayFromImage(image)  # 转换为numpy数组

    # info record
    itkOriginalSpacing = image.GetSpacing()
    itkOriginalOrigin = image.GetOrigin()
    itkOriginalDirection = image.GetDirection()
    itkOriginalSize = image.GetSize()

    return array, image, itkOriginalSpacing, itkOriginalDirection, itkOriginalOrigin

def calculate_bbox_center(bbox_coords, bbox_size):
    """根据bbox坐标和尺寸计算bbox的中心"""
    center = [bbox_coords[i] + bbox_size[i] // 2 for i in range(3)]
    return center

def crop_or_pad(image, center, output_size, islabel=False):
    """从中心裁剪或填充图像到指定大小"""
    z_size, y_size, x_size = image.shape
    target_z, target_y, target_x = output_size

    # 计算裁剪或填充的范围 (PA和AIIB不需要计算z轴，因为在z轴上已经固定好了大小)
    # 对于crown，不需要
    z_start = max(0, center[0] - target_z // 2)
    z_end = min(z_size, z_start + target_z)
    y_start = max(0, center[1] - target_y // 2)
    y_end = min(y_size, y_start + target_y)
    x_start = max(0, center[2] - target_x // 2)
    x_end = min(x_size, x_start + target_x)

    # 裁剪图像
    # cropped_image = image[:, y_start:y_end, x_start:x_end].dtype(np.int16)
    cropped_image = image[z_start:z_end, y_start:y_end, x_start:x_end]
    # cropped_image = cropped_image.astype(np.int16)

    # 如果裁剪后的图像不足目标大小，进行填充
    pad_z = (target_z - cropped_image.shape[0]) // 2
    pad_y = (target_y - cropped_image.shape[1]) // 2
    pad_x = (target_x - cropped_image.shape[2]) // 2

    mini_value = cropped_image.min()

    if islabel:
        padded_image = np.pad(cropped_image,
                            # ((0, 0),
                            ((pad_z, target_z - cropped_image.shape[0] - pad_z),
                            (pad_y, target_y - cropped_image.shape[1] - pad_y),
                            (pad_x, target_x - cropped_image.shape[2] - pad_x)),
                            mode='constant', constant_values=0)
    else:
        padded_image = np.pad(cropped_image,
                            # ((0, 0),
                            ((pad_z, target_z - cropped_image.shape[0] - pad_z),
                            (pad_y, target_y - cropped_image.shape[1] - pad_y),
                            (pad_x, target_x - cropped_image.shape[2] - pad_x)),
                            mode='constant', constant_values=mini_value)

    return padded_image

def process_nifti_with_bbox(pkl_file, volume_file, label_file, output_size=(200, 360, 360)):
    """处理nifti文件并根据bbox对volume和label进行裁剪或填充"""
    data = load_from_pkl(pkl_file)
    
    # 读取volume和label
    volume_array, volume_sitk, itkOriginalSpacing, itkOriginalDirection, itkOriginalOrigin = load_nifti_file(volume_file)
    label_array, label_sitk, _, _, _ = load_nifti_file(label_file)

    # 假设pkl文件中对应当前volume文件的数据项
    for entry in data:
        if entry['name'] == volume_file.split('/')[-1]:  # 匹配volume文件名
            bbox_coords = entry['BBox Coordinates']
            bbox_size = entry['BBox Size']
            center = calculate_bbox_center(bbox_coords, bbox_size)

            # 根据中心裁剪或填充volume和label
            processed_volume = crop_or_pad(volume_array, center, output_size, islabel=False)
            processed_label = crop_or_pad(label_array, center, output_size, islabel=True)

            # 将结果转换回SimpleITK格式并保存
            result_volume = sitk.GetImageFromArray(processed_volume)
            result_volume.SetSpacing(itkOriginalSpacing)
            result_volume.SetOrigin(itkOriginalOrigin)
            result_volume.SetDirection(itkOriginalDirection)
            # sitk.WriteImage(result_volume, 'output_volume_' + volume_file.split('/')[-1])

            result_label = sitk.GetImageFromArray(processed_label)
            result_label.SetSpacing(itkOriginalSpacing)
            result_label.SetOrigin(itkOriginalOrigin)
            result_label.SetDirection(itkOriginalDirection)
            # sitk.WriteImage(result_label, 'output_label_' + label_file.split('/')[-1])
            return result_volume, result_label


if __name__ == "__main__":
    sdir = '/path/to/nnUNet_raw_data_base/nnUNet_raw_data/'
    task = 'Task407_Parse22_pulmonary_artery'
    targetSpacing = np.array([1.0, 0.651, 0.651])

    pkl_file = sdir + task + '/resample.pkl'

    imagesTrdir = sdir + task + '/imagesTr_Resamp'
    labelsTrdir = sdir + task + '/labelsTr_Resamp'
    imagesTsdir = sdir + task + '/imagesTs_Resamp'
    labelsTsdir = sdir + task + '/labelsTs_Resamp'
    maybe_mkdir_p(join(sdir + task, "imagesTr"))
    maybe_mkdir_p(join(sdir + task, "labelsTr"))
    maybe_mkdir_p(join(sdir + task, "imagesTs"))
    maybe_mkdir_p(join(sdir + task, "labelsTs"))
    predimagesTsdir = sdir + task + '/pred_input_resamp'
    predlabelsTsdir = sdir + task + '/pred_mt_resamp'

    # for imgn in tqdm(os.listdir(imagesTrdir)):

    #     print(imgn)
    #     imgP = join(imagesTrdir, imgn)
    #     labelP = join(labelsTrdir, imgn.split('_0000')[0] + '.nii.gz')

    #     # 处理volume和label
    #     rvolume, rlabel = process_nifti_with_bbox(pkl_file, imgP, labelP)
    #     sitk.WriteImage(rvolume, join(sdir + task, "imagesTr", imgn))
    #     sitk.WriteImage(rlabel, join(sdir + task, "labelsTr", imgn.split('_0000')[0] + '.nii.gz'))

    # for imgn in tqdm(os.listdir(imagesTsdir)):

    #     print(imgn)
    #     imgP = join(imagesTsdir, imgn)
    #     labelP = join(labelsTsdir, imgn.split('_0000')[0] + '.nii.gz')

    #     # 处理volume和label
    #     rvolume, rlabel = process_nifti_with_bbox(pkl_file, imgP, labelP)
    #     sitk.WriteImage(rvolume, join(sdir + task, "imagesTs", imgn))
    #     sitk.WriteImage(rlabel, join(sdir + task, "labelsTs", imgn.split('_0000')[0] + '.nii.gz'))

    for imgn in tqdm(os.listdir(predimagesTsdir)):

        print(imgn)
        imgP = join(predimagesTsdir, imgn)
        labelP = join(predlabelsTsdir, imgn.split('_0000')[0] + '.nii.gz')

        # 处理volume和label
        rvolume, rlabel = process_nifti_with_bbox(pkl_file, imgP, labelP)
        sitk.WriteImage(rvolume, join(sdir + task, "pred_input_out", imgn))
        sitk.WriteImage(rlabel, join(sdir + task, "pred_mt_out", imgn.split('_0000')[0] + '.nii.gz'))
