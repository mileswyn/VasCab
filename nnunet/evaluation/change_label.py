import os
import SimpleITK as sitk
import numpy as np

origin_dir = '/data2/wyn/nnUNetFrame/nnUNet_raw_data_base/nnUNet_raw_data/Task101_MTvessel/labelsTs/'
target_dir = '/data2/wyn/nnUNetFrame/nnUNet_raw_data_base/nnUNet_raw_data/Task101_MTvessel/labelsTs_single_label/'
for im in os.listdir(origin_dir):
    if im.endswith('.nii.gz'):
        img = sitk.ReadImage(os.path.join(origin_dir, im))
        arr = sitk.GetArrayFromImage(img)
        arr[arr > 0] = 1
        nim = sitk.GetImageFromArray(arr)
        nim.SetSpacing(img.GetSpacing())
        nim.SetDirection(img.GetDirection())
        nim.SetOrigin(img.GetOrigin())
        sitk.WriteImage(nim, os.path.join(target_dir, im))
