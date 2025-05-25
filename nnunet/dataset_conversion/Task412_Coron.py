import shutil

from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk
from nnunet.paths import nnUNet_raw_data
from collections import OrderedDict
import numpy as np

if __name__ == "__main__":
    folder = "/data2/wyn/vessel_dataset/Task10_MRA_topCow/"
    #extra_folder = '/hdd1/wyn/MRA19_extra/'
    out_folder = "/data2/wyn/nnUNetFrame/nnUNet_raw_data_base/nnUNet_raw_data/Task412_Coron/"

    maybe_mkdir_p(join(out_folder, "imagesTr"))
    maybe_mkdir_p(join(out_folder, "imagesTs"))
    maybe_mkdir_p(join(out_folder, "labelsTr"))
    maybe_mkdir_p(join(out_folder, "labelsTs"))
    # train
    # current_dir = join(folder, "train", "image")
    # label_folder = join(folder, "train", "gt")
    current_dir = join(out_folder, "imagesTr")
    label_folder = join(out_folder, "labelsTr")
    
    # test
    # current_dir_test = join(folder, "test", "image")
    # label_folder_test = join(folder, "test", "gt")
    current_dir_test = join(out_folder, "imagesTs")
    label_folder_test = join(out_folder, "labelsTs")
    segmentations = []
    num_patient = []
    num_patient_label = []
    raw_data = []

    for i in os.listdir(current_dir):
        num_patient.append(i)
        raw_data.append(os.path.join(current_dir, i))
    for i in os.listdir(label_folder):
        num_patient_label.append(i)
        segmentations.append(os.path.join(label_folder,i))
    #segmentations = subfiles(current_dir, suffix="GT.nii.gz")
    #segmentations = [i for i in subfiles(join(folder, "labelsTr"), suffix="nii.gz")]
    #raw_data = [i for i in subfiles(current_dir, suffix="nii.gz") if not i.endswith("GT.nii.gz")]

    num_patient.sort()
    raw_data.sort()
    num_patient_label.sort()
    segmentations.sort()

    # for i,data in enumerate(raw_data):
    #     print(num_patient[i])
    #     out_fname = join(out_folder, "imagesTr", num_patient[i].split('_0000')[0] + "_0000.nii.gz")
    #     print(data)
    #     img = sitk.ReadImage(data)
    #     # img.SetOrigin((0,0,0))
    #     # img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    #     sitk.WriteImage(img, out_fname)
    # for i,data in enumerate(segmentations):
    #     out_fname = join(out_folder, "labelsTr", num_patient_label[i].split('.')[0]  + ".nii.gz")
    #     print(data, out_fname)
    #     gt = sitk.ReadImage(data)
    #     # ori_spacing = gt.GetSpacing()
    #     # gt_arr = sitk.GetArrayFromImage(gt)
    #     # gt_arr[gt_arr == 2] = 0
    #     # gt = sitk.GetImageFromArray(gt_arr)
    #     # gt.SetSpacing(ori_spacing)
    #     # gt.SetOrigin((0,0,0))
    #     # gt.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    #     sitk.WriteImage(gt, out_fname)

    test_data = []
    test_label = []
    num_patient_test = []
    num_patient_test_label = []
    # test_dir = join(folder, "test", "image")
    for i in os.listdir(current_dir_test):
        num_patient_test.append(i)
        test_data.append(os.path.join(current_dir_test, i))
    
    for i in os.listdir(label_folder_test):
        num_patient_test_label.append(i)
        test_label.append(os.path.join(label_folder_test, i))
    
    # #test_data = subfiles(current_dir, suffix="nii.gz")
    # for i,data in enumerate(test_data):
    #     print(num_patient_test[i])
    #     out_fname = join(out_folder, "imagesTs", num_patient_test[i].split('_0000')[0] + "_0000.nii.gz")
    #     print(data)
    #     img = sitk.ReadImage(data)
    #     # img.SetOrigin((0,0,0))
    #     # img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    #     # sitk.WriteImage(img, out_fname)
    #     # test_im = sitk.ReadImage(os.path.join(data, "label_cut.nii.gz"))
    #     # test_arr = sitk.GetArrayFromImage(test_im)
    #     # testarr_mandible = np.zeros_like(test_arr)
    #     # testarr_mandible[test_arr==3] = 1
    #     # test_mad_im = sitk.GetImageFromArray(testarr_mandible.astype(np.uint8))
    #     sitk.WriteImage(img, out_fname)
        
    # for i,data in enumerate(test_label):
    #     out_fname = join(out_folder, "labelsTs", num_patient_test_label[i].split('.')[0]  + ".nii.gz")
    #     print(data, out_fname)
    #     gt = sitk.ReadImage(data)
    #     # ori_spacing = gt.GetSpacing()
    #     # gt_arr = sitk.GetArrayFromImage(gt)
    #     # gt_arr[gt_arr == 2] = 0
    #     # gt = sitk.GetImageFromArray(gt_arr)
    #     # gt.SetSpacing(ori_spacing)
    #     # gt.SetOrigin((0,0,0))
    #     # gt.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    #     sitk.WriteImage(gt, out_fname)
    #     # sitk.WriteImage(sitk.ReadImage(data), out_fname_gt)

        
    json_dict = OrderedDict()
    json_dict['name'] = "CTA"
    json_dict['description'] = "Coron"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = ""
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "foreground",
    }
    json_dict['numTraining'] = len(raw_data)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s" % num_patient[i].split('_0000')[0] + ".nii.gz", "label": "./labelsTr/%s" % num_patient[i].split('_0000')[0] + ".nii.gz"} for i in range(0, len(raw_data))]
    # json_dict['test'] = ["./imagesTs/%s" % num_patient_test[i].split('.')[0] + ".nii.gz" for i in range(0, len(test_data))]
    json_dict['test'] = ["./imagesTs/%s" % num_patient_test[i].split('_0000')[0] + ".nii.gz" for i in range(0, len(num_patient_test))]
    save_json(json_dict, os.path.join(out_folder, "dataset.json"))