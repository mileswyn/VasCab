import os
import shutil

dir_type = ['softmax_301', 'softmax_302', 'softmax_303', 'softmax_405', 'softmax_406', 'softmax_407', 'softmax_409', 'softmax_410', 'softmax_411', 'corres']
MultiTalent_type = ['301', '302', '303', '405', '406', '407', '409', '410', '411', 'corres']
save_dir = '/data2/wyn/nnUNetFrame/nnUNet_trained_models/nnUNet/3d_fullres/Task103_MTvessels_mod_spacing/MultiTalent_trainer_ddp__MultiTalent_bs4/all/all/'

for dr in dir_type:
    if not os.path.exists(os.path.join(save_dir, dr)):
        os.mkdir(os.path.join(save_dir, dr))
    # else:
    #     os.rmdir(os.path.join(save_dir, dr))
    #     os.mkdir(os.path.join(save_dir, dr))
        
union_dir = os.path.join(save_dir, 'individual')
for pred in os.listdir(union_dir):
    if 'pkl' in pred:
        continue
    else:
        pred_type = pred.split('.nii.gz')[0]
        origin_type = pred.split('_')[0]
        if 'hubei' in origin_type:
            if '301' in pred_type:
                new_pred_name = pred.split('_301')[0] + '.nii.gz'
                # shutil.copy(os.path.join(union_dir, pred), os.path.join(save_dir, 'softmax_' + '301', new_pred_name))
            else:
                pass
        elif 'IXI' in origin_type:
            if '302' in pred_type:
                new_pred_name = pred.split('_302')[0] + '.nii.gz'
                # shutil.copy(os.path.join(union_dir, pred), os.path.join(save_dir, 'softmax_' + '302', new_pred_name))
            else:
                pass
        elif 'AIIB' in origin_type:
            if '405' in pred_type:
                new_pred_name = pred.split('_405')[0] + '.nii.gz'
                shutil.copy(os.path.join(union_dir, pred), os.path.join(save_dir, 'softmax_' + '405', new_pred_name))
            else:
                pass
        elif 'subject' in origin_type:
            if '406' in pred_type:
                new_pred_name = pred.split('_406')[0] + '.nii.gz'
                shutil.copy(os.path.join(union_dir, pred), os.path.join(save_dir, 'softmax_' + '406', new_pred_name))
            else:
                pass
        elif 'PA' in origin_type:
            if '407' in pred_type:
                new_pred_name = pred.split('_407')[0] + '.nii.gz'
                # shutil.copy(os.path.join(union_dir, pred), os.path.join(save_dir, 'softmax_' + '407', new_pred_name))
            else:
                pass
        elif 'topcow' in origin_type:
            subtype = pred.split('_')[1]
            if subtype == 'ct':
                if '409' in pred_type:
                    new_pred_name = pred.split('_409')[0] + '.nii.gz'
                    # shutil.copy(os.path.join(union_dir, pred), os.path.join(save_dir, 'softmax_' + '409', new_pred_name))
                else:
                    pass
            elif subtype == 'mr':
                if '410' in pred_type:
                    new_pred_name = pred.split('_410')[0] + '.nii.gz'
                    # shutil.copy(os.path.join(union_dir, pred), os.path.join(save_dir, 'softmax_' + '410', new_pred_name))
                else:
                    pass
            else:
                pass
        elif 'crown' in origin_type:
            if '411' in pred_type:
                new_pred_name = pred.split('_411')[0] + '.nii.gz'
                # shutil.copy(os.path.join(union_dir, pred), os.path.join(save_dir, 'softmax_' + '411', new_pred_name))
            else:
                pass    
        else:
            if '303' in pred_type:
                new_pred_name = pred.split('_303')[0] + '.nii.gz'
                # shutil.copy(os.path.join(union_dir, pred), os.path.join(save_dir, 'softmax_' + '303', new_pred_name))
            else:
                pass
        # if origin_type in pred_type:
        #     shutil.copy(os.path.join(union_dir, pred), os.path.join(save_dir, 'corres', new_pred_name))
        # for tp in MultiTalent_type:
        #     if tp in pred_type:
        #         shutil.copy(os.path.join(union_dir, pred), os.path.join(save_dir, 'softmax_' + tp, new_pred_name))
