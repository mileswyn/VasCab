dataset_lab_map = { 
    'lits': [0, 1], 
    'kits': [2, 3], 
    'amos_ct': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    'amos_mr': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'bcv': [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    'structseg_oar': [32, 33, 34, 35, 36, 37],
    'chaos': [38, 39, 40, 41],
    'structseg_head_oar': [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
    'mnm': [64, 65, 66],
    'brain_structure': [67, 68, 69],
    'autopet': [70]
}  

# CT: 0, PET: 1, T1 MRI: 2, T2 MRI: 3, cineMRI: 4
dataset_modality_map = { 
    'lits': 0, 
    'kits': 0, 
    'amos_ct': 0,
    'amos_mr': -1, # unknown MRI sequences, don't compute loss
    'bcv': 0,
    'structseg_oar': 0,
    'chaos_t1_in': 2,
    'chaos_t1_out': 2,
    'chaos_t2': 3,
    'structseg_head_oar': 0,
    'mnm': 4,
    'brain_structure': 2,
    'autopet': 1,
}