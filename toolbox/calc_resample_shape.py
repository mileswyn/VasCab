# 计算重采样之后的数据尺寸
import pickle
import numpy as np

def load_from_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

def calculate_bbox_stats(data):
    # 提取所有的 BBox Size
    bbox_sizes = [entry['BBox Size'] for entry in data]

    # 转换为 numpy 数组，方便计算
    bbox_sizes = np.array(bbox_sizes)

    # 计算最大值、最小值、平均值和中值
    max_bbox = np.max(bbox_sizes, axis=0)
    min_bbox = np.min(bbox_sizes, axis=0)
    mean_bbox = np.mean(bbox_sizes, axis=0)
    median_bbox = np.median(bbox_sizes, axis=0)

    return {
        'max_bbox': max_bbox,
        'min_bbox': min_bbox,
        'mean_bbox': mean_bbox,
        'median_bbox': median_bbox
    }


pkl_file = '/path/to/nnUNet_raw_data_base/nnUNet_raw_data/Task405_CTA_AIIB23trachea/resample.pkl'  # 你的pkl文件路径
data = load_from_pkl(pkl_file)
bbox_stats = calculate_bbox_stats(data)


# print(data)
print(f"最大 BBox Size: {bbox_stats['max_bbox']}")
print(f"最小 BBox Size: {bbox_stats['min_bbox']}")
print(f"平均 BBox Size: {bbox_stats['mean_bbox']}")
print(f"中值 BBox Size: {bbox_stats['median_bbox']}")