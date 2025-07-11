import h5py
import numpy as np
from tqdm import tqdm

def rotate_keypoints(kps, angle):
    """以图像中心 (0.5, 0.5) 为中心旋转关键点"""
    x = kps[:, 0]
    y = kps[:, 1]
    z = kps[:, 2]
    if angle == 90:
        x_new = 0.5 + (y - 0.5)
        y_new = 0.5 - (x - 0.5)
    elif angle == 180:
        x_new = 1.0 - x
        y_new = 1.0 - y
    elif angle == 270:
        x_new = 0.5 - (y - 0.5)
        y_new = 0.5 + (x - 0.5)
    else:
        raise ValueError("旋转角度必须是 90、180 或 270")
    return np.stack([x_new, y_new, z], axis=1)

def augment_and_save_h5(input_path, output_path):
    # 读取原始 h5 文件
    with h5py.File(input_path, 'r') as f_in:
        keypoints = f_in['keypoints'][:]       # shape: (N, 21, 3)
        labels = f_in['labels'][:]             # shape: (N,)
        handedness = f_in['handedness'][:]     # shape: (N,)

    # 准备增强后的数据
    augmented_kps = [keypoints]
    augmented_labels = [labels]
    augmented_handedness = [handedness]

    for angle in [90, 180, 270]:
        rotated_kps = np.array([rotate_keypoints(kp, angle) for kp in keypoints])
        augmented_kps.append(rotated_kps)
        augmented_labels.append(labels.copy())
        augmented_handedness.append(handedness.copy())

    # 拼接所有数据
    all_kps = np.concatenate(augmented_kps, axis=0)
    all_labels = np.concatenate(augmented_labels, axis=0)
    all_handedness = np.concatenate(augmented_handedness, axis=0)

    print(f"原始样本数: {len(labels)}，增强后样本总数: {len(all_labels)}")

    # 写入新 h5 文件（不保存图像）
    with h5py.File(output_path, 'w') as f_out:
        f_out.create_dataset("keypoints", data=all_kps, dtype='float32')
        f_out.create_dataset("labels", data=all_labels, dtype='int64')
        f_out.create_dataset("handedness", data=all_handedness, dtype='int8')

    print(f"增强后的数据已保存至: {output_path}")

if __name__ == "__main__":
    augment_and_save_h5("archive1/process_training.h5", "archive1/augmented_training_no_images.h5")
