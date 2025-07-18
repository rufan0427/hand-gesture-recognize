import os
import cv2
import h5py
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import math
# 初始化 MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, model_complexity=1, max_num_hands=1,min_detection_confidence=0.5  )

def rotate_keypoints(keypoints, angle_degrees):
    """
    对关键点进行绕Z轴的二维旋转。
    keypoints: numpy array of shape (21, 3)
    angle_degrees: 旋转角度（度）
    """
    angle_radians = math.radians(angle_degrees)
    rotation_matrix = np.array([
        [math.cos(angle_radians), -math.sin(angle_radians), 0],
        [math.sin(angle_radians),  math.cos(angle_radians), 0],
        [0,                      0,                      1]
    ])
    rotated_keypoints = np.dot(keypoints, rotation_matrix)
    return rotated_keypoints

def flip_keypoints_x(keypoints):
    """
    对关键点进行X轴镜像翻转。
    keypoints: numpy array of shape (21, 3)
    """
    flipped_keypoints = keypoints.copy()
    flipped_keypoints[:, 0] = -flipped_keypoints[:, 0]  # 翻转X坐标
    return flipped_keypoints

def flip_keypoints_y(keypoints):
    """
    对关键点进行y轴镜像翻转。
    keypoints: numpy array of shape (21, 3)
    """
    flipped_keypoints = keypoints.copy()
    flipped_keypoints[:, 1] = -flipped_keypoints[:, 1]  # 翻转X坐标
    return flipped_keypoints

def process_dataset_to_h5(dataset_dir, output_h5_path):
    # 获取所有类别文件夹
    class_dirs = sorted([d for d in os.listdir(dataset_dir) 
                        if os.path.isdir(os.path.join(dataset_dir, d))])
    
    # 创建 HDF5 文件
    with h5py.File(output_h5_path, 'w') as f_out:
        # 创建数据集
        f_out.create_dataset('keypoints', shape=(0, 21, 3), 
                           maxshape=(None, 21, 3), dtype='float32')
        f_out.create_dataset('labels', shape=(0,), 
                           maxshape=(None,), dtype='int64')
        f_out.create_dataset('handedness', shape=(0,), 
                           maxshape=(None,), dtype='int8')
        
        # 处理每个类别文件夹
        for class_dir in tqdm(class_dirs, desc="Processing Classes"):
            class_path = os.path.join(dataset_dir, class_dir)
            label = int(class_dir)  # 文件夹名就是标签

            rotation_angles = [0, 5, 355 , 10 , 350 , 90, 180, 270] # 原始、旋转90、180、270度
            apply_flip = True # 是否应用镜像对称

            # 获取类别文件夹中的所有图片
            image_files = [f for f in os.listdir(class_path) 
                          if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            # 处理每张图片
            for img_file in tqdm(image_files, desc=f"Processing Class {label}", leave=False):
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"无法读取图片: {img_path}")
                    continue
                    
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 提取关键点
                results = hands.process(img_rgb)
                if results.multi_hand_world_landmarks:
                    hand_landmarks = results.multi_hand_world_landmarks[0]
                    kps = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    
                    # 确定手性 (0=左手, 1=右手)
                    handedness = 0 if results.multi_handedness[0].classification[0].label == 'Left' else 1
                    
                    # 扩展数据集
                    f_out['keypoints'].resize((f_out['keypoints'].shape[0] + 1, 21, 3))
                    f_out['keypoints'][-1] = kps
                    
                    f_out['labels'].resize((f_out['labels'].shape[0] + 1,))
                    f_out['labels'][-1] = label
                    
                    f_out['handedness'].resize((f_out['handedness'].shape[0] + 1,))
                    f_out['handedness'][-1] = handedness

                    for angle in rotation_angles:
                        rotated_kps = rotate_keypoints(kps, angle)
                        
                        # 保存原始旋转数据
                        f_out['keypoints'].resize((f_out['keypoints'].shape[0] + 1, 21, 3))
                        f_out['keypoints'][-1] = rotated_kps
                        
                        f_out['labels'].resize((f_out['labels'].shape[0] + 1,))
                        f_out['labels'][-1] = label
                        
                        f_out['handedness'].resize((f_out['handedness'].shape[0] + 1,))
                        f_out['handedness'][-1] = handedness

                        # 如果启用镜像，则对旋转后的数据进行镜像
                        if apply_flip:
                            flipped_rotated_kps = flip_keypoints_x(rotated_kps)
                            flipped_handedness = 1 - handedness # 翻转手性
                            
                            f_out['keypoints'].resize((f_out['keypoints'].shape[0] + 1, 21, 3))
                            f_out['keypoints'][-1] = flipped_rotated_kps
                            
                            f_out['labels'].resize((f_out['labels'].shape[0] + 1,))
                            f_out['labels'][-1] = label
                            
                            f_out['handedness'].resize((f_out['handedness'].shape[0] + 1,))
                            f_out['handedness'][-1] = flipped_handedness

                            flipped_rotated_kps = flip_keypoints_y(rotated_kps)
                            flipped_handedness = 1 - handedness # 翻转手性
                            
                            f_out['keypoints'].resize((f_out['keypoints'].shape[0] + 1, 21, 3))
                            f_out['keypoints'][-1] = flipped_rotated_kps
                            
                            f_out['labels'].resize((f_out['labels'].shape[0] + 1,))
                            f_out['labels'][-1] = label
                            
                            f_out['handedness'].resize((f_out['handedness'].shape[0] + 1,))
                            f_out['handedness'][-1] = flipped_handedness
                else:
                    print(f"未检测到手部: {img_path}")
            #cv2.imshow("photo",img_rgb)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

if __name__ == "__main__":
    dataset_dir = "dataset1/Chinese-Number-Gestures-Recognition-main/data/RGB/train"  # 包含0-9文件夹的根目录
    output_h5_path = "archive1/hand_landmarks_dataset1.h5"
    process_dataset_to_h5(dataset_dir, output_h5_path)