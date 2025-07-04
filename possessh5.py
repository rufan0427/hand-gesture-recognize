import h5py
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 初始化MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def process_h5(input_path, output_path):
    # 读取原始数据
    with h5py.File(input_path, 'r') as f_in:
        images = f_in['train_set_x'][:]#对于test数据要换成test_set
        labels = f_in['train_set_y'][:]
    
    # 创建新HDF5文件
    with h5py.File(output_path, 'w') as f_out:
        # 创建可扩展数据集
        f_out.create_dataset('images', 
                            shape=(0,224,224,3), 
                            maxshape=(None,224,224,3),
                            dtype='uint8')
        f_out.create_dataset('keypoints',
                            shape=(0,21,3),
                            maxshape=(None,21,3),
                            dtype='float32')
        f_out.create_dataset('labels',
                            shape=(0,),
                            maxshape=(None,),
                            dtype='int64')
        f_out.create_dataset('handedness',
                            shape=(0,),
                            maxshape=(None,),
                            dtype='int8')
        
        # 处理每张图像
        for i in tqdm(range(len(images))):
            img = images[i]
            img_rgb = img
            
            # 调整图像大小
            img_resized = cv2.resize(img_rgb, (224,224))
            
            # MediaPipe处理
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                # 获取关键点
                hand_landmarks = results.multi_hand_landmarks[0]
                kps = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                labelfind = labels[i]

                # 获取左右手信息
                handedness = 0 if results.multi_handedness[0].classification[0].label == 'Left' else 1
                
                # 扩展数据集
                for ds_name, data in [('images', img_resized),
                                    ('keypoints', kps),
                                    ('labels',labelfind),
                                    ('handedness', handedness)]:
                    f_out[ds_name].resize((f_out[ds_name].shape[0]+1, *f_out[ds_name].shape[1:]))
                    f_out[ds_name][-1] = data
            else:
                print(f"No hand detected in image {i}, skipping...")

if __name__ == "__main__":
    process_h5('archive1/Signs_Data_Training.h5', 'archive1/process_training.h5')
    # 检查第10个样本的处理结果