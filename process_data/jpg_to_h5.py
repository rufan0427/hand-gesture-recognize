import os
import cv2
import h5py
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# 初始化 MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def process_images_to_h5(image_dir, output_h5_path):
    # 获取所有图片文件
    '''image_files = sorted(
        [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))],
        key=lambda x: int(x.split('.')[0])  # 按数字排序 (1.jpg, 2.jpg, ...)
    )'''
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    
    # 创建 HDF5 文件
    with h5py.File(output_h5_path, 'w') as f_out:
        # 创建数据集
        f_out.create_dataset('keypoints', shape=(0, 21, 3), 
                           maxshape=(None, 21, 3), dtype='float32')
        f_out.create_dataset('labels', shape=(0,), 
                           maxshape=(None,), dtype='int64')
        f_out.create_dataset('handedness',shape=(0,),maxshape=(None,),dtype='int8')
        # 处理每张图片
        for img_file in tqdm(image_files, desc="Processing Images"):
            img_path = os.path.join(image_dir, img_file)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 调整大小
            img_resized = cv2.resize(img_rgb, (224, 224))
            
            # 提取关键点
            results = hands.process(img_rgb)
            if results.multi_hand_world_landmarks:
                hand_landmarks = results.multi_hand_world_landmarks[0]
                kps = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                handedness = 0 if results.multi_handedness[0].classification[0].label == 'Left' else 1
    
                # 扩展数据集
                for ds_name, data in [
                    ('keypoints', kps),
                    ('labels', 5),  # 假设文件名是标签 (1.jpg → 标签1)
                    ('handedness', handedness)
                ]:
                    f_out[ds_name].resize((f_out[ds_name].shape[0] + 1, *f_out[ds_name].shape[1:]))
                    f_out[ds_name][-1] = data
            else:
                print(f"未检测到手部: {img_file}")

if __name__ == "__main__":
    image_dir = "archive2/Finger_Dataset/five"  # 存放 1.jpg, 2.jpg...的文件夹
    output_h5_path = "archive2/processed_images.h5"
    process_images_to_h5(image_dir, output_h5_path)

#新的数据集来自：https://github.com/daocunyang/Chinese-Number-Gestures-Recognition 和 https://github.com/nadalii/Chinese-Number-Gesture-Recognition-Static-