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

def calculate_angle(p1, p2, p3):
    """
    计算由三个点 P1-P2-P3 形成的夹角 (以 P2 为顶点)。
    p1, p2, p3: numpy array of shape (3,) representing (x, y, z) coordinates.
    返回角度（度数）。
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    # 避免除以零
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0 # 或者其他表示无效角度的值

    dot_product = np.dot(v1, v2)
    angle_rad = np.arccos(np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0))
    return np.degrees(angle_rad)
def calculate_angle2(p1,p2,p3,p4):
    v1 = p2 - p1
    v2 = p4 - p3
    
    # 避免除以零
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0 # 或者其他表示无效角度的值

    dot_product = np.dot(v1, v2)
    angle_rad = np.arccos(np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0))
    return np.degrees(angle_rad)

def get_finger_angles(keypoints):
    """
    从 MediaPipe 关键点中提取有意义的手指角度。
    keypoints: numpy array of shape (21, 3)
    返回一个包含多个角度的 numpy 数组。
    """
    angles = []

    # ----------------------------------------------------
    # 1. 每根手指的弯曲角度 (内部关节角度，反映手指的弯曲程度)
    # ----------------------------------------------------
    
    # 拇指：(0)手腕-(1)拇指掌骨-(2)拇指近端-(3)拇指中间-(4)拇指尖
    # 拇指 CMC (腕掌) 关节角度 (0-1-2)
    angles.append(calculate_angle(keypoints[0], keypoints[1], keypoints[2]))
    # 拇指 MCP (掌指) 关节角度 (1-2-3)
    angles.append(calculate_angle(keypoints[1], keypoints[2], keypoints[3]))
    # 拇指 IP (指间) 关节角度 (2-3-4)
    angles.append(calculate_angle(keypoints[2], keypoints[3], keypoints[4]))

    # 其他四根手指：食指、中指、无名指、小指
    # 关键点索引:
    # 食指: 5(掌骨), 6(MCP), 7(PIP), 8(DIP/尖)
    # 中指: 9(掌骨), 10(MCP), 11(PIP), 12(DIP/尖)
    # 无名指: 13(掌骨), 14(MCP), 15(PIP), 16(DIP/尖)
    # 小指: 17(掌骨), 18(MCP), 19(PIP), 20(DIP/尖)
    
    finger_metacarpal_bases = [5, 9, 13, 17] # 掌骨基部 (靠近手腕)
    finger_mcp_joints = [6, 10, 14, 18] # 掌指关节
    finger_pip_joints = [7, 11, 15, 19] # 近端指间关节
    finger_tip_joints = [8, 12, 16, 20] # 远端指间关节/指尖

    for i in range(4): # 遍历食指、中指、无名指、小指
        base_idx = finger_metacarpal_bases[i]
        mcp_idx = finger_mcp_joints[i]
        pip_idx = finger_pip_joints[i]
        tip_idx = finger_tip_joints[i] # MediaPipe的指尖就是DIP关节

        # MCP 关节角度 (base_idx - mcp_idx - pip_idx)
        angles.append(calculate_angle(keypoints[base_idx], keypoints[mcp_idx], keypoints[pip_idx]))
        # PIP 关节角度 (mcp_idx - pip_idx - tip_idx)
        angles.append(calculate_angle(keypoints[mcp_idx], keypoints[pip_idx], keypoints[tip_idx]))
        pass

    # 2. 手指之间的张开角度 
    # 拇指与食指之间的张开角度 
    angles.append(calculate_angle2(keypoints[1],keypoints[2], keypoints[5],keypoints[6]))
    # 食指与中指之间的张开角度
    angles.append(calculate_angle2(keypoints[5],keypoints[6], keypoints[9], keypoints[10]))
    # 中指与无名指之间的张开角度
    angles.append(calculate_angle2(keypoints[9],keypoints[10], keypoints[13], keypoints[14]))
    # 无名指与小指之间的张开角度 
    angles.append(calculate_angle2(keypoints[13], keypoints[14],keypoints[17], keypoints[18]))

    return np.array(angles, dtype=np.float32)#3+8+4 = 15

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

def flip_keypoints_z(keypoints):
    """
    对关键点进行y轴镜像翻转。
    keypoints: numpy array of shape (21, 3)
    """
    flipped_keypoints = keypoints.copy()
    flipped_keypoints[:, 2] = -flipped_keypoints[:, 2]  # 翻转X坐标
    return flipped_keypoints


def process_dataset_to_h5(dataset_dir, output_h5_path):
    # 获取所有类别文件夹
    #class_dirs = sorted([d for d in os.listdir(dataset_dir) 
    #                    if os.path.isdir(os.path.join(dataset_dir, d))])
    
    # 创建 HDF5 文件
    with h5py.File(output_h5_path, 'w') as f_out:
        # 创建数据集
        f_out.create_dataset('keypoints', shape=(0, 21, 3), 
                           maxshape=(None, 21, 3), dtype='float32')
        f_out.create_dataset('labels', shape=(0,), 
                           maxshape=(None,), dtype='int64')
        f_out.create_dataset('handedness', shape=(0,), 
                           maxshape=(None,), dtype='int8')
        f_out.create_dataset('angles',shape=(0,15),maxshape=(None,15),dtype='float32')
        # 处理每个类别文件夹
        #for class_dir in tqdm(class_dirs, desc="Processing Classes"):
            #class_path = os.path.join(dataset_dir, class_dir)
            #Wlabel = int(class_dir)  # 文件夹名就是标签
            
        class_path  = dataset_dir
        rotation_angles = [0, 5,355, 10,350] # 原始、旋转90、180、270度
        apply_flip = True # 是否应用镜像对称

        # 获取类别文件夹中的所有图片
        image_files = [f for f in os.listdir(class_path) 
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
            
        # 处理每张图片
        for i,img_file in zip(range(10),tqdm(image_files, desc=f"Processing Class 8", leave=False)):

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
                angles=get_finger_angles(kps)
                label = 5
                # 确定手性 (0=左手, 1=右手)
                handedness = 0 if results.multi_handedness[0].classification[0].label == 'Left' else 1
            
                # 扩展数据集
                f_out['keypoints'].resize((f_out['keypoints'].shape[0] + 1, 21, 3))
                f_out['keypoints'][-1] = kps
                    
                f_out['labels'].resize((f_out['labels'].shape[0] + 1,))
                f_out['labels'][-1] = label
                    
                f_out['handedness'].resize((f_out['handedness'].shape[0] + 1,))
                f_out['handedness'][-1] = handedness

                f_out['angles'].resize((f_out['angles'].shape[0] + 1, angles.shape[0])) # 调整形状
                f_out['angles'][-1] = angles
                for angle in rotation_angles:
                    rotated_kps = rotate_keypoints(kps, angle)
                    rotated_angles=get_finger_angles(rotated_kps)

                    # 保存原始旋转数据
                    f_out['keypoints'].resize((f_out['keypoints'].shape[0] + 1, 21, 3))
                    f_out['keypoints'][-1] = rotated_kps
                        
                    f_out['labels'].resize((f_out['labels'].shape[0] + 1,))
                    f_out['labels'][-1] = label
                        
                    f_out['handedness'].resize((f_out['handedness'].shape[0] + 1,))
                    f_out['handedness'][-1] = handedness

                    f_out['angles'].resize((f_out['angles'].shape[0] + 1, rotated_angles.shape[0])) 
                    f_out['angles'][-1] = rotated_angles
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

                        f_out['angles'].resize((f_out['angles'].shape[0] + 1, rotated_angles.shape[0])) 
                        f_out['angles'][-1] = rotated_angles

                        flipped_rotated_kps = flip_keypoints_y(rotated_kps)
                        flipped_handedness = 1 - handedness # 翻转手性
                            
                        f_out['keypoints'].resize((f_out['keypoints'].shape[0] + 1, 21, 3))
                        f_out['keypoints'][-1] = flipped_rotated_kps
                            
                        f_out['labels'].resize((f_out['labels'].shape[0] + 1,))
                        f_out['labels'][-1] = label
                            
                        f_out['handedness'].resize((f_out['handedness'].shape[0] + 1,))
                        f_out['handedness'][-1] = flipped_handedness

                        f_out['angles'].resize((f_out['angles'].shape[0] + 1, rotated_angles.shape[0])) 
                        f_out['angles'][-1] = rotated_angles

                        flipped_rotated_kps = flip_keypoints_z(rotated_kps)
                        flipped_handedness = 1 - handedness # 翻转手性
                            
                        f_out['keypoints'].resize((f_out['keypoints'].shape[0] + 1, 21, 3))
                        f_out['keypoints'][-1] = flipped_rotated_kps
                            
                        f_out['labels'].resize((f_out['labels'].shape[0] + 1,))
                        f_out['labels'][-1] = label
                            
                        f_out['handedness'].resize((f_out['handedness'].shape[0] + 1,))
                        f_out['handedness'][-1] = flipped_handedness

                        f_out['angles'].resize((f_out['angles'].shape[0] + 1, rotated_angles.shape[0])) 
                        f_out['angles'][-1] = rotated_angles
                else:
                    print(f"未检测到手部: {img_path}")
            #cv2.imshow("photo",img_rgb)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

if __name__ == "__main__":
    dataset_dir = "dataset1/Chinese-Number-Gestures-Recognition-main/data/RGB/test/5"  # 包含0-9文件夹的根目录
    output_h5_path = "archive1/hand_landmarks_dataset_train2.h5"
    process_dataset_to_h5(dataset_dir, output_h5_path)