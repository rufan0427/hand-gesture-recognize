import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import torch
import math
import torch.nn as nn

# --- Start: Copy from all_to_h5.py (用于计算角度的辅助函数) ---
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

    return np.array(angles, dtype=np.float32) # 3+8+4 = 15
# --- End: Copy from all_to_h5.py ---

class KeypointGestureRecognizer(nn.Module):
    def __init__(self, num_classes, num_keypoints=21, num_angle_features=0):
        super(KeypointGestureRecognizer, self).__init__()
        # 计算输入特征的总维度
        input_dim = 0
        if num_keypoints is not None:
            input_dim += num_keypoints * 3 # 每个关键点有x,y,z 3个坐标
        if num_angle_features is not None:
            input_dim += num_angle_features
            
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.fc_layers(x)
        return x

# Constants
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

# 用于绘制关键点的函数
def draw_landmarks_on_image(rgb_image, detection_result):
    if detection_result is None or detection_result.hand_landmarks is None:
        return rgb_image
    
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Convert to protobuf格式以绘制
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in hand_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())

        # 文字位置
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        
    global current_prediction_label
    if current_prediction_label is not None:
            cv2.putText(annotated_image, current_prediction_label,
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

    return annotated_image

# 结果回调
last_image=None
last_result=None
is_processing = False 
current_prediction_label = None

def result_callback(result, output_image: mp.Image, timestamp_ms: int):
    global last_result, last_image, is_processing, current_prediction_label

    if result is not None and result.hand_world_landmarks:
        # 处理第一只手
        hand_landmarks_1 = result.hand_world_landmarks[0]  
        kps_np_1 = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks_1], dtype=np.float32)
        angles_np_1 = get_finger_angles(kps_np_1)
        
        combined_input_tensor_1 = torch.cat((torch.from_numpy(kps_np_1).flatten(), torch.from_numpy(angles_np_1)), dim=0).unsqueeze(0).to(net.device)
        
        with torch.no_grad():
            prediction_1 = net(combined_input_tensor_1)
            predicted_label_1 = prediction_1.argmax(dim=1).item()
            
        predicted_label_str = f"result: {predicted_label_1}" 
        
        # 处理第二只手 (如果存在)
        if len(result.hand_world_landmarks) > 1:
            hand_landmarks_2 = result.hand_world_landmarks[1]
            kps_np_2 = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks_2], dtype=np.float32)
            angles_np_2 = get_finger_angles(kps_np_2) 
            
            combined_input_tensor_2 = torch.cat((torch.from_numpy(kps_np_2).flatten(), torch.from_numpy(angles_np_2)), dim=0).unsqueeze(0).to(net.device)
            
            with torch.no_grad():
                prediction_2 = net(combined_input_tensor_2)
                predicted_label_2 = prediction_2.argmax(dim=1).item()

            hand_1_handedness_label = result.handedness[0][0].category_name
            
            if hand_1_handedness_label == 'Left':
                predicted_label_str = f"result: Left:{predicted_label_1}, Right:{predicted_label_2}"
            else: 
                predicted_label_str = f"result: Right:{predicted_label_1}, Left:{predicted_label_2}"
                
        current_prediction_label = predicted_label_str

    last_result = result
    last_image = output_image.numpy_view()
    is_processing = False 

# 初始化 HandLandmarker
base_options = python.BaseOptions(model_asset_path=r'D:\\Desktop\\contest\\hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=result_callback,
    num_hands=2)

detector = vision.HandLandmarker.create_from_options(options)

num_classes = 10
num_angle_features = 15 

net = KeypointGestureRecognizer(num_classes, num_keypoints=21, num_angle_features=num_angle_features)

net.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
net.load_state_dict(torch.load("keypoint_gesture_recognizer.pth", map_location=net.device))
net.eval() # 设置为评估模式
net.to(net.device) # 将模型移动到正确的设备

# 摄像头输入
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cv2.namedWindow("Hand Landmarks", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Landmarks", 800, 600)

timestamp = 0
annotated_frame_bgr = None # 初始化，避免在循环中首次使用时出现未定义错误

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        if not is_processing: # 避免卡顿，只有上一帧处理完，再去处理最新的一帧
          timestamp += 1
          detector.detect_async(mp_image, timestamp)
          is_processing = True
          
          if last_image is not None and last_result is not None:
              annotated = draw_landmarks_on_image(last_image, last_result)
              annotated_frame_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
              cv2.imshow("Hand Landmarks", annotated_frame_bgr)
          else:
              cv2.imshow("Hand Landmarks", frame)
        else:
            # 如果 MediaPipe 正在处理上一帧，并且已经有标注帧，则显示旧的标注帧以保持流畅
            if annotated_frame_bgr is not None:
                cv2.imshow("Hand Landmarks", annotated_frame_bgr)
            else:
                cv2.imshow("Hand Landmarks", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()