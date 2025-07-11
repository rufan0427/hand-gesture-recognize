import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np

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

    return annotated_image

# 结果回调
last_image=None
last_result=None
is_processing = False 
def result_callback(result, output_image: mp.Image, timestamp_ms: int):
    global last_result, last_image,is_processing
    last_result = result
    last_image = output_image.numpy_view()
    is_processing = False 

# 初始化 HandLandmarker
base_options = python.BaseOptions(model_asset_path=r'D:\Desktop\contest\hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=result_callback,
    num_hands=2)

detector = vision.HandLandmarker.create_from_options(options)

# 摄像头输入
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cv2.namedWindow("Hand Landmarks", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Landmarks", 800, 600)

timestamp = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 每帧时间戳（毫秒）
        if not is_processing:#避免卡顿，抽帧，只有上一帧处理完，再去处理最新的一帧
          timestamp += 1
          detector.detect_async(mp_image, timestamp)
          is_processing = True
          annotated = draw_landmarks_on_image(last_image,last_result)#产生时延的关键语句
        #这里如果直接用mp_image.numpy_view()会出现画面没有延迟，标注点延迟的情况

        # 如果有结果则显示
          if last_result is not None:
              annotated_frame_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
              cv2.imshow("Hand Landmarks", annotated_frame_bgr)
          else:
              cv2.imshow("Hand Landmarks", frame)
        else:
            if last_result is None:#防止抽帧特征点的显示断断续续闪烁，没有抽出的帧直接显示上一次抽帧的结果，很流畅
                cv2.imshow("Hand Landmarks", frame)
            else:
                cv2.imshow("Hand Landmarks", annotated_frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()


# 参考链接
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python?hl=zh-cn
# mediapipe_python部署注意： python 版本要在3.8~3.10，3.12版本过高
# conda 新建python 3.9环境
# 需要安装Microsoft Visual C++ 2015-2022 Redistributable (注意年份)
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb?hl=zh-cn#scrollTo=_JVO3rvPD4RN