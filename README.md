cv_process_picture保存了通过传统的openCV处理手部轮廓，这最终并没有用于神经网络的训练。
process_data文件夹主要保存了jpg数据集->mediapipe标注的h5文件，以及普通h5文件进行标注的脚本，最终形成训练所用的数据集。
其中mediapipe标注使用了hand_landmarker.task模型,mediapipe_identify_knots.py作为mediapipe的例程，实现视频流的手势标注。
number_recognize_linear.py训练模型keypoint_gesture_recognizer.pth，执行identify_stream.py实现视频的实时识别。

手势识别数据集来源：https://www.kaggle.com/datasets/maneesh99/signs-detection-dataset

.h5 格式的图片数据集,已划分为train和test，经过processh5.py (https://github.com/rufan0427/hand-gesture-recognize/blob/main/possessh5.py) 的处理，利用mediapipe进行标注，神经网络利用标注的关键点作为input。

利用 https://github.com/rufan0427/hand-gesture-recognize/blob/main/jpg_to_h5.py 可以实现jpg格式到h5格式的转换。

处理前后的数据集都在https://drive.google.com/file/d/1AkPZBz0_GjU2eEbtklE1dlereHTmROPp/view?usp=drive_link 上（由于过大，无法上传github）。不过建议使用keypoint_gesture_recognizer.pth,因为经过了某些神奇的训练。
