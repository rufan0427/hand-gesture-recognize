import cv2
import numpy as np

img = cv2.imread("D:/Desktop/contest/image4.jpg")

# Step 1: 初始肤色检测 (YCrCb)
YCrCb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
lower_skin = np.array([0, 139, 85], np.uint8)
upper_skin = np.array([255, 173, 120], np.uint8)
mask_YCrCb = cv2.inRange(YCrCb_img, lower_skin, upper_skin)

# Step 2: 获取最大轮廓
contours, _ = cv2.findContours(mask_YCrCb, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
max_contour = max(contours, key=cv2.contourArea)

# Step 3: 计算轮廓区域的 HSV 均值
mask_contour = np.zeros_like(mask_YCrCb)
cv2.drawContours(mask_contour, [max_contour], -1, 255, -1)  # 填充轮廓

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mean_val = cv2.mean(hsv_img, mask=mask_contour)[:3]  # 获取 H,S,V 均值

# Step 4: 根据均值动态调整 HSV 范围
h_margin, s_margin, v_margin = 10, 50, 40  # 可调参数
lower_HSV = np.array([
    max(0, mean_val[0] - h_margin),
    max(0, mean_val[1] - s_margin),
    max(0, mean_val[2] - v_margin)
], np.uint8)
upper_HSV = np.array([
    min(179, mean_val[0] + h_margin),
    min(255, mean_val[1] + s_margin),
    min(255, mean_val[2] + v_margin)
], np.uint8)

# Step 5: 应用动态阈值
mask_HSV = cv2.inRange(hsv_img, lower_HSV, upper_HSV)
final_mask = cv2.bitwise_and(mask_YCrCb, mask_HSV)

# 形态学处理
kernel = np.ones((3, 3), np.uint8)
final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
final_image = np.zeros_like(img)
if contours:
    max_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(final_image, [max_contour], -1, (0, 255, 0), 2)

# 显示结果
cv2.namedWindow("Final Mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Final Mask",800,600)
cv2.imshow("Final Mask", final_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()