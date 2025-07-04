import cv2
import numpy as np

img = cv2.imread("D:/Desktop/contest/week 1/image.jpg")

# Step 1: YCrCb 肤色检测
YCrCb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
lower_skin = np.array([0, 139, 85], np.uint8)
upper_skin = np.array([255, 173, 120], np.uint8)
mask_YCrCb = cv2.inRange(YCrCb_img, lower_skin, upper_skin)

# Step 2: HSV 肤色检测

HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_HSV = np.array([0, 0, 50], np.uint8)
upper_HSV = np.array([45, 255, 255], np.uint8)
mask_HSV = cv2.inRange(HSV_img, lower_HSV, upper_HSV)

# 结合 YCrCb 和 HSV 的掩码
combined_mask = cv2.bitwise_and(mask_YCrCb, mask_HSV)

# Step 3: Canny 边缘检测 + 反转（保留平滑区域）
edges = cv2.Canny(img, 50, 150)
mask_edges = cv2.bitwise_not(edges)  # 反转边缘

# Step 4: 结合肤色掩码和边缘掩码
final_mask = cv2.bitwise_and(mask_YCrCb, mask_edges)

# 形态学处理（去噪）
kernel = np.ones((3, 3), np.uint8)
final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

# 查找最大轮廓并绘制
contours, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
final_image = np.zeros_like(img)
if contours:
    max_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(final_image, [max_contour], -1, (0, 255, 0), 2)

# 显示结果
cv2.namedWindow("identify_image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("identify_image", 700, 600)
cv2.imshow("identify_image", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()