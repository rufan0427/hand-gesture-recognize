import cv2
import numpy as np

# 全局变量存储滑动条值
bg_lower = [0, 0, 0]
bg_upper = [255, 200, 200]

def update_mask():
    global img, mask, mask2, mask_bg
    
    # 更新mask2的范围，mask2用于去除背景
    mask2 = cv2.inRange(img, np.array(bg_lower, np.uint8), np.array(bg_upper, np.uint8))
    # 反转背景掩码
    mask_bg = cv2.bitwise_not(mask2)
    #结合肤色掩码
    combined_mask = cv2.bitwise_and(mask, mask_bg)
    
    # 形态学处理
    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    contour,_=cv2.findContours(final_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    length=len(contour)
    maxarea=-1
    for i in range(length):
        temp=contour[i]
        this_area=cv2.contourArea(temp)
        if this_area>maxarea:
            maxarea=this_area
            id=i
    final_image = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    cv2.drawContours(final_image,[contour[id]],0,(0,255,0),2)
    final_image_green = final_image[:, :, 1] 
    cv2.imshow("identify_image",np.hstack((final_mask, final_image_green)))

# 滑动条回调函数
def on_lower_B(val): bg_lower[0] = val; update_mask()
def on_lower_G(val): bg_lower[1] = val; update_mask()
def on_lower_R(val): bg_lower[2] = val; update_mask()
def on_upper_B(val): bg_upper[0] = val; update_mask()
def on_upper_G(val): bg_upper[1] = val; update_mask()
def on_upper_R(val): bg_upper[2] = val; update_mask()

# 读取图像
img = cv2.imread("D:/Desktop/contest/archive3/8.jpg")

# 初始肤色检测
YCrCb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
lower_skin = np.array([0, 139, 85], np.uint8)
upper_skin = np.array([255, 173, 120], np.uint8)
mask = cv2.inRange(YCrCb_img, lower_skin, upper_skin)

# 创建窗口和滑动条
cv2.namedWindow("identify_image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("identify_image", 1400, 800)

# 创建背景范围调整滑动条
cv2.createTrackbar("Lower B", "identify_image", bg_lower[0], 255, on_lower_B)
cv2.createTrackbar("Lower G", "identify_image", bg_lower[1], 255, on_lower_G)
cv2.createTrackbar("Lower R", "identify_image", bg_lower[2], 255, on_lower_R)
cv2.createTrackbar("Upper B", "identify_image", bg_upper[0], 255, on_upper_B)
cv2.createTrackbar("Upper G", "identify_image", bg_upper[1], 255, on_upper_G)
cv2.createTrackbar("Upper R", "identify_image", bg_upper[2], 255, on_upper_R)

# 初始更新
update_mask()

cv2.waitKey(0)
cv2.destroyAllWindows()