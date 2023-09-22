import os
import cv2

# 打开相机
cap = cv2.VideoCapture(0)

# 检查相机是否已打开
if not cap.isOpened():
    print("无法打开相机。")
    exit()

# 从相机读取图像
ret, frame = cap.read()

# 检查是否成功读取图像
if not ret:
    print("无法获取图像。")
    exit()

# 释放相机资源
cap.release()

# 获取当前工作目录
current_directory = os.getcwd()

# 保存图像到当前工作目录
image_filename = "camera_image.jpg"
image_path = os.path.join(current_directory, image_filename)
# frame = cv2.resize(frame, (1920, 1080))
cv2.imwrite(image_path, frame)

print(f"图像已保存至当前文件夹：{image_filename}")
