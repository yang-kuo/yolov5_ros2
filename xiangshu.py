import cv2

"""
该函数能够实现鼠标点击图片显示该点像素值
"""
# 鼠标回调函数
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 绘制像素坐标
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        # 显示像素坐标
        text = f'({x}, {y})'
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('Image', image)

# 读取图像
image = cv2.imread('camera_image.jpg')

# 创建窗口并显示图像
cv2.namedWindow('Image')
cv2.imshow('Image', image)

# 设置鼠标回调函数
cv2.setMouseCallback('Image', click_event)

# 等待按键按下
cv2.waitKey(0)
cv2.destroyAllWindows()
