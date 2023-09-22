#!/home/nvidia/anaconda3/envs/yolov5env/bin/python
# -*- coding: utf-8 -*-

import sys
import os

sys.path.insert(0, "/home/nvidia/anaconda3/envs/yolov5env/lib/python3.8/site-packages")  # add this setence we can import the torch
sys.path.append("/home/nvidia/yolov5_ros2/src/yolov5_ros2/yolov5_ros2")  
# current_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
# new_ld_library_path = "/home/nvidia/anaconda3/envs/yolov5env/lib:" + current_ld_library_path 
# os.environ['LD_LIBRARY_PATH'] = new_ld_library_path
# print(sys.path)

import cv2
import torch  
import rclpy  # 此处改为ROS2版本
from rclpy.node import Node  # 引入ROS2的Node
import numpy as np

# 注意代码第一行是python的路径，需要根据自己的环境进行修改,在终端中输入which python可以查看python的路径
from pixPos import getPos  
from std_msgs.msg import Int32
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from yolov5_ros2_msgs.msg import BoundingBox, BoundingBoxes  
import time

class Yolo_Dect(Node):
    def __init__(self, name):
        super().__init__(name)  # 初始化节点
        
        # load parameters
        # yolov5_path = rospy.get_param('/yolov5_path', '')
        # 标号为0的相机为前部相机，标号为1的相机为后部相机
        yolov5_path = '/home/nvidia/yolov5_ros2/src/yolov5_ros2/yolov5_ros2/yolov5'
        camera_front_topic = '/camera0/image_raw'  # 前部相机话题名称
        camera_rear_topic = '/camera1/image_raw'  # 后部相机话题名称
        # weight_path = rospy.get_param('~weight_path', '')
        weight_path = '/home/nvidia/yolov5_ros2/src/yolov5_ros2/yolov5_ros2/weights/best1.pt'

        direction_topic = '/my_topic'  # 这儿写上要订阅到的运动方向话题

        # pub_topic = rospy.get_param('~pub_topic', '/yolov5/BoundingBoxes')
        pub_topic = '/yolov5/BoundingBoxes'

        # self.camera_frame = rospy.get_param('~camera_frame', '')
        self.camera_frame = 'camera_color_frame'

        # conf = rospy.get_param('~conf', '0.5')
        conf = '0.4'
        
        # load local repository(YoloV5:v6.0)
        self.model = torch.hub.load(yolov5_path, 'custom', path=weight_path, source='local') 
        
        # which device will be used
        self.model.cuda()  # 使用GPU进行目标检测

        self.model.conf = conf
        self.color_image = Image()
        self.depth_image = Image()
        self.getImageStatus = False

        # Load class color
        self.classes_colors = {}
        
        self.camera_sub_front = self.create_subscription(Image, camera_front_topic, self.image_callback_front, 10)
        self.camera_sub_rear = self.create_subscription(Image, camera_rear_topic, self.image_callback_rear, 10)
        
        # 通过调用汽车运动话题来决定需要订阅哪个相机话题
        self.direction_sub = self.create_subscription(Int32, direction_topic, self.direction_callback, 10)
       
        # output publishers
        # 创建一个发布者，发布目标检测的结果
        self.position_pub = self.create_publisher(BoundingBoxes, pub_topic, 1)
        
        # 创建一个发布者，发布目标检测的图像
        # self.image_pub = self.create_publisher(Image, '/yolov5/detection_image', 1)
        self.direction = 0
        self.active_sub = None
     
    # 此处为自定义的通过订阅方向话题来控制相机开启的函数，
    def direction_callback(self, msg):
     
        self.direction = msg.data
        if self.direction == 0:
            self.active_sub = self.camera_sub_front
            self.get_logger().info("Front-camera!")     
        elif self.direction == 1:
            self.active_sub = self.camera_sub_rear
            self.get_logger().info("Rear-camera!")

    def image_callback_front(self, image):  # 图像处理，进来一个图像就检测这幅图像上的所有目标
        if self.active_sub == self.camera_sub_front:
            camera_detc_flag = 0  # 定义一个标志位用来进行可行驶区域裁剪，当0时进行前部相机区域裁剪
            self.boundingBoxes = BoundingBoxes()
            self.boundingBoxes.header = image.header
            self.boundingBoxes.image_header = image.header
            self.getImageStatus = True
            self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
                image.height, image.width, -1)
            self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

            # The following three lines of code are for image dedistortion, release the code when use.
            # distortion = np.array([0, 0, 0, 0, 0])
            # camera_matrix = np.array([[100, 0, 50], [0, 100, 100], [0, 0, 1]])
            # self.color_image = cv2.undistort(self.color_image, camera_matrix, distortion)

            results = self.model(self.color_image)  # error exist
            boxs = results.pandas().xyxy[0].values
            if boxs.shape[0] == 0:  
                boxs = [[0, 0, 0, 0, 0, 'background']]
            
            self.dectshow(self.color_image, boxs, image.height, image.width, camera_detc_flag)
            cv2.waitKey(2)

    def image_callback_rear(self, image):  # 图像处理，进来一个图像就检测这幅图像上的所有目标
        if self.active_sub == self.camera_sub_rear:
            camera_detc_flag = 1  # 定义一个标志位用来进行可行驶区域裁剪，当1时进行后部相机区域裁剪
            self.boundingBoxes = BoundingBoxes()
            self.boundingBoxes.header = image.header
            self.boundingBoxes.image_header = image.header
            self.getImageStatus = True
            self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
                image.height, image.width, -1)
            self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

            # The following three lines of code are for image dedistortion, release the code when use.
            # distortion = np.array([0, 0, 0, 0, 0])
            # camera_matrix = np.array([[100, 0, 50], [0, 100, 100], [0, 0, 1]])
            # self.color_image = cv2.undistort(self.color_image, camera_matrix, distortion)

            results = self.model(self.color_image)  # error exist
    
            boxs = results.pandas().xyxy[0].values
            if boxs.shape[0] == 0:  
                boxs = [[0, 0, 0, 0, 0, 'background']]
            
            self.dectshow(self.color_image, boxs, image.height, image.width, camera_detc_flag)
            cv2.waitKey(2)

    def dectshow(self, org_img, boxs, height, width, camera_detc_flag):

        if camera_detc_flag == 0:
            region = [(700, 200), (1300, 200), (1400, 700), (600, 700)]  # 这是由四个像素点围成的四边形,前相机裁剪
        if camera_detc_flag == 1:
            region = [(500, 100), (1750, 100), (1800, 800), (140, 800)]  # 后相机裁剪
        
        points = np.array(region)  # 要画出可行驶区域范围
        img = org_img.copy()
        count = 0

        for box in boxs:  # 循环所有的方形框
            boundingBox = BoundingBox()
            boundingBox.probability =np.float64(box[4])
            box[0] = np.int64(box[0])
            box[1] = np.int64(box[1])
            box[2] = np.int64(box[2])
            box[3] = np.int64(box[3])
            boundingBox.xmin = np.float64(box[0])
            boundingBox.ymin = np.float64(box[1])
            boundingBox.xmax = np.float64(box[2])
            boundingBox.ymax = np.float64(box[3])
            boundingBox.object = box[-1]
            
            target_point = ((boundingBox.xmin + boundingBox.xmax) / 2, boundingBox.ymax)  # 计算出来目标检测矩形框的下部线的中点
            # print(target_point)
            result = self.is_point_in(target_point, region)  # 得到判断的结果

            # 根据中心点的坐标来判断目标的位姿信息
            center_point_u = (boundingBox.xmin + boundingBox.xmax) / 2
            center_point_v = boundingBox.ymax

            pos_array = np.zeros((3, ))
            pos_array = getPos(center_point_u, center_point_v)

            boundingBox.posx = np.float64(pos_array[0])
            boundingBox.posy = np.float64(pos_array[1])
            boundingBox.posz = np.float64(pos_array[2])

            if result:  # 如果在区域内部，那么肯定就检测到了目标
                count += 1  # 此时障碍物的数量加1
            else:  # 如果目标不在区域内
                if boundingBox.object == 'background':  # 如果没有检测到目标发布背景信息
                    count = 0
                else:  # 如果检测到目标也发布背景信息
                    count = 0
            
            boundingBox.num = np.float64(count) # 有几个障碍物，发布话题中障碍物数量就为几

            if box[-1] in self.classes_colors.keys():
                color = self.classes_colors[box[-1]]
            else:
                color = np.random.randint(0, 183, 3)
                self.classes_colors[box[-1]] = color

            cv2.rectangle(img, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (int(color[0]),int(color[1]), int(color[2])), 2)
            
            cv2.polylines(img, [points], isClosed=True, color=(34, 139, 34), thickness=4)

            # 可行驶区域文字
            text = "Freespace"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 3
            text_color = (0, 0, 255)
            text_position = (points[0][0], points[0][1] - 10)
            cv2.putText(img, text, text_position, font, font_scale, text_color, thickness)

            if box[1] < 20:
                text_pos_y = box[1] + 30
            else:
                text_pos_y = box[1] - 10
                
            cv2.putText(img, box[-1],
                        (int(box[0]), int(text_pos_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
            
            if result:  # 如果目标在可行驶区域内才将障碍物信息追加到话题消息内
                self.boundingBoxes.bounding_boxes.append(boundingBox)

        self.position_pub.publish(self.boundingBoxes)

        # self.publish_image(img, height, width)
        cv2.namedWindow('Object-Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Object-Detection', 960, 540)
        cv2.imshow('Object-Detection', img)

    # def publish_image(self, imgdata, height, width):
    #     image_temp = Image()
    #     # header = Header(stamp=rclpy.get_clock().now())  # 这个获取时间的方式不知道对不对
    #     header = Header()
    #     header.frame_id = self.camera_frame
    #     image_temp.height = height
    #     image_temp.width = width
    #     image_temp.encoding = 'bgr8'
    #     image_temp.data = np.array(imgdata).tobytes()
    #     image_temp.header = header
    #     image_temp.step = width * 3
    #     self.image_pub.publish(image_temp)

    # 判断一个像素点是否位于由四个像素坐标点围成的四边形内
    def is_point_in(self, target_point, rectangle):

        x, y = target_point
        vertices = rectangle

        intersect_count = 0
        num_vertices = len(vertices)

        for i in range(num_vertices):
            j = (i + 1) % num_vertices
            if (vertices[i][1] < y and vertices[j][1] >= y) or (vertices[j][1] < y and vertices[i][1] >= y):
                if vertices[i][0] + (y - vertices[i][1]) / (vertices[j][1] - vertices[i][1]) * (vertices[j][0] - vertices[i][0]) < x:
                    intersect_count += 1

        if intersect_count % 2 == 1:
            return True  # 如果在四边形内部就返回True
        else:
            return False  # 如果在四边形外部就返回False


def main():

    rclpy.init(args=None)  # 初始化ROS2客户端

    yolo_dect = Yolo_Dect("yolov5_ros2_node")  # 创建节点对象

    rclpy.spin(yolo_dect)

    yolo_dect.destroy_node()

    rclpy.shutdown()
    
if __name__ == "__main__":

    main()

