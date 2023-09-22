from launch import LaunchDescription
from launch_ros.actions import Node

# 显示Node的使用
def generate_launch_description():

    yolov5_ros2 = Node(
        package = "yolov5_ros2", # 被执行程序所属的功能包
        executable = "yolov5_ros2",  # 可执行程序

    )

    return LaunchDescription([yolov5_ros2])