from setuptools import setup
import os
from glob import glob
from urllib.request import urlretrieve
from setuptools import find_packages

package_name = 'yolov5_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob("launch/*_launch.py"))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yangkuo',
    maintainer_email='yangkuo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 'yolov5_ros2 = yolov5_ros2.yolov5_ros2:main' # 这是单相机检测
            'yolov5_ros2 = yolov5_ros2.yolov5_ros2_dir:main'  # 这是双相机切换检测

        ],
    },
)
