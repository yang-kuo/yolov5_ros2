#!/bin/bash
gnome-terminal -- bash -c 'cd /home/nvidia/gmsl_ros2_ws; source install/setup.bash; ros2 launch gmsl_ros2 gmsl_launch.py; exec bash'
sleep 2

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/nvidia/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/nvidia/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/nvidia/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/nvidia/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
cd /home/nvidia/yolov5_ros2

conda activate yolov5env

source install/setup.bash

ros2 run yolov5_ros2 yolov5_ros2









