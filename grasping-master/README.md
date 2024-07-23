
# GQ-CNN Neural Network for Grab Detection

## step1 ：Start Camera Node

cd catkin_ws

source ./devel/setup.bash

roslaunch realsense2_camera rs_camera.launch filters:=spatial,temporal,hole_filling clip_distance:=1

## step2 ：Start Data Receiving Node

cd grasping-master

source ./devel/setup.bash

roslaunch gqcnn primesense.launch

## step3 ：Start Grasping Node

cd grasping-master

source ./devel/setup.bash

roslaunch gqcnn grasp_planning_service.launch

## x86 requirements:

python==3.8

pip install -r requirements.txt

sudo apt install python3-pip

git clone https://github.com/BerkeleyAutomation/autolab_core.git

cd autolab_core

sudo python3 setup.py install

git clone https://github.com/BerkeleyAutomation/perception.git

cd perception

sudo python3 setup.py install

git clone https://github.com/BerkeleyAutomation/visualization.git

cd visualization

sudo python3 setup.py install

sudo apt install ros-noetic-ros-numpy

### For GPU

nvidia-detector 

sudo apt install nvidia-utils-(detector)

Installing the graphics card driver

Install CUDA

Install Cudnn

Install TensorRt

## arm64 requirements:

python==3.8

git clone https://github.com/BerkeleyAutomation/autolab_core.git

cd autolab_core

sudo python setup.py install

git clone https://github.com/BerkeleyAutomation/perception.git

cd perception

sudo python setup.py install

git clone https://github.com/BerkeleyAutomation/visualization.git

cd visualization

sudo python setup.py install

sudo apt-get install cmake

sudo pip install scikit-build

pip install --upgrade pip

python -m pip install opencv-python

pip install scipy

pip install matplotlib

pip install gputill

pip install psutil

pip install scikit-image

pip install scikit-learn

pip install multiprocess

pip install setproctitle

git clone https://github.com/ludwigschwardt/python-gnureadline.git

cd python-gnureadline/

sudo python setup.py install

pip install pyOpenGL

find a file _bz2.cpython-36m-aarch64-linux-gnu.so(maybe in /usr/lib/python3.6/lib-dynload) move to /usr/local/python3.X/lib/python3.7/lib-dynload and rename file to "37m"

pip install freetype-py

wget https://files.pythonhosted.org/packages/23/cd/31ee764b0ab2638c245cf4ac6b3e902434a5f10a5424c56d507c9b6b25e0/pyglet-1.4.10.zip

cd pyglet-1.4.10

sudo python setup.py install

pip install shapely

sudo pip install -U numpy==1.22 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig packaging h5py==3.6.0

sudo pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v51 tensorflow==2.11.0+nv23.01


