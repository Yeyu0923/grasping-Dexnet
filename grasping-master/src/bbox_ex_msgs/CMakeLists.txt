cmake_minimum_required(VERSION 2.8.12)
project(bboxes_ex_msgs)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

find_package(catkin REQUIRED
  COMPONENTS
    std_msgs
    sensor_msgs
    message_generation
)

add_message_files(
  FILES
    BoundingBox.msg
    BoundingBoxes.msg
    RGBD.msg
    GRconvnet.msg
)

generate_messages(
  DEPENDENCIES
    std_msgs
    sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    std_msgs
)