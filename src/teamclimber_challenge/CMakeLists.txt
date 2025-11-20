cmake_minimum_required(VERSION 3.8)
project(teamclimber_challenge)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(geometry_msgs REQUIRED) 
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(referee_pkg REQUIRED)
find_package(pose_msg REQUIRED)
find_package(CUDA REQUIRED)

# Set TensorRT path if not set in environment variables
set(TENSORRT_DIR "/home/zoupeng/TensorRT-10.14.1.48")

# Include TensorRT
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)
set(TENSORRT_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser)

include_directories(
  include
  include/YOLOv11
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR})
  
# Add source files
set(SOURCES
    src/vision_node.cpp
    src/image_deal.cpp
    src/shape_tools.cpp
    src/YOLOv11/YOLOv11.cpp
)

# Add CUDA source files
set(CUDA_SOURCES src/YOLOv11/preprocess.cu)

# Add headers
set(HEADERS
    include/image_deal.h
    include/shape_tools.h
    include/YOLOv11/YOLOv11.h
    include/YOLOv11/macros.h
    include/YOLOv11/logging.h
    include/YOLOv11/cuda_utils.h
    include/YOLOv11/preprocess.h
    include/YOLOv11/common.h
)

# Create an executable
cuda_add_executable(vision_node ${SOURCES} ${HEADERS} ${CUDA_SOURCES})

# Link libraries
target_link_libraries(vision_node
    ${typesupport_target}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)

ament_target_dependencies(vision_node
  rclcpp
  std_msgs
  OpenCV 
  geometry_msgs
  cv_bridge
  pose_msg
  referee_pkg
)

install(TARGETS
vision_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
