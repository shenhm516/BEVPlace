cmake_minimum_required(VERSION 3.10)
project(pointcloud_to_bev)

# 设置C++17标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 查找OpenCV
find_package(OpenCV REQUIRED)

# 查找PCL
find_package(PCL 1.10 REQUIRED)

# 包含目录
include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

# 链接目录
link_directories(${PCL_LIBRARY_DIRS})

# 添加编译定义
add_definitions(${PCL_DEFINITIONS})

# 添加可执行文件
add_executable(pointcloud_to_bev scan_to_bev.cpp)

# 链接库
target_link_libraries(pointcloud_to_bev ${PCL_LIBRARIES} ${OpenCV_LIBS})
