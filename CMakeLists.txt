cmake_minimum_required(VERSION 3.8)
project(doppler_icp_stitcher_open3d_1)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Enable CUDA
enable_language(CUDA)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find ROS2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rosbag2_cpp REQUIRED)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Find Open3D - multiple search methods
find_package(Open3D QUIET)
if(NOT Open3D_FOUND)
    find_path(Open3D_INCLUDE_DIRS
        NAMES open3d/Open3D.h
        PATHS
            /usr/local/include
            /usr/include
            ${CMAKE_INSTALL_PREFIX}/include
    )

    find_library(Open3D_LIBRARIES
        NAMES Open3D open3d
        PATHS
            /usr/local/lib
            /usr/local/lib64
            /usr/lib
            /usr/lib64
            ${CMAKE_INSTALL_PREFIX}/lib
    )

    if(Open3D_INCLUDE_DIRS AND Open3D_LIBRARIES)
        set(Open3D_FOUND TRUE)
        message(STATUS "Found Open3D: ${Open3D_LIBRARIES}")
    else()
        message(FATAL_ERROR "Open3D not found. Please install Open3D and set Open3D_DIR")
    endif()
endif()

# ------------------------------------------------------------------------------
#  SOURCE FILES
# ------------------------------------------------------------------------------

# Create executable with CUDA support
add_executable(stitch_node_Version
    src/stitch_node_Version.cpp
)

# Mark the target as needing CUDA
set_target_properties(stitch_node_Version PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "native"
)

# ------------------------------------------------------------------------------
#  INCLUDE PATHS
# ------------------------------------------------------------------------------

target_include_directories(stitch_node_Version PRIVATE
    ${EIGEN3_INCLUDE_DIRS}
    ${Open3D_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ------------------------------------------------------------------------------
#  LINK LIBRARIES
# ------------------------------------------------------------------------------

target_link_libraries(stitch_node_Version
    Eigen3::Eigen
    ${Open3D_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    pthread
)

ament_target_dependencies(stitch_node_Version
    rclcpp
    sensor_msgs
    geometry_msgs
    std_msgs
    tf2_ros
    rosbag2_cpp
)

# ------------------------------------------------------------------------------
#  COMPILE OPTIONS
# ------------------------------------------------------------------------------

if(CMAKE_CUDA_COMPILER)
    target_compile_options(stitch_node_Version PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-arch=native>
    )
endif()

# ------------------------------------------------------------------------------
#  INSTALL TARGETS
# ------------------------------------------------------------------------------

install(TARGETS
    stitch_node_Version
    DESTINATION lib/${PROJECT_NAME}
)

# Install include directories
install(DIRECTORY include/
  DESTINATION include/
)

# Install launch directory *only if it exists*
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
  )
endif()

ament_package()