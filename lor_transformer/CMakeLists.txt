cmake_minimum_required(VERSION 3.8)
project(lor_transformer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(image_geometry REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(lor_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

# Include directories
include_directories(include)

add_executable(lor_transformer
  src/lor_transformer.cpp
)

target_link_libraries(lor_transformer
  ${image_geometry_LIBRARIES}
  ${cv_bridge_LIBRARIES}
)

ament_target_dependencies(lor_transformer
  rclcpp
  sensor_msgs
  message_filters
  image_geometry
  cv_bridge
  lor_interfaces
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
)

install(TARGETS lor_transformer
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
