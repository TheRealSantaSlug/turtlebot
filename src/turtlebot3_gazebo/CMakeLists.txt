################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 3.5)
project(turtlebot3_gazebo)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(MSVC)
  add_compile_definitions(_USE_MATH_DEFINES)
endif()

################################################################################
# Find ament packages and libraries for ament and system dependencies
################################################################################
find_package(ament_cmake REQUIRED)
find_package(gazebo REQUIRED)
find_package(gazebo_ros_pkgs REQUIRED)

################################################################################
# Build
################################################################################
link_directories(
  ${GAZEBO_LIBRARY_DIRS}
)

include_directories(
  include
  ${GAZEBO_INCLUDE_DIRS}
)


#add_library(obstacle1 SHARED models/turtlebot3_dqn_world/obstacle_plugin/obstacle1.cc)
#target_link_libraries(obstacle1 ${GAZEBO_LIBRARIES})
#
#add_library(obstacle2 SHARED models/turtlebot3_dqn_world/obstacle_plugin/obstacle2.cc)
#target_link_libraries(obstacle2 ${GAZEBO_LIBRARIES})
#
#add_library(obstacles SHARED models/turtlebot3_dqn_world/obstacle_plugin/obstacles.cc)
#target_link_libraries(obstacles ${GAZEBO_LIBRARIES})

################################################################################
# Install
################################################################################

install(DIRECTORY launch models rviz urdf worlds maps params
  DESTINATION share/${PROJECT_NAME}/
)

install(
  PROGRAMS
    src/lab4/task1.py
    src/lab4/task2.py
    src/lab4/task2_bonus.py
    src/lab4/task3.py
    src/lab4/dynamic_obstacles.py
  DESTINATION lib/${PROJECT_NAME}
)


################################################################################
# Macro for ament package
################################################################################
ament_export_include_directories(include)
ament_export_dependencies(gazebo_ros_pkgs)
ament_package()
