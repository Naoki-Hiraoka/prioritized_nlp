cmake_minimum_required(VERSION 3.0.2)
project(prioritized_nlp_ifopt_sample)

add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  prioritized_nlp_ifopt)

catkin_package()

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
)

add_executable(sample0 src/sample0.cpp)
target_link_libraries(sample0
  ${catkin_LIBRARIES}
  )


install(TARGETS sample0
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
