cmake_minimum_required(VERSION 3.20)

project(Main LANGUAGES CXX)

find_package(OpenCV REQUIRED)

file(GLOB SRC "src/*.cpp")

include_directories("src/")

add_executable(SlicSample "${CMAKE_SOURCE_DIR}/samples/slic_test.cpp" ${SRC})
target_link_libraries(SlicSample PRIVATE ${OpenCV_LIBS} )
target_include_directories(SlicSample PRIVATE ${OpenCV_INCLUDE_DIRS} )

add_executable(ClaheSample "${CMAKE_SOURCE_DIR}/samples/clahe_test.cpp" ${SRC})
target_link_libraries(ClaheSample PRIVATE ${OpenCV_LIBS} )
target_include_directories(ClaheSample PRIVATE ${OpenCV_INCLUDE_DIRS} )

add_executable(Sample "${CMAKE_SOURCE_DIR}/samples/test.cpp" ${SRC})
target_link_libraries(Sample PRIVATE ${OpenCV_LIBS} )
target_include_directories(Sample PRIVATE ${OpenCV_INCLUDE_DIRS} )

# add_executable(SiftSample ${SRC})
# target_link_libraries(Main1 PRIVATE ${OpenCV_LIBS} )
# target_include_directories(Main1 PRIVATE ${OpenCV_INCLUDE_DIRS} )
