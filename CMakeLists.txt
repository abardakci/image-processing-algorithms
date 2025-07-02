cmake_minimum_required(VERSION 3.20)

project(Main LANGUAGES CXX)

find_package(OpenCV REQUIRED)

file(GLOB SRC "src/*.cpp")

include_directories("src/")

add_executable(slic "${CMAKE_SOURCE_DIR}/samples/slic_test.cpp" ${SRC})
target_link_libraries(slic PRIVATE ${OpenCV_LIBS} )
target_include_directories(slic PRIVATE ${OpenCV_INCLUDE_DIRS} )

add_executable(clahe "${CMAKE_SOURCE_DIR}/samples/clahe_test.cpp" ${SRC})
target_link_libraries(clahe PRIVATE ${OpenCV_LIBS} )
target_include_directories(clahe PRIVATE ${OpenCV_INCLUDE_DIRS} )

add_executable(nlm "${CMAKE_SOURCE_DIR}/samples/nlm_test.cpp" ${SRC})
target_link_libraries(nlm PRIVATE ${OpenCV_LIBS} )
target_include_directories(nlm PRIVATE ${OpenCV_INCLUDE_DIRS} )

add_executable(sample "${CMAKE_SOURCE_DIR}/samples/test.cpp" ${SRC})
target_link_libraries(sample PRIVATE ${OpenCV_LIBS} )
target_include_directories(sample PRIVATE ${OpenCV_INCLUDE_DIRS} )
