cmake_minimum_required(VERSION 3.18)

project(Main LANGUAGES C CXX)

find_package(OpenCV REQUIRED)

file(GLOB SRC
    "src/*.cpp"
)

add_executable(Main ${SRC})

target_link_libraries(Main PRIVATE
    ${OpenCV_LIBS}
)

target_include_directories(Main PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)
