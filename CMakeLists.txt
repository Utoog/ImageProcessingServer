cmake_minimum_required(VERSION 3.6)
set(CMAKE_SUPPRESS_REGENERATION true)

project(ImageProcessingServer)

find_package(OpenCV 4.9.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(PROJECT_SOURCES
    main.cpp 
    inference.h
    inference.cpp 
)

add_executable(ImageProcessingServer ${PROJECT_SOURCES})
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ImageProcessingServer)
target_link_libraries(ImageProcessingServer ${OpenCV_LIBS})