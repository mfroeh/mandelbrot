cmake_minimum_required(VERSION 3.12)

project(mandel)

set(CXX_STANDARD 20)
set(CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)

include_directories(include ${CUDA_INCLUDE_DIRS})
enable_language(CUDA)

add_executable(mandel main.cu glad.c)
target_link_libraries(mandel PRIVATE glfw OpenGL::GL ${CUDA_LIBRARIES})
