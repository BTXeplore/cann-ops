cmake_minimum_required(VERSION 3.16)
set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "Target architecture")
set(CMAKE_ASC_RUN_MODE "npu" CACHE STRING "npu or cpu")
find_package(ASC REQUIRED)
project(sample LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 14)
add_executable(sample main.asc)
target_compile_options(sample PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
target_link_libraries(sample PRIVATE
    tiling_api register platform unified_dlog dl m graph_base)
