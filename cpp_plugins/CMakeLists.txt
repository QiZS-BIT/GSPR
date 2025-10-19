cmake_minimum_required(VERSION 3.0.2)

project(point2voxel)

set(PYTHON_EXECUTABLE /home/octane17/anaconda3/envs/gaussian_splatting/bin/python)
set(pybind11_DIR "/home/octane17/anaconda3/envs/gaussian_splatting/lib/python3.7/site-packages/pybind11/share/cmake/pybind11/")

include_directories(
  include
)

find_package(pybind11 REQUIRED)

pybind11_add_module(point2voxel src/point2voxel.cpp)

