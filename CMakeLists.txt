cmake_minimum_required(VERSION 3.15)
project(plcoding_cpp_core LANGUAGES CXX)

# find packages
find_package(pybind11 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS NumPy)

# set parameters
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# source module
pybind11_add_module(source
    plcoding/cpp_core/source.cpp
)
target_link_libraries(source PRIVATE Python3::NumPy fftw3)
install(TARGETS source DESTINATION plcoding/cpp_core)

# iterator module
pybind11_add_module(iterator
    plcoding/cpp_core/iterator.cpp
)
target_link_libraries(iterator PRIVATE Python3::NumPy fftw3)
install(TARGETS iterator DESTINATION plcoding/cpp_core)
