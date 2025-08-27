cmake_minimum_required(VERSION 3.15)
project(plcoding_cpp_core LANGUAGES CXX)

# find packages
find_package(pybind11 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS NumPy)

# set parameters
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# functions module
pybind11_add_module(functions
    plcoding/cpp_core/functions.cpp
)
target_link_libraries(functions PRIVATE Python3::NumPy fftw3)
install(TARGETS functions DESTINATION plcoding/cpp_core)

# classics module
pybind11_add_module(classics
    plcoding/cpp_core/classics/decoder.cpp
    plcoding/cpp_core/classics/utils.cpp
)
target_link_libraries(classics PRIVATE Python3::NumPy fftw3)
install(TARGETS classics DESTINATION plcoding/cpp_core)

# monotone module
pybind11_add_module(monotone
    plcoding/cpp_core/monotone/decoder.cpp
    plcoding/cpp_core/monotone/utils.cpp
)
target_link_libraries(monotone PRIVATE Python3::NumPy fftw3)
install(TARGETS monotone DESTINATION plcoding/cpp_core)
