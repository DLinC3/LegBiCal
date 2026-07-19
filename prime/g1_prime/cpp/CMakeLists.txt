cmake_minimum_required(VERSION 3.16)
project(g1cal_overlay CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(pinocchio REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system serialization)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
# Prefer the pybind11 installed in the active Python environment; some hosts
# expose an incomplete system CMake package.
execute_process(
  COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE G1CAL_PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
set(pybind11_DIR ${G1CAL_PYBIND11_CMAKE_DIR} CACHE PATH
  "pybind11 CMake package from the active environment" FORCE)
find_package(pybind11 CONFIG REQUIRED)

get_filename_component(G1CAL_ROOT ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
set(PRIME_SRC ${G1CAL_ROOT}/third_party/PRIME CACHE PATH
  "Vendored PRIME source tree")
set(PRIME_BUILD ${G1CAL_ROOT}/build/prime CACHE PATH
  "PRIME build directory containing libcrocoddyl.so")

if(NOT EXISTS ${PRIME_BUILD}/libcrocoddyl.so)
  message(FATAL_ERROR
    "libcrocoddyl.so not found under ${PRIME_BUILD}. Build the vendored "
    "PRIME first (the top-level superbuild does this automatically).")
endif()

add_library(prime_crocoddyl SHARED IMPORTED)
set_target_properties(prime_crocoddyl PROPERTIES
  IMPORTED_LOCATION ${PRIME_BUILD}/libcrocoddyl.so)
target_include_directories(prime_crocoddyl INTERFACE
  ${PRIME_SRC}/include
  ${PRIME_BUILD}/include
  ${PRIME_SRC}/experiments/common)
target_compile_definitions(prime_crocoddyl INTERFACE
  PINOCCHIO_ENABLE_COMPATIBILITY_WITH_VERSION_2
  CROCODDYL_WITH_MULTITHREADING
  CROCODDYL_WITH_NTHREADS=8)
target_link_libraries(prime_crocoddyl INTERFACE
  pinocchio::pinocchio Boost::filesystem Boost::system Boost::serialization
  OpenMP::OpenMP_CXX)

add_library(g1cal_overlay INTERFACE)
target_include_directories(g1cal_overlay INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(g1cal_overlay INTERFACE prime_crocoddyl)

add_executable(g1_motion_fie apps/g1_motion_fie.cpp)
target_link_libraries(g1_motion_fie PRIVATE g1cal_overlay)

# Persistent in-process entry point sharing the exact runner translation
# unit with the CLI, so both paths execute identical C++.
pybind11_add_module(_g1cal_cpp bindings/g1cal_pybind.cpp)
target_link_libraries(_g1cal_cpp PRIVATE g1cal_overlay)
set_target_properties(_g1cal_cpp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${G1CAL_ROOT}/python/g1cal)

enable_testing()
foreach(t
    test_dimensions
    test_sim_parity
    test_action_parity
    test_fd_derivatives
    test_inertia_invariance
    test_covariance_cost)
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} PRIVATE g1cal_overlay)
  target_compile_definitions(${t} PRIVATE
    G1CAL_SOURCE_ROOT="${G1CAL_ROOT}")
  add_test(NAME ${t} COMMAND ${t})
endforeach()
