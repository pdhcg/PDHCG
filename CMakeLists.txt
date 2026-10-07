# Copyright 2026 Hongpei Li
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -----------------------------------------------------------------------------
# ROOT CMakeLists.txt for PDHCG (Unified Build System)
# -----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)

set(_PDHCG_CUDA_ARCHITECTURES_EXPLICIT FALSE)
if(DEFINED CMAKE_CUDA_ARCHITECTURES OR DEFINED ENV{CUDAARCHS})
    set(_PDHCG_CUDA_ARCHITECTURES_EXPLICIT TRUE)
endif()

project(pdhcg LANGUAGES C)
include(cmake/PDHCGDetectDevices.cmake)
if("CPU" IN_LIST PDHCG_DEVICES)
    include(cmake/PDHCGBlasThreads.cmake)
endif()

set(PDHCG_VERSION_MAJOR 0)
set(PDHCG_VERSION_MINOR 4)
set(PDHCG_VERSION_PATCH 0)

set(PDHCG_VERSION "${PDHCG_VERSION_MAJOR}.${PDHCG_VERSION_MINOR}.${PDHCG_VERSION_PATCH}")
add_compile_definitions(PDHCG_VERSION="${PDHCG_VERSION}")

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# C/C++ standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Set default build type to Release if not specified (Optimized builds)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if("CUDA" IN_LIST PDHCG_DEVICES AND NOT _PDHCG_CUDA_ARCHITECTURES_EXPLICIT)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
        set(_PDHCG_DEFAULT_CUDA_ARCHITECTURES all)
    else()
        set(_PDHCG_DEFAULT_CUDA_ARCHITECTURES 60 70 75 80 86 89 90)
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${_PDHCG_DEFAULT_CUDA_ARCHITECTURES}"
        CACHE STRING "CUDA architectures to build" FORCE)
    unset(_PDHCG_DEFAULT_CUDA_ARCHITECTURES)
endif()
unset(_PDHCG_CUDA_ARCHITECTURES_EXPLICIT)

# Global Compile flags (corresponding to CFLAGS/NVCCFLAGS)
add_compile_options(-fPIC -O3 -Wall -Wextra -g)

# Windows compatibility
if (WIN32)
    add_definitions(-Dstrtok_r=strtok_s)
endif()

# CUDA standards and RDC (Relocatable Device Code)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# CONTROL OPTIONS
# -----------------------------------------------------------------------------
include(CMakeDependentOption)

option(PDHCG_BUILD_STATIC_LIB "Build the PDHCG static library" ON)
option(PDHCG_BUILD_SHARED_LIB "Build the PDHCG shared library" ON)
option(PDHCG_COMPILE_PREFOS "Download and use PreFOS for presolving" OFF)
option(PDHCG_PREFOS_ENABLE_CUDA "Enable the PreFOS CUDA propagation backend" ON)
option(PDHCG_COMPILE_DISTRIBUTED "Enable distributed computing with MPI" OFF)
option(PDHCG_ABSOLUTE_ONLY_TERMINATION
       "Use absolute residuals and objective gap for optimal termination"
       OFF)

# format: cmake_dependent_option(OPTION "docstring" DEFAULT_VALUE "DEPENDENCY_VARIABLE" FORCE_OFF_VALUE)
cmake_dependent_option(PDHCG_BUILD_PYTHON "Build the PDHCG Python bindings" OFF
                       "PDHCG_BUILD_STATIC_LIB" OFF)

cmake_dependent_option(PDHCG_BUILD_CLI "Build the PDHCG command-line executable" ON
                       "PDHCG_BUILD_STATIC_LIB" OFF)

cmake_dependent_option(PDHCG_BUILD_TESTS "Build the PDHCG test suite" OFF
                       "PDHCG_BUILD_STATIC_LIB" OFF)

# -----------------------------------------------------------------------------
# FIND DEPENDENCIES
# -----------------------------------------------------------------------------
# C++ is only needed by bindings or optional external/distributed dependencies.
if(PDHCG_BUILD_PYTHON OR PDHCG_COMPILE_PREFOS OR PDHCG_COMPILE_DISTRIBUTED)
    enable_language(CXX)
endif()
if("CPU" IN_LIST PDHCG_DEVICES)
    if(PDHCG_COMPILE_PREFOS AND PDHCG_PREFOS_ENABLE_CUDA)
        message(FATAL_ERROR "Builds including CPU require PDHCG_PREFOS_ENABLE_CUDA=OFF when PreFOS is enabled")
    endif()
endif()
if(PDHCG_COMPILE_DISTRIBUTED)
    if(NOT "CUDA" IN_LIST PDHCG_DEVICES)
        message(FATAL_ERROR "Distributed execution requires the CUDA backend; CPU uses single-process threads")
    endif()
    find_package(MPI REQUIRED COMPONENTS C CXX)
    find_library(PDHCG_NCCL_LIBRARY NAMES nccl
        PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} $ENV{NCCL_ROOT}/lib
              /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)
    if(NOT PDHCG_NCCL_LIBRARY)
        set(PDHCG_NCCL_LIBRARY nccl)
    endif()
endif()
find_package(ZLIB REQUIRED)

if (PDHCG_BUILD_PYTHON)
    # Dependencies required only for Python bindings
    find_package(pybind11 CONFIG REQUIRED)
    find_package(Python3 COMPONENTS Interpreter REQUIRED) # For versioning script and pybind11
endif()

include(FetchContent)

# -----------------------------------------------------------------------------
# PreFOS Presolver Integration
# -----------------------------------------------------------------------------
if(PDHCG_COMPILE_PREFOS)
    set(PREFOS_VERSION_TAG "v0.1.0" CACHE STRING "PreFOS git tag/branch to use")

    FetchContent_Declare(
      prefos
      GIT_REPOSITORY https://github.com/Lhongpei/PreFOS.git
      GIT_TAG        ${PREFOS_VERSION_TAG}
    )

    # Embed PreFOS statically so installed PDHCG binaries have no additional
    # presolver runtime dependency. The source override
    # FETCHCONTENT_SOURCE_DIR_PREFOS remains available for local development.
    set(_PDHCG_BUILD_SHARED_LIBS_WAS_DEFINED FALSE)
    if(DEFINED BUILD_SHARED_LIBS)
        set(_PDHCG_BUILD_SHARED_LIBS_WAS_DEFINED TRUE)
        set(_PDHCG_SAVED_BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS}")
    endif()
    set(BUILD_SHARED_LIBS OFF)
    set(PREFOS_BUILD_TESTING OFF)
    set(PREFOS_ENABLE_CUDA ${PDHCG_PREFOS_ENABLE_CUDA})
    FetchContent_MakeAvailable(prefos)
    if(_PDHCG_BUILD_SHARED_LIBS_WAS_DEFINED)
        set(BUILD_SHARED_LIBS "${_PDHCG_SAVED_BUILD_SHARED_LIBS}")
    else()
        unset(BUILD_SHARED_LIBS)
    endif()

    if(TARGET PreFOS::PreFOS)
        message(STATUS "PreFOS ${PREFOS_VERSION_TAG} configured successfully")
    else()
        message(FATAL_ERROR "PreFOS target was not created")
    endif()
else()
    message(STATUS "PreFOS integration disabled (PDHCG_COMPILE_PREFOS=OFF).")
endif()

# Optional definitions shared by the selected backends.
set(PDHCG_OPTIONAL_DEFINES "")
if(PDHCG_COMPILE_PREFOS)
    list(APPEND PDHCG_OPTIONAL_DEFINES PREFOS_AVAILABLE)
    if(PDHCG_PREFOS_ENABLE_CUDA)
        list(APPEND PDHCG_OPTIONAL_DEFINES PDHCG_PREFOS_CUDA_ENABLED)
    endif()
endif()
if(PDHCG_ABSOLUTE_ONLY_TERMINATION)
    list(APPEND PDHCG_OPTIONAL_DEFINES PDHCG_ABSOLUTE_ONLY_TERMINATION)
endif()

if(PDHCG_BUILD_TESTS)
    enable_testing()
endif()
include(cmake/PDHCGDevice.cmake)
foreach(device IN LISTS PDHCG_DEVICES)
    pdhcg_add_device("${device}")
endforeach()
pdhcg_add_interface()
if(PDHCG_BUILD_PYTHON)
    add_subdirectory(python_bindings)
endif()
install(DIRECTORY include/ DESTINATION include/ FILES_MATCHING PATTERN "*.h")

# Report the resolved targets after all backend and Python targets are defined.
list(JOIN PDHCG_DEVICES ", " _pdhcg_enabled_devices)
message(STATUS "")
message(STATUS "PDHCG device summary")
message(STATUS "  Requested devices : ${_PDHCG_REQUESTED_DEVICES}")
message(STATUS "  Devices to build  : ${_pdhcg_enabled_devices}")
message(STATUS "  Default device    : ${PDHCG_DEFAULT_DEVICE}")
message(STATUS "  CPU               : ${_PDHCG_CPU_STATUS}")
if("CPU" IN_LIST PDHCG_DEVICES)
    message(STATUS "  CPU BLAS threads  : ${_PDHCG_CPU_BLAS_STATUS}")
endif()
message(STATUS "  CUDA              : ${_PDHCG_CUDA_STATUS}")
message(STATUS "")
unset(_pdhcg_enabled_devices)
