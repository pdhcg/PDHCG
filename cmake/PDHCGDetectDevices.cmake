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

# Keep the requested selection in the cache. PDHCG_DEVICES below is the
# resolved list for this configure, so AUTO is evaluated again on reconfigure.
if(DEFINED PDHCG_DEVICE AND NOT DEFINED PDHCG_DEVICES)
    set(PDHCG_DEVICES "${PDHCG_DEVICE}" CACHE STRING "AUTO or a semicolon-separated list of devices")
else()
    set(PDHCG_DEVICES "AUTO" CACHE STRING "AUTO or a semicolon-separated list of devices")
endif()
string(TOUPPER "${PDHCG_DEVICES}" PDHCG_DEVICES)
string(REPLACE "," ";" PDHCG_DEVICES "${PDHCG_DEVICES}")
list(REMOVE_DUPLICATES PDHCG_DEVICES)
if(NOT PDHCG_DEVICES)
    message(FATAL_ERROR "PDHCG_DEVICES must be AUTO or a list containing CPU and/or CUDA")
endif()
if("AUTO" IN_LIST PDHCG_DEVICES AND NOT PDHCG_DEVICES STREQUAL "AUTO")
    message(FATAL_ERROR "Use PDHCG_DEVICES=AUTO by itself, or explicitly list CPU and/or CUDA")
endif()

set(_PDHCG_REQUESTED_DEVICES "${PDHCG_DEVICES}")
set(_PDHCG_CPU_STATUS "not requested")
set(_PDHCG_CUDA_STATUS "not requested")

if(PDHCG_DEVICES STREQUAL "AUTO")
    set(PDHCG_DEVICES "")
    find_package(OpenMP QUIET COMPONENTS C)
    set(BLA_SIZEOF_INTEGER 4)
    find_package(LAPACK QUIET)
    set(_pdhcg_missing_cpu_dependencies)
    if(NOT OpenMP_C_FOUND OR NOT TARGET OpenMP::OpenMP_C)
        list(APPEND _pdhcg_missing_cpu_dependencies "OpenMP C")
    endif()
    if(NOT LAPACK_FOUND)
        list(APPEND _pdhcg_missing_cpu_dependencies "LP64 BLAS/LAPACK")
    endif()
    if(_pdhcg_missing_cpu_dependencies)
        list(JOIN _pdhcg_missing_cpu_dependencies ", " _pdhcg_missing_cpu_text)
        set(_PDHCG_CPU_STATUS "skipped (missing ${_pdhcg_missing_cpu_text})")
        message(STATUS "PDHCG AUTO: CPU ${_PDHCG_CPU_STATUS}")
    else()
        list(APPEND PDHCG_DEVICES CPU)
        set(_PDHCG_CPU_STATUS "enabled (OpenMP C, LP64 BLAS/LAPACK)")
    endif()
    unset(_pdhcg_missing_cpu_dependencies)
    unset(_pdhcg_missing_cpu_text)

    # Probe the build toolchain, not the hardware on the build machine.
    # CheckLanguage tries enabling CUDA in a separate project; no GPU is used.
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        find_package(CUDAToolkit 12.4 QUIET)
        if(CUDAToolkit_FOUND)
            set(_pdhcg_missing_cuda_libraries)
            foreach(library cudart cublas cusolver cusparse)
                if(NOT TARGET CUDA::${library})
                    list(APPEND _pdhcg_missing_cuda_libraries "CUDA::${library}")
                endif()
            endforeach()
            if(_pdhcg_missing_cuda_libraries)
                list(JOIN _pdhcg_missing_cuda_libraries ", " _pdhcg_missing_cuda_text)
                set(_PDHCG_CUDA_STATUS "skipped (missing ${_pdhcg_missing_cuda_text})")
                unset(_pdhcg_missing_cuda_text)
            else()
                enable_language(CUDA)
                # Prefer CUDA as the default whenever it is available.
                list(PREPEND PDHCG_DEVICES CUDA)
                set(_PDHCG_CUDA_STATUS "enabled (Toolkit ${CUDAToolkit_VERSION})")
            endif()
            unset(_pdhcg_missing_cuda_libraries)
        else()
            set(_PDHCG_CUDA_STATUS "skipped (CUDA Toolkit >= 12.4 is unavailable)")
        endif()
    else()
        set(_PDHCG_CUDA_STATUS "skipped (no working CUDA compiler)")
    endif()
    if(NOT "CUDA" IN_LIST PDHCG_DEVICES)
        message(STATUS "PDHCG AUTO: CUDA ${_PDHCG_CUDA_STATUS}")
    endif()
    if(NOT PDHCG_DEVICES)
        message(FATAL_ERROR
            "PDHCG AUTO found no buildable devices.\n"
            "  CPU: ${_PDHCG_CPU_STATUS}\n"
            "  CUDA: ${_PDHCG_CUDA_STATUS}\n"
            "Install the build dependencies for at least one backend.")
    endif()
else()
    foreach(device IN LISTS PDHCG_DEVICES)
        if(NOT device STREQUAL "CPU" AND NOT device STREQUAL "CUDA")
            message(FATAL_ERROR "Unsupported device '${device}'; choose AUTO, CPU and/or CUDA")
        endif()
    endforeach()
    if("CPU" IN_LIST PDHCG_DEVICES)
        find_package(OpenMP REQUIRED COMPONENTS C)
        set(BLA_SIZEOF_INTEGER 4)
        find_package(LAPACK REQUIRED)
        set(_PDHCG_CPU_STATUS "enabled (OpenMP C, LP64 BLAS/LAPACK)")
    endif()
    if("CUDA" IN_LIST PDHCG_DEVICES)
        # An explicit request must fail instead of silently dropping a backend.
        enable_language(CUDA)
        find_package(CUDAToolkit 12.4 REQUIRED)
        foreach(library cudart cublas cusolver cusparse)
            if(NOT TARGET CUDA::${library})
                message(FATAL_ERROR "CUDA backend requires CUDA::${library}")
            endif()
        endforeach()
        set(_PDHCG_CUDA_STATUS "enabled (Toolkit ${CUDAToolkit_VERSION})")
    endif()
endif()

set(PDHCG_DEFAULT_DEVICE "" CACHE STRING "Default backend; empty selects the first resolved device")
string(TOUPPER "${PDHCG_DEFAULT_DEVICE}" PDHCG_DEFAULT_DEVICE)
if(NOT PDHCG_DEFAULT_DEVICE)
    list(GET PDHCG_DEVICES 0 PDHCG_DEFAULT_DEVICE)
endif()
if(NOT PDHCG_DEFAULT_DEVICE IN_LIST PDHCG_DEVICES)
    message(FATAL_ERROR "PDHCG_DEFAULT_DEVICE must be one of the compiled devices: ${PDHCG_DEVICES}")
endif()
