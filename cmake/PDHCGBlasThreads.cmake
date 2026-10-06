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

# Probe the actual link set, including static libraries. Do not inspect an
# unrelated BLAS already loaded by NumPy or require vendor development headers.
function(pdhcg_detect_blas_threads)
    # API detection must link even for toolchains whose other try-compiles only
    # build static archives. This override stays local to this function.
    set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
    include(CheckCSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES} OpenMP::OpenMP_C)
    if(NOT WIN32)
        list(APPEND CMAKE_REQUIRED_LIBRARIES m)
    endif()
    set(CMAKE_REQUIRED_FLAGS "${LAPACK_LINKER_FLAGS}")
    # Recheck on configure: users may switch the BLAS provider in an existing build.
    unset(_PDHCG_HAVE_MKL_THREADS CACHE)
    unset(_PDHCG_HAVE_OPENBLAS_THREADS CACHE)
    check_c_source_compiles("
        extern int MKL_Set_Num_Threads_Local(int);
        int main(void) { return MKL_Set_Num_Threads_Local(1); }
    " _PDHCG_HAVE_MKL_THREADS)
    if(_PDHCG_HAVE_MKL_THREADS)
        set(PDHCG_CPU_BLAS_DEFINE PDHCG_BLAS_MKL)
        set(_PDHCG_CPU_BLAS_STATUS "MKL (automatic, thread-local)")
    else()
        check_c_source_compiles("
            extern int openblas_get_num_threads(void);
            extern void openblas_set_num_threads(int);
            int main(void) {
                openblas_set_num_threads(1);
                return openblas_get_num_threads();
            }
        " _PDHCG_HAVE_OPENBLAS_THREADS)
        if(_PDHCG_HAVE_OPENBLAS_THREADS)
            set(PDHCG_CPU_BLAS_DEFINE PDHCG_BLAS_OPENBLAS)
            set(_PDHCG_CPU_BLAS_STATUS "OpenBLAS (automatic, shared scope)")
        else()
            set(PDHCG_CPU_BLAS_DEFINE "")
            set(_PDHCG_CPU_BLAS_STATUS "no supported API (serial BLAS needs no control)")
            message(STATUS "PDHCG: BLAS thread control API not found. Serial BLAS/LAPACK needs no setup; other providers must be configured for one thread.")
        endif()
    endif()
    cmake_pop_check_state()
    set(PDHCG_CPU_BLAS_DEFINE "${PDHCG_CPU_BLAS_DEFINE}" PARENT_SCOPE)
    set(_PDHCG_CPU_BLAS_STATUS "${_PDHCG_CPU_BLAS_STATUS}" PARENT_SCOPE)
endfunction()
pdhcg_detect_blas_threads()
