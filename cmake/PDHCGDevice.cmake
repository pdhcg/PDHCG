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

# Compile backend objects once; private archives serve device tests.
# The public libraries combine those objects with a single dispatch entry point.
include(${CMAKE_CURRENT_LIST_DIR}/PDHCGNamespace.cmake)

function(pdhcg_add_device device)
    string(TOLOWER "${device}" suffix)
    set(core pdhcg_backend_${suffix})
    set(objects pdhcg_objects_${suffix})
    set(test_suffix _${suffix})
    if(device STREQUAL PDHCG_DEFAULT_DEVICE)
        set(test_suffix "")
    endif()

    file(GLOB sources CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.c")
    list(REMOVE_ITEM sources
        "${PROJECT_SOURCE_DIR}/src/cli.c"
        "${PROJECT_SOURCE_DIR}/src/device_dispatch.c")
    set(includes "${PROJECT_SOURCE_DIR}/internal" "${PROJECT_SOURCE_DIR}/src"
                 "${PROJECT_SOURCE_DIR}/distributed")
    set(libraries ZLIB::ZLIB)
    set(defines ${PDHCG_OPTIONAL_DEFINES})
    if(NOT WIN32)
        list(APPEND libraries m)
    endif()
    if(device STREQUAL "CUDA")
        file(GLOB device_sources CONFIGURE_DEPENDS
            "${PROJECT_SOURCE_DIR}/src/device/cuda/*.cu"
            "${PROJECT_SOURCE_DIR}/src/device/cuda/kernels/*.cu")
        list(APPEND includes "${PROJECT_SOURCE_DIR}/src/device/cuda/include")
        list(APPEND libraries CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::cusparse)
        list(APPEND defines CUSPARSE_ENABLE_EXPERIMENTAL_API)
    elseif(device STREQUAL "CPU")
        file(GLOB device_sources CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/device/cpu/*.c")
        list(APPEND libraries OpenMP::OpenMP_C ${LAPACK_LIBRARIES})
    endif()
    list(APPEND sources ${device_sources})

    set(distributed FALSE)
    if(device STREQUAL "CUDA" AND PDHCG_COMPILE_DISTRIBUTED)
        set(distributed TRUE)
        file(GLOB dist_sources "${PROJECT_SOURCE_DIR}/distributed/*.c"
                               "${PROJECT_SOURCE_DIR}/distributed/*.cu")
        list(REMOVE_ITEM dist_sources
            "${PROJECT_SOURCE_DIR}/distributed/distributed_ops_stub.c"
            "${PROJECT_SOURCE_DIR}/distributed/distributed_conic_stub.c")
        list(APPEND sources ${dist_sources})
        list(APPEND defines PDHCG_COMPILE_DISTRIBUTED)
        list(APPEND libraries MPI::MPI_C MPI::MPI_CXX ${PDHCG_NCCL_LIBRARY})
    else()
        list(APPEND sources "${PROJECT_SOURCE_DIR}/distributed/distributed_ops_stub.c"
                            "${PROJECT_SOURCE_DIR}/distributed/distributed_conic_stub.c")
    endif()
    if(PDHCG_COMPILE_PREFOS)
        list(APPEND libraries PreFOS::PreFOS)
    endif()

    add_library(${objects} OBJECT ${sources})
    target_include_directories(${objects} PUBLIC "${PROJECT_SOURCE_DIR}/include" PRIVATE ${includes})
    target_link_libraries(${objects} PUBLIC ${libraries})
    target_compile_definitions(${objects} PUBLIC ${defines})
    if(device STREQUAL "CPU")
        target_compile_definitions(${objects} PRIVATE ${PDHCG_CPU_BLAS_DEFINE})
    endif()
    set_target_properties(${objects} PROPERTIES
        POSITION_INDEPENDENT_CODE ON C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden)
    pdhcg_namespace_target(${objects} "${device}")
    if(device STREQUAL "CUDA")
        set_target_properties(${objects} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        if(NOT WIN32)
            target_compile_options(${objects} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility=hidden>)
        endif()
    endif()

    add_library(${core} STATIC EXCLUDE_FROM_ALL)
    target_link_libraries(${core} PUBLIC ${objects})
    if(device STREQUAL "CUDA")
        set_target_properties(${core} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif()
    set_property(GLOBAL APPEND PROPERTY PDHCG_BACKEND_OBJECTS ${objects})
    set_property(GLOBAL APPEND PROPERTY PDHCG_BACKEND_LIBRARIES ${libraries})

    if(PDHCG_BUILD_TESTS)
        if(device STREQUAL "CPU")
            set(thread_test test_cpu_threads${test_suffix})
            add_executable(${thread_test} "${PROJECT_SOURCE_DIR}/tests/native/cpu_threads.c")
            target_include_directories(${thread_test} PRIVATE ${includes})
            target_link_libraries(${thread_test} PRIVATE ${core})
            target_compile_definitions(${thread_test} PRIVATE ${PDHCG_CPU_BLAS_DEFINE})
            add_test(NAME ${thread_test} COMMAND ${thread_test})
            set_tests_properties(${thread_test} PROPERTIES LABELS "cpu" PROCESSORS 4)
            # Exercise legacy OpenBLAS OpenMP behavior even when CI links a
            # pthread-based library. The test supplies its own BLAS functions.
            set(blas_scope_test test_blas_thread_scope${test_suffix})
            add_executable(${blas_scope_test}
                "${PROJECT_SOURCE_DIR}/tests/native/blas_thread_scope.c"
                "${PROJECT_SOURCE_DIR}/src/device/cpu/threads.c"
                "${PROJECT_SOURCE_DIR}/src/device/cpu/psd.c")
            target_include_directories(${blas_scope_test} PRIVATE
                "${PROJECT_SOURCE_DIR}/include" ${includes})
            target_compile_definitions(${blas_scope_test} PRIVATE PDHCG_BLAS_OPENBLAS)
            target_link_libraries(${blas_scope_test} PRIVATE OpenMP::OpenMP_C)
            if(NOT WIN32)
                target_link_libraries(${blas_scope_test} PRIVATE m)
            endif()
            add_test(NAME ${blas_scope_test} COMMAND ${blas_scope_test})
            set_tests_properties(${blas_scope_test} PROPERTIES LABELS "cpu" PROCESSORS 4
                ENVIRONMENT "OMP_NUM_THREADS=2,3,4;OMP_DYNAMIC=FALSE;OMP_MAX_ACTIVE_LEVELS=3")
        endif()
        set(contract test_device_contract${test_suffix})
        add_executable(${contract} "${PROJECT_SOURCE_DIR}/tests/native/device_contract.c")
        target_include_directories(${contract} PRIVATE ${includes})
        target_link_libraries(${contract} PRIVATE ${core})
        add_test(NAME ${contract} COMMAND ${contract})
        set_tests_properties(${contract} PROPERTIES LABELS "${suffix}")
        foreach(certificate IN ITEMS infeasibility conic_infeasibility)
            set(certificate_test test_${certificate}${test_suffix})
            add_executable(${certificate_test} "${PROJECT_SOURCE_DIR}/tests/native/${certificate}.c")
            target_include_directories(${certificate_test} PRIVATE ${includes})
            target_link_libraries(${certificate_test} PRIVATE ${core})
            add_test(NAME ${certificate_test} COMMAND ${certificate_test})
            set_tests_properties(${certificate_test} PROPERTIES LABELS "${suffix}")
        endforeach()
        file(GLOB test_sources "${PROJECT_SOURCE_DIR}/test/*.c")
        if(device STREQUAL "CUDA")
            file(GLOB cuda_tests "${PROJECT_SOURCE_DIR}/test/*.cu")
            list(APPEND test_sources ${cuda_tests})
        endif()
        set(test_tools export_qcqp_socp inspect_qcqp_exportability qcqp_probe solve_exported_qcqp_pdhcg)
        foreach(source IN LISTS test_sources)
            get_filename_component(name "${source}" NAME_WE)
            set(target ${name}${test_suffix})
            add_executable(${target} "${source}")
            target_link_libraries(${target} PRIVATE ${core})
            target_include_directories(${target} PRIVATE ${includes})
            set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests")
            if(name MATCHES "^test_distributed_(conic|psd)$" AND distributed)
                add_test(NAME ${target}
                    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                            $<TARGET_FILE:${target}> ${MPIEXEC_POSTFLAGS})
                set_tests_properties(${target} PROPERTIES SKIP_RETURN_CODE 77 PROCESSORS 2 TIMEOUT 90 LABELS "${suffix}")
            elseif(NOT name IN_LIST test_tools)
                add_test(NAME ${target} COMMAND ${target})
                set_tests_properties(${target} PROPERTIES LABELS "${suffix}")
            endif()
        endforeach()
    endif()
endfunction()

# Public targets must not inherit a private backend's namespace macros.
function(pdhcg_add_interface)
    get_property(objects GLOBAL PROPERTY PDHCG_BACKEND_OBJECTS)
    get_property(libraries GLOBAL PROPERTY PDHCG_BACKEND_LIBRARIES)
    set(sources "${PROJECT_SOURCE_DIR}/src/device_dispatch.c")
    foreach(object IN LISTS objects)
        list(APPEND sources $<TARGET_OBJECTS:${object}>)
    endforeach()
    list(REMOVE_DUPLICATES libraries)
    string(TOLOWER "${PDHCG_DEFAULT_DEVICE}" default_device)
    set(targets)
    if(PDHCG_BUILD_STATIC_LIB)
        add_library(pdhcg_core STATIC ${sources})
        add_library(pdhcg::pdhcg ALIAS pdhcg_core)
        list(APPEND targets pdhcg_core)
    endif()
    if(PDHCG_BUILD_SHARED_LIB)
        add_library(PDHCG_shared SHARED ${sources})
        if(NOT PDHCG_BUILD_STATIC_LIB)
            add_library(pdhcg::pdhcg ALIAS PDHCG_shared)
        endif()
        list(APPEND targets PDHCG_shared)
    endif()
    foreach(target IN LISTS targets)
        target_include_directories(${target} PUBLIC "${PROJECT_SOURCE_DIR}/include"
            PRIVATE "${PROJECT_SOURCE_DIR}/internal")
        target_link_libraries(${target} PUBLIC ${libraries})
        target_compile_definitions(${target} PRIVATE PDHCG_DEFAULT_DEVICE_NAME="${default_device}")
        target_compile_definitions(${target} PUBLIC ${PDHCG_OPTIONAL_DEFINES})
        foreach(device IN LISTS PDHCG_DEVICES)
            target_compile_definitions(${target} PRIVATE PDHCG_HAS_${device}=1)
        endforeach()
        if(PDHCG_COMPILE_DISTRIBUTED)
            target_compile_definitions(${target} PUBLIC PDHCG_COMPILE_DISTRIBUTED)
        endif()
        set_target_properties(${target} PROPERTIES OUTPUT_NAME pdhcg POSITION_INDEPENDENT_CODE ON)
        if("CUDA" IN_LIST PDHCG_DEVICES)
            set_target_properties(${target} PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        endif()
        if(NOT PDHCG_BUILD_PYTHON)
            install(TARGETS ${target} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
        endif()
    endforeach()

    if(PDHCG_BUILD_CLI)
        add_executable(PDHCG_cli "${PROJECT_SOURCE_DIR}/src/cli.c")
        target_include_directories(PDHCG_cli PRIVATE
            "${PROJECT_SOURCE_DIR}/internal" "${PROJECT_SOURCE_DIR}/distributed")
        target_link_libraries(PDHCG_cli PRIVATE pdhcg::pdhcg)
        set_target_properties(PDHCG_cli PROPERTIES OUTPUT_NAME pdhcg)
        if(NOT PDHCG_BUILD_PYTHON)
            install(TARGETS PDHCG_cli RUNTIME DESTINATION bin)
        endif()
    endif()

    if(PDHCG_BUILD_TESTS)
        foreach(target IN LISTS targets)
            if(target STREQUAL "pdhcg_core")
                set(linkage static)
            else()
                set(linkage shared)
            endif()
            set(test test_device_dispatch_${linkage})
            add_executable(${test} "${PROJECT_SOURCE_DIR}/tests/native/device_dispatch.c")
            target_link_libraries(${test} PRIVATE ${target})
            add_test(NAME ${test} COMMAND ${test})
            set_tests_properties(${test} PROPERTIES LABELS "dispatch" TIMEOUT 60)
        endforeach()
    endif()
endfunction()
