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

# One extension per device, all built from the same binding source and C solver.
# Imports are lazy, so CPU use does not load any CUDA libraries.
set(PDHCG_PYTHON_DEVICES "")
foreach(device IN LISTS PDHCG_DEVICES)
    string(TOLOWER "${device}" suffix)
    set(module _pdhcg_${suffix})
    pybind11_add_module(${module} MODULE _core_bindings.cpp)
    target_link_libraries(${module} PRIVATE pdhcg::${suffix})
    target_include_directories(${module} PRIVATE "${PROJECT_SOURCE_DIR}/internal")
    target_compile_definitions(${module} PRIVATE
        PDHCG_PYTHON_MODULE=${module} PYBIND11_DETAILED_ERROR_MESSAGES=1)
    # Prevent symbols from static dependencies from interposing on another
    # backend even when Python extensions are imported with RTLD_GLOBAL.
    if(UNIX AND NOT APPLE)
        target_link_options(${module} PRIVATE "LINKER:--exclude-libs,ALL")
    endif()
    set_target_properties(${module} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
    install(TARGETS ${module} LIBRARY DESTINATION pdhcg RUNTIME DESTINATION pdhcg)
    string(APPEND PDHCG_PYTHON_DEVICES "\"${suffix}\", ")
endforeach()
string(TOLOWER "${PDHCG_DEFAULT_DEVICE}" PDHCG_PYTHON_DEFAULT_DEVICE)
configure_file(_build_config.py.in "${CMAKE_CURRENT_BINARY_DIR}/_build_config.py" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_build_config.py" DESTINATION pdhcg)
