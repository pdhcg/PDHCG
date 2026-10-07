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

# Python passes params.device to the same native entry point used by C clients.
pybind11_add_module(_pdhcg_core MODULE _core_bindings.cpp)
target_link_libraries(_pdhcg_core PRIVATE pdhcg::pdhcg)
target_include_directories(_pdhcg_core PRIVATE "${PROJECT_SOURCE_DIR}/internal")
target_compile_definitions(_pdhcg_core PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES=1)
if(UNIX AND NOT APPLE)
    target_link_options(_pdhcg_core PRIVATE "LINKER:--exclude-libs,ALL")
endif()
set_target_properties(_pdhcg_core PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
install(TARGETS _pdhcg_core LIBRARY DESTINATION pdhcg RUNTIME DESTINATION pdhcg)
