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

include_guard(GLOBAL)

# Backends compile the same C solver with different private symbol names. The
# inventory comes from project declarations, keeping additions to the device
# contract automatic and leaving algorithm and kernel sources unchanged.
function(pdhcg_collect_backend_symbols output)
    file(GLOB headers CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/include/*.h"
        "${PROJECT_SOURCE_DIR}/internal/*.h"
        "${PROJECT_SOURCE_DIR}/distributed/*.h"
        "${PROJECT_SOURCE_DIR}/src/device/cpu/threads.h"
        "${PROJECT_SOURCE_DIR}/src/device/cuda/include/pdhcg_*.h")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${headers})

    # These existing source-local definitions have external linkage but no
    # header declarations. The frontend also forwards the cancellation flag.
    set(symbols
        g_pdhcg_cancel_request
        diag_q_primal_update
        initialize_quadratic_term_information
        lp_primal_update
        optimality_criteria_met
        primal_BB_step_size_update)
    foreach(header IN LISTS headers)
        file(READ "${header}" declarations)
        string(REGEX REPLACE "/\\*([^*]|\\*+[^*/])*\\*+/" " " declarations "${declarations}")
        string(REGEX REPLACE "//[^\n]*" " " declarations "${declarations}")
        string(REGEX REPLACE "\\\\\r?\n" " " declarations "${declarations}")
        string(REGEX REPLACE "(^|\n)[ \t]*#[^\n]*" "\n" declarations "${declarations}")
        string(REGEX REPLACE "extern[ \t\r\n]+\"C\"[ \t\r\n]*\\{" " " declarations "${declarations}")

        # Remove definitions, including structs with function-pointer fields and
        # static inline bodies. Their contents are not external declarations.
        while(declarations MATCHES "\\{[^{}]*\\}")
            string(REGEX REPLACE "\\{[^{}]*\\}" ";" declarations "${declarations}")
        endwhile()
        foreach(declaration IN LISTS declarations)
            string(STRIP "${declaration}" declaration)
            if(declaration MATCHES "(^|[ \t\r\n])(static|inline|typedef)[ \t\r\n]")
                continue()
            endif()
            if(declaration MATCHES "^[A-Za-z_][A-Za-z_0-9 \t\r\n*]*[ \t\r\n*][A-Za-z_][A-Za-z_0-9]*[ \t\r\n]*\\([^;{}]*\\)[ \t\r\n]*$")
                string(REGEX MATCH "([A-Za-z_][A-Za-z_0-9]*)[ \t\r\n]*\\(" function_name "${declaration}")
                list(APPEND symbols "${CMAKE_MATCH_1}")
            elseif(declaration MATCHES "^extern[ \t\r\n]+[A-Za-z_][A-Za-z_0-9 \t\r\n*]*[ \t\r\n*]([A-Za-z_][A-Za-z_0-9]*)[ \t\r\n]*$")
                list(APPEND symbols "${CMAKE_MATCH_1}")
            endif()
        endforeach()
    endforeach()
    # utils.h also redeclares these file-local constants with extern linkage.
    list(REMOVE_ITEM symbols HOST_ONE HOST_ZERO)
    list(REMOVE_DUPLICATES symbols)
    list(SORT symbols)
    set(${output} "${symbols}" PARENT_SCOPE)
endfunction()

function(pdhcg_namespace_target target device)
    string(TOLOWER "${device}" suffix)
    pdhcg_collect_backend_symbols(symbols)
    set(header "${PROJECT_BINARY_DIR}/generated/pdhcg_${suffix}_namespace.h")
    set(contents "/* Generated private backend symbols. Do not edit. */\n#pragma once\n")
    foreach(symbol IN LISTS symbols)
        string(APPEND contents "#define ${symbol} pdhcg_${suffix}_${symbol}\n")
    endforeach()
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/generated")
    file(CONFIGURE OUTPUT "${header}" CONTENT "${contents}" @ONLY)

    # Direct consumers of a private backend (native device tests)
    # use its declarations too. Public dispatcher targets must not inherit this.
    if(MSVC)
        target_compile_options(${target} PUBLIC "$<$<COMPILE_LANGUAGE:C,CXX>:/FI${header}>")
    else()
        target_compile_options(${target} PUBLIC "$<$<COMPILE_LANGUAGE:C,CXX>:SHELL:-include \"${header}\">")
    endif()
    target_compile_options(${target} PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--pre-include \"${header}\">")
endfunction()
