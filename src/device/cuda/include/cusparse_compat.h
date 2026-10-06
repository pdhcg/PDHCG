/*
Copyright 2026 Hongpei Li

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include <cusparse.h>

// cusparseSpMVOp_bufferSize was introduced in cuSPARSE 12.7.3.
// Older CUDA/cuSPARSE versions should use the standard cusparseSpMV path.
#if defined(CUSPARSE_VERSION) && CUSPARSE_VERSION >= 12703
#define PDHCG_USE_SPMVOP 1
#else
#define PDHCG_USE_SPMVOP 0
#endif

// CUDA Toolkit 13.3 / cuSPARSE 12.8.1 added the alg parameter to
// cusparseSpMVOp_bufferSize and cusparseSpMVOp_createDescr. CUDA 13.2 /
// cuSPARSE 12.7.10 still uses the older signature without alg.
#if defined(CUSPARSE_VERSION) && CUSPARSE_VERSION >= 12801
#define PDHCG_CUSPARSE_SPMVOP_HAS_ALG_PARAM 1
#else
#define PDHCG_CUSPARSE_SPMVOP_HAS_ALG_PARAM 0
#endif

#if !PDHCG_USE_SPMVOP
// The SpMVOp types were added to cusparse.h before the functions
#if !defined(CUSPARSE_VERSION) || CUSPARSE_VERSION < 12700
typedef void *cusparseSpMVOpDescr_t;
typedef void *cusparseSpMVOpPlan_t;
#endif
#endif
