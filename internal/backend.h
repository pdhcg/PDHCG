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

#include "pdhcg.h"
#include "presolve_wrapper.h"

/* Dispatch once per solve. Backend solver and kernel calls remain direct.
   Problems, results, and parser/presolve objects contain host-owned storage. */
typedef struct
{
    qp_problem_t *(*create)(const double *, const matrix_desc_t *, const matrix_desc_t *,
                            const matrix_desc_t *, const matrix_desc_t *, const double *,
                            const double *, const double *, const double *, const double *,
                            int, const cone_spec_t *, const matrix_desc_t *, const double *,
                            int, const cone_spec_t *);
    void (*set_start)(qp_problem_t *, const double *, const double *);
    int (*set_fixed)(qp_problem_t *, int, int, double);
    qp_problem_t *(*to_socp)(const qp_problem_t *, cone_type_t);
    pdhcg_result_t *(*solve)(const qp_problem_t *, const pdhg_parameters_t *);
    pdhcg_result_t *(*solve_distributed)(const pdhg_parameters_t *, const qp_problem_t *);
    void (*defaults)(pdhg_parameters_t *);
    int (*validate)(const pdhg_parameters_t *, char *, size_t);
    void (*free_result)(pdhcg_result_t *);
    void (*free_problem)(qp_problem_t *);
    qp_problem_t *(*read_mps)(const char *);
    qp_problem_t *(*read_cbf)(const char *);
    const char *(*presolve_version)(void);
    int (*presolve_available)(void);
    const char *(*presolve_status)(int);
    pdhcg_presolve_info_t *(*presolve)(const qp_problem_t *, const pdhg_parameters_t *);
    pdhcg_result_t *(*presolve_result)(const pdhcg_presolve_info_t *, const qp_problem_t *);
    int (*postsolve)(const pdhcg_presolve_info_t *, pdhcg_result_t *, const qp_problem_t *);
    void (*free_presolve)(pdhcg_presolve_info_t *);
    void *(*allocate_host)(size_t);
    const char *(*termination_name)(termination_reason_t);
    int (*cone_size)(cone_type_t, int);
    int (*cone_block_size)(const cone_blocks_t *, int);
    int (*initialize)(void);
} pdhcg_backend_t;

#ifdef __cplusplus
extern "C"
{
#endif
    /* CMake prefixes this accessor and all backend symbols for each device. */
    const pdhcg_backend_t *pdhcg_backend(void);
    /* Used by the Python signal handler; only writes atomic signal flags. */
    void pdhcg_set_cancel_request(int requested);
#ifdef __cplusplus
}
#endif
