/* Copyright 2026 Hongpei Li. Licensed under the Apache License, Version 2.0. */
#include "device_general_op.h"
#include "pdhcg.h"
#include "pdhcg_psd_cone.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "thread check failed at line %d: %s\n", __LINE__, #condition); abort(); \
} } while (0)

#if defined(PDHCG_BLAS_OPENBLAS)
extern int openblas_get_num_threads(void);
extern void openblas_set_num_threads(int);
#elif defined(PDHCG_BLAS_MKL)
extern int MKL_Get_Max_Threads(void);
extern void MKL_Set_Num_Threads(int);
extern int MKL_Set_Num_Threads_Local(int);
#endif

static int blas_threads(void)
{
#if defined(PDHCG_BLAS_OPENBLAS)
    return openblas_get_num_threads();
#elif defined(PDHCG_BLAS_MKL)
    return MKL_Get_Max_Threads();
#else
    return 0;
#endif
}

static void check_blas(int expected)
{
#if defined(PDHCG_BLAS_OPENBLAS) || defined(PDHCG_BLAS_MKL)
    CHECK(blas_threads() == expected);
#else
    (void)expected;
#endif
}

static int caller_blas_threads;

typedef struct
{
    int requested;
    int return_null;
    int actual;
} probe_t;

static void *probe(void *argument)
{
    probe_t *p = argument;
    CHECK(omp_get_max_threads() == p->requested);
    CHECK(omp_get_dynamic() == 0);
    check_blas(1);
#pragma omp parallel
    {
#pragma omp single
        {
            p->actual = omp_get_num_threads();
            CHECK(p->actual == p->requested);
            /* A nested library using the same OpenMP runtime cannot multiply
               the solver's thread budget. */
#pragma omp parallel
            {
                CHECK(omp_get_num_threads() == 1);
            }
        }
    }
    check_blas(1);
    return p->return_null ? NULL : argument;
}

static void *unchanged(void *argument)
{
    CHECK(omp_get_max_threads() == 3);
    CHECK(omp_get_dynamic() == 1);
    CHECK(omp_get_max_active_levels() == 3);
    /* Zero inherits OpenMP settings, but must still restrict BLAS. */
    check_blas(1);
    return argument;
}

static void check_caller(void)
{
    CHECK(omp_get_max_threads() == 3);
    CHECK(omp_get_dynamic() == 1);
    CHECK(omp_get_max_active_levels() == 3);
    check_blas(caller_blas_threads);
}

static void *nested(void *argument)
{
    CHECK(omp_get_max_threads() == 2);
    check_blas(1);
    probe_t p = {4, 0, 0};
    CHECK(pdhcg_device_with_threads(4, probe, &p) == &p);
    CHECK(omp_get_max_threads() == 2);
    check_blas(1);
    p.requested = 2;
    p.return_null = 1;
    CHECK(pdhcg_device_with_threads(0, probe, &p) == NULL);
    CHECK(omp_get_max_threads() == 2);
    check_blas(1);
    return argument;
}

static const int budgets[] = {0, 1, 2, 4};

static void solve_and_restore(void)
{
    const int rows[] = {0, 1}, columns[] = {0};
    const double values[] = {1.0}, c[] = {-1.0}, lb[] = {0.0}, ub[] = {2.0};
    matrix_desc_t Q = {0};
    Q.m = Q.n = 1;
    Q.fmt = matrix_csr;
    Q.data.csr.nnz = 1;
    Q.data.csr.row_ptr = rows;
    Q.data.csr.col_ind = columns;
    Q.data.csr.vals = values;
    qp_problem_t *problem = create_qp_problem(c, &Q, NULL, NULL, NULL, NULL, NULL,
                                             lb, ub, NULL, 0, NULL, NULL, NULL, 0, NULL);
    CHECK(problem != NULL);
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = 0;
    for (size_t i = 0; i < sizeof(budgets) / sizeof(budgets[0]); ++i)
    {
        params.num_threads = budgets[i];
        pdhcg_result_t *result = solve_qp_problem(problem, &params);
        CHECK(result != NULL);
        CHECK(result->termination_reason == TERMINATION_REASON_OPTIMAL);
        CHECK(fabs(result->primal_solution[0] - 1.0) < 1e-3);
        pdhcg_result_free(result);
        check_caller();
    }
    params.num_threads = -1;
    CHECK(solve_qp_problem(problem, &params) == NULL);
    check_caller();
    qp_problem_free(problem);
}

/* Four blocks enter the parallel PSD path. Each target matrix is a positive
   multiple of [[-1,2],[2,-1]], whose PSD projection is [[.5,.5],[.5,.5]]. */
static void *psd_projection(void *argument)
{
    const int starts[] = {0, 3, 6, 9}, orders[] = {2, 2, 2, 2};
    const int previous = blas_threads();
    double vector[12];
    for (int k = 0; k < 4; ++k)
    {
        vector[3 * k] = vector[3 * k + 2] = -(k + 1.0);
        vector[3 * k + 1] = 2 * sqrt(2.0) * (k + 1);
    }
    psd_projection_runtime_t *runtime = create_psd_projection_runtime(starts, orders, 4, 0);
    CHECK(runtime != NULL);
    check_blas(previous);
    project_psd_cones(runtime, vector);
    check_blas(previous);
    for (int k = 0; k < 4; ++k)
    {
        CHECK(fabs(vector[3 * k] - 0.5 * (k + 1)) < 1e-12);
        CHECK(fabs(vector[3 * k + 1] - sqrt(0.5) * (k + 1)) < 1e-12);
        CHECK(fabs(vector[3 * k + 2] - 0.5 * (k + 1)) < 1e-12);
    }
    free_psd_projection_runtime(runtime);
    return argument;
}

static void psd_solve_and_restore(void)
{
    int rows[13], columns[12];
    double values[12], c[12];
    cone_spec_t cones[4] = {0};
    for (int i = 0; i < 12; ++i)
    {
        rows[i] = columns[i] = i;
        values[i] = 1;
        c[i] = (i % 3 == 1 ? -2 * sqrt(2.0) : 1) * (i / 3 + 1);
    }
    rows[12] = 12;
    for (int k = 0; k < 4; ++k)
    {
        cones[k].type = CONE_PSD;
        cones[k].start_idx = 3 * k;
        cones[k].v_dim = 2;
    }
    matrix_desc_t Q = {0};
    Q.m = Q.n = 12;
    Q.fmt = matrix_csr;
    Q.data.csr.nnz = 12;
    Q.data.csr.row_ptr = rows;
    Q.data.csr.col_ind = columns;
    Q.data.csr.vals = values;
    qp_problem_t *problem = create_qp_problem(c, &Q, NULL, NULL, NULL, NULL, NULL,
                                             NULL, NULL, NULL, 4, cones, NULL, NULL, 0, NULL);
    CHECK(problem != NULL);
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = 0;
    params.termination_criteria.eps_optimal_relative = 1e-6;
    params.termination_criteria.eps_feasible_relative = 1e-6;
    for (size_t k = 0; k < sizeof(budgets) / sizeof(budgets[0]); ++k)
    {
        params.num_threads = budgets[k];
        pdhcg_result_t *result = solve_qp_problem(problem, &params);
        CHECK(result != NULL);
        CHECK(result->termination_reason == TERMINATION_REASON_OPTIMAL);
        for (int i = 0; i < 12; ++i)
        {
            double expected = (i % 3 == 1 ? sqrt(0.5) : 0.5) * (i / 3 + 1);
            CHECK(fabs(result->primal_solution[i] - expected) < 1e-3);
        }
        pdhcg_result_free(result);
        check_caller();
    }
    qp_problem_free(problem);
}

static void wait_for(int *flag)
{
    double deadline = omp_get_wtime() + 10;
    int ready = 0;
    while (!ready)
    {
#pragma omp atomic read
        ready = *flag;
        CHECK(omp_get_wtime() < deadline);
    }
}

typedef struct
{
    int id;
    int *second_entered;
    int *first_finished;
} overlap_t;

static void *overlap(void *argument)
{
    overlap_t *p = argument;
    check_blas(1);
    if (p->id == 0)
        wait_for(p->second_entered);
    else
    {
#pragma omp atomic write
        *p->second_entered = 1;
        wait_for(p->first_finished);
        /* The first solve has returned, but this solve is still active. */
        check_blas(1);
    }
    probe_t probe_data = {p->id + 1, 0, 0};
    CHECK(probe(&probe_data) == &probe_data);
    return argument;
}

static void concurrent_scopes(void)
{
    int second_entered = 0, first_finished = 0;
    omp_set_dynamic(0);
#pragma omp parallel num_threads(2)
    {
        CHECK(omp_get_num_threads() == 2);
        const int previous = omp_get_max_threads();
#if defined(PDHCG_BLAS_MKL)
        /* Existing thread-local settings must survive each solve separately. */
        const int previous_local = MKL_Set_Num_Threads_Local(3);
#endif
        overlap_t p = {omp_get_thread_num(), &second_entered, &first_finished};
        CHECK(pdhcg_device_with_threads(p.id + 1, overlap, &p) == &p);
        CHECK(omp_get_max_threads() == previous);
        CHECK(omp_get_dynamic() == 0);
        CHECK(omp_get_max_active_levels() == 3);
#if defined(PDHCG_BLAS_MKL)
        check_blas(3);
        MKL_Set_Num_Threads_Local(previous_local);
#elif defined(PDHCG_BLAS_OPENBLAS)
        if (p.id == 0)
            check_blas(1);
#endif
        if (p.id == 0)
        {
#pragma omp atomic write
            first_finished = 1;
        }
    }
    omp_set_dynamic(1);
    check_caller();
}

int main(void)
{
    int original_blas_threads = blas_threads();
#if defined(PDHCG_BLAS_OPENBLAS)
    openblas_set_num_threads(3);
#elif defined(PDHCG_BLAS_MKL)
    MKL_Set_Num_Threads(3);
    const int original_local = MKL_Set_Num_Threads_Local(2);
#endif
    (void)original_blas_threads;
    caller_blas_threads = blas_threads();
    omp_set_num_threads(3);
    omp_set_dynamic(1);
    omp_set_max_active_levels(3);
    for (int threads = 1; threads <= 4; threads *= 2)
    {
        probe_t p = {threads, 0, 0};
        CHECK(pdhcg_device_with_threads(threads, probe, &p) == &p);
        check_caller();
        p.return_null = 1;
        CHECK(pdhcg_device_with_threads(threads, probe, &p) == NULL);
        check_caller();
        printf("requested=%d observed=%d, caller settings restored\n", threads, p.actual);
    }
    int marker = 0;
    CHECK(pdhcg_device_with_threads(0, unchanged, &marker) == &marker);
    check_caller();
    CHECK(pdhcg_device_with_threads(2, nested, &marker) == &marker);
    check_caller();
    solve_and_restore();
    concurrent_scopes();

    /* Standalone PSD calls also protect LAPACK, outside any solve scope. */
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    CHECK(psd_projection(&marker) == &marker);
    omp_set_num_threads(3);
    omp_set_dynamic(1);
    check_caller();
    for (size_t i = 0; i < sizeof(budgets) / sizeof(budgets[0]); ++i)
    {
        CHECK(pdhcg_device_with_threads(budgets[i], psd_projection, &marker) == &marker);
        check_caller();
    }
    psd_solve_and_restore();
#if defined(PDHCG_BLAS_MKL)
    /* Restore the inheritance sentinel, not an equal effective thread count. */
    const int inherited_local = MKL_Set_Num_Threads_Local(0);
    probe_t inherited = {1, 0, 0};
    CHECK(pdhcg_device_with_threads(1, probe, &inherited) == &inherited);
    CHECK(MKL_Set_Num_Threads_Local(0) == 0);
    MKL_Set_Num_Threads_Local(inherited_local);
    check_caller();
#endif
#if defined(PDHCG_BLAS_OPENBLAS)
    openblas_set_num_threads(original_blas_threads);
#elif defined(PDHCG_BLAS_MKL)
    MKL_Set_Num_Threads_Local(original_local);
    MKL_Set_Num_Threads(original_blas_threads);
#endif
    puts("per-solve CPU and BLAS thread controls passed");
    return 0;
}
