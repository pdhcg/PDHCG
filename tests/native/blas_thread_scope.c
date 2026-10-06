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

/* Link the real CPU threads.c and psd.c with this small legacy-OpenBLAS
 * substitute, without a BLAS/LAPACK library. Run with OMP_NUM_THREADS=2,3,4,
 * OMP_DYNAMIC=FALSE, and OMP_MAX_ACTIVE_LEVELS=3. The substitute models the
 * OpenBLAS 0.3.21 OpenMP behavior: its setter changes the caller's ICV, and
 * BLAS calls may reset the library-wide count from that ICV. */
#include "device_general_op.h"
#include "pdhcg_psd_cone.h"
#include "utils.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "BLAS scope check failed at line %d: %s\n", __LINE__, #condition); \
    abort(); \
} } while (0)

static int blas_threads = 4;
static int setter_calls;
static int workspace_queries;
static int eigensolves;

int openblas_get_num_threads(void)
{
    int result;
#pragma omp atomic read
    result = blas_threads;
    return result;
}

void openblas_set_num_threads(int count)
{
    CHECK(count > 0);
    /* A direct call here must not overwrite PDHCG's or its caller's ICV. */
    omp_set_num_threads(count);
#pragma omp atomic write
    blas_threads = count;
#pragma omp atomic update
    ++setter_calls;
}

void dsyev_(const char *job, const char *triangle, const int *n, double *matrix,
            const int *lda, double *eigenvalues, double *work,
            const int *lwork, int *info)
{
    const int requested = omp_get_max_threads();
    if (requested != 1 && !omp_in_parallel() && requested != openblas_get_num_threads())
        openblas_set_num_threads(requested);

    /* In particular, num_threads(1) alone does not set the next region's
     * thread count when OMP_NUM_THREADS contains multiple entries. Removing
     * omp_set_num_threads(1) from the real LAPACK wrapper must fail here. */
    CHECK(requested == 1);
    CHECK(openblas_get_num_threads() == 1);
    CHECK(*job == 'V' && *triangle == 'L');
    CHECK(*n == 2 && *lda == 2);
    *info = 0;
    if (*lwork == -1)
    {
        *work = 16;
#pragma omp atomic update
        ++workspace_queries;
        return;
    }

    /* Only diagonal 2-by-2 matrices are used. Eigenvalues and an identity
     * eigenbasis let the real PSD code complete its projection normally. */
    CHECK(*lwork >= 16);
    CHECK(matrix[1] == 0 && matrix[2] == 0);
    CHECK(matrix[0] <= matrix[3]);
    eigenvalues[0] = matrix[0];
    eigenvalues[1] = matrix[3];
    matrix[0] = matrix[3] = 1;
    matrix[1] = matrix[2] = 0;
#pragma omp atomic update
    ++eigensolves;
}

void *safe_malloc(size_t size)
{
    void *result = malloc(size);
    CHECK(result != NULL);
    return result;
}

void *safe_calloc(size_t count, size_t size)
{
    void *result = calloc(count, size);
    CHECK(result != NULL);
    return result;
}

static void check_caller_icv(void)
{
    CHECK(omp_get_max_threads() == 2);
    CHECK(omp_get_dynamic() == 0);
    CHECK(omp_get_max_active_levels() == 3);
    /* Serialized regions expose successive entries of the inherited list
     * without creating additional workers or changing the original list. */
#pragma omp parallel num_threads(1)
    {
        CHECK(omp_get_max_threads() == 3);
#pragma omp parallel num_threads(1)
        { CHECK(omp_get_max_threads() == 4); }
    }
}

static void check_projection(const double *vector, int count)
{
    for (int i = 0; i < count; ++i)
    {
        CHECK(fabs(vector[3 * i]) < 1e-12);
        CHECK(fabs(vector[3 * i + 1]) < 1e-12);
        CHECK(fabs(vector[3 * i + 2] - 5) < 1e-12);
    }
}

static void check_direct_psd_calls(void)
{
    const int start[] = {0}, order[] = {2};
    double vector[] = {-2, 0, 5};
    psd_projection_runtime_t *runtime = create_psd_projection_runtime(start, order, 1, 0);
    CHECK(runtime != NULL);
    CHECK(workspace_queries == 1);
    CHECK(openblas_get_num_threads() == 4);
    check_caller_icv();

    project_psd_cones(runtime, vector);
    CHECK(eigensolves == 1);
    CHECK(openblas_get_num_threads() == 4);
    check_caller_icv();
    check_projection(vector, 1);
    free_psd_projection_runtime(runtime);
}

static void *project_in_solver_scope(void *argument)
{
    const int expected_threads = *(const int *)argument;
    const int start[] = {0, 3, 6, 9}, order[] = {2, 2, 2, 2};
    double vector[] = {-2, 0, 5, -2, 0, 5, -2, 0, 5, -2, 0, 5};
    CHECK(omp_get_max_threads() == expected_threads);
    CHECK(openblas_get_num_threads() == 1);
    psd_projection_runtime_t *runtime = create_psd_projection_runtime(start, order, 4, 0);
    CHECK(runtime != NULL);
    CHECK(omp_get_max_threads() == expected_threads);
    project_psd_cones(runtime, vector);
    CHECK(omp_get_max_threads() == expected_threads);
    CHECK(openblas_get_num_threads() == 1);
    check_projection(vector, 4);
    free_psd_projection_runtime(runtime);
    return argument;
}

int main(void)
{
    check_caller_icv();
    check_direct_psd_calls();
    for (int selected = 0; selected <= 3; selected += 3)
    {
        int expected = selected ? selected : 2;
        const int before = setter_calls;
        CHECK(pdhcg_device_with_threads(selected, project_in_solver_scope, &expected) == &expected);
        CHECK(openblas_get_num_threads() == 4);
        CHECK(setter_calls == before + 2);
        check_caller_icv();
    }
    CHECK(workspace_queries == 9);
    CHECK(eigensolves == 9);
    puts("legacy OpenBLAS OpenMP scopes and nested caller ICVs passed");
    return 0;
}
