/* SPDX-License-Identifier: Apache-2.0 */
#include "threads.h"
#include "device_general_op.h"
#include <omp.h>

/* Detect these stable C entry points against the libraries actually linked by
 * this backend. No vendor headers, dynamic lookup, or Python-only dependency. */
#if defined(PDHCG_BLAS_MKL)
extern int MKL_Set_Num_Threads_Local(int num_threads);
#elif defined(PDHCG_BLAS_OPENBLAS)
extern int openblas_get_num_threads(void);
extern void openblas_set_num_threads(int num_threads);

static unsigned int active_scopes;
static int saved_threads;

static void set_openblas_threads(int num_threads)
{
    /* Older OpenMP OpenBLAS releases also change the calling task's ICV.
     * A one-thread task scope preserves the entire original ICV, including
     * nested OMP_NUM_THREADS lists, without adding a worker. */
#pragma omp parallel num_threads(1)
    { openblas_set_num_threads(num_threads); }
}
#endif

int pdhcg_cpu_blas_enter(void)
{
#if defined(PDHCG_BLAS_MKL)
    return MKL_Set_Num_Threads_Local(1);
#elif defined(PDHCG_BLAS_OPENBLAS)
    /* Only entry/exit is serialized. Solves and PSD blocks may run together.
     * Hold a scope across a whole solve to avoid reconfiguring the thread pool
     * for every eigensolve. Direct PSD calls acquire their own shorter scope. */
#pragma omp critical(pdhcg_openblas_threads)
    {
        if (active_scopes++ == 0)
        {
            saved_threads = openblas_get_num_threads();
            if (saved_threads != 1)
                set_openblas_threads(1);
        }
    }
#endif
    return 0;
}

void pdhcg_cpu_blas_leave(int previous)
{
#if defined(PDHCG_BLAS_MKL)
    MKL_Set_Num_Threads_Local(previous);
#else
    (void)previous;
#if defined(PDHCG_BLAS_OPENBLAS)
#pragma omp critical(pdhcg_openblas_threads)
    {
        if (--active_scopes == 0 && saved_threads != 1)
            set_openblas_threads(saved_threads);
    }
#endif
#endif
}

void *pdhcg_device_with_threads(int num_threads, void *(*function)(void *), void *argument)
{
    const int previous_blas_threads = pdhcg_cpu_blas_enter();
    void *result = NULL;
    if (num_threads == 0)
        result = function(argument);
    else
    {
        /* A one-thread region scopes the OpenMP ICV without another worker.
         * Solver loops may create one active team; nested teams stay serial.
         * OpenMP restores all caller settings when this region ends. */
#pragma omp parallel num_threads(1) shared(result)
        {
            omp_set_num_threads(num_threads);
            omp_set_dynamic(0);
            omp_set_max_active_levels(omp_get_active_level() + 1);
            result = function(argument);
        }
    }
    pdhcg_cpu_blas_leave(previous_blas_threads);
    return result;
}
