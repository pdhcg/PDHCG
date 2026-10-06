/*
Copyright 2025 Haihao Lu
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

/* Internal device contract. No vendor headers are allowed in this file.
 * CMake links one implementation into each backend library (PDHCG_DEVICES).
 * Multiple backend libraries may coexist; Python selects one per solve.
 * The solver owns the algorithm, while a device owns memory and numerical
 * execution. New devices implement this API, device_kernels.h, the cone launch
 * operations in cone_kernel_ops.h/device_cones.h, spmv_backend.h and the opaque
 * PSD runtime. Each backend owns its kernel and projection implementations.
 * Buffers belong to the selected device and must not be dereferenced by the
 * shared solver. Copy directions are explicit even when CPU storage is shared.
 */
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C"
{
#endif
    /* Handles own their vendor resources; vector descriptors borrow array storage. */
    typedef struct pdhcg_device_blas *pdhcg_device_blas_t;
    typedef struct pdhcg_device_sparse *pdhcg_device_sparse_t;
    typedef struct pdhcg_device_vector *pdhcg_device_vector_t;
    typedef enum
    {
        PDHCG_POINTER_HOST,
        PDHCG_POINTER_DEVICE
    } pdhcg_device_pointer_mode_t;
    typedef enum
    {
        PDHCG_TRIANGLE_LOWER,
        PDHCG_TRIANGLE_UPPER
    } pdhcg_device_triangle_t;
    typedef enum
    {
        PDHCG_COPY_HOST_TO_DEVICE,
        PDHCG_COPY_DEVICE_TO_HOST,
        PDHCG_COPY_DEVICE_TO_DEVICE
    } pdhcg_device_copy_kind_t;

    const char *pdhcg_device_name(void);
    int pdhcg_device_initialize(void);
    /* Run synchronously with a per-call CPU OpenMP thread budget. num_threads
       must be nonnegative; 0 leaves OpenMP settings unchanged. Positive values
       include the caller and disable nested teams within this solve. The caller's
       OpenMP settings are preserved on return, including a NULL callback result.
       Supported BLAS libraries use one thread within this scope and CPU PSD
       calls, including with num_threads=0. Their settings are restored afterward.
       This does not cap all process threads.
       CUDA calls the function directly. */
    void *pdhcg_device_with_threads(int num_threads, void *(*function)(void *), void *argument);
    int pdhcg_device_allocate(void **ptr, size_t bytes);
    int pdhcg_device_free(void *ptr);
    /* Blocking copy makes the destination available on return. Async operations use
   the backend's ordered default execution queue; host sources must remain alive
   until synchronization. CPU operations complete synchronously. Zero bytes are valid. */
    int pdhcg_device_copy(void *dst, const void *src, size_t bytes, pdhcg_device_copy_kind_t kind);
    int pdhcg_device_copy_async(void *dst, const void *src, size_t bytes, pdhcg_device_copy_kind_t kind);
    int pdhcg_device_zero(void *dst, int value, size_t bytes);
    int pdhcg_device_zero_async(void *dst, int value, size_t bytes);
    int pdhcg_device_synchronize(void);
    int pdhcg_device_last_error(void);
    int pdhcg_device_blas_create(pdhcg_device_blas_t *handle);
    int pdhcg_device_blas_destroy(pdhcg_device_blas_t handle);
    int pdhcg_device_sparse_create(pdhcg_device_sparse_t *handle);
    int pdhcg_device_sparse_destroy(pdhcg_device_sparse_t handle);
    /* Pointer mode applies to BLAS scalar inputs AND reduction outputs. Host-mode
   reductions complete before returning. Device mode preserves device residency. */
    int pdhcg_device_set_pointer_mode(pdhcg_device_blas_t handle, pdhcg_device_pointer_mode_t mode);
    int pdhcg_device_get_pointer_mode(pdhcg_device_blas_t handle, pdhcg_device_pointer_mode_t *mode);
    int pdhcg_device_dot(
        pdhcg_device_blas_t handle, int n, const double *x, int incx, const double *y, int incy, double *result);
    int pdhcg_device_nrm2(pdhcg_device_blas_t handle, int64_t n, const double *x, int64_t incx, double *result);
    int pdhcg_device_asum(pdhcg_device_blas_t handle, int n, const double *x, int incx, double *result);
    /* Index result follows the existing BLAS convention: one-based; zero for n=0. */
    int pdhcg_device_iamax(pdhcg_device_blas_t handle, int n, const double *x, int incx, int *result);
    int pdhcg_device_scal(pdhcg_device_blas_t handle, int n, const double *alpha, double *x, int incx);
    int pdhcg_device_axpy(
        pdhcg_device_blas_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy);
    int pdhcg_device_symv(pdhcg_device_blas_t handle,
                          pdhcg_device_triangle_t triangle,
                          int n,
                          const double *alpha,
                          const double *a,
                          int lda,
                          const double *x,
                          int incx,
                          const double *beta,
                          double *y,
                          int incy);
    int pdhcg_device_vector_create(pdhcg_device_vector_t *vector, int64_t size, double *values);
    int pdhcg_device_vector_destroy(pdhcg_device_vector_t vector);
    int pdhcg_device_vector_set_values(pdhcg_device_vector_t vector, double *values);
    /* CSR arrays use zero-based int indices and double values. Output storage is
   supplied by the caller: n+1 row offsets and nnz indices/values. */
    int pdhcg_device_csr_transpose(pdhcg_device_sparse_t handle,
                                   int m,
                                   int n,
                                   int nnz,
                                   const double *values,
                                   const int *rows,
                                   const int *columns,
                                   double *out_values,
                                   int *out_rows,
                                   int *out_columns);
#ifdef __cplusplus
}
#endif
