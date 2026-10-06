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

#include "checks.h"
#include "device_general_op.h"
#include <stdint.h>
const char *pdhcg_device_name()
{
    return "cuda";
}
int pdhcg_device_initialize()
{
    return cudaFree(nullptr);
}
void *pdhcg_device_with_threads(int num_threads, void *(*function)(void *), void *argument)
{
    (void)num_threads;
    return function(argument);
}
int pdhcg_device_allocate(void **ptr, size_t bytes)
{
    return cudaMalloc(ptr, bytes ? bytes : 1);
}
int pdhcg_device_free(void *ptr)
{
    return cudaFree(ptr);
}
static cudaMemcpyKind copy_kind(pdhcg_device_copy_kind_t k)
{
    return k == PDHCG_COPY_HOST_TO_DEVICE
        ? cudaMemcpyHostToDevice
        : (k == PDHCG_COPY_DEVICE_TO_HOST ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice);
}
int pdhcg_device_copy(void *dst, const void *src, size_t bytes, pdhcg_device_copy_kind_t kind)
{
    return bytes ? cudaMemcpy(dst, src, bytes, copy_kind(kind)) : 0;
}
int pdhcg_device_copy_async(void *dst, const void *src, size_t bytes, pdhcg_device_copy_kind_t kind)
{
    return bytes ? cudaMemcpyAsync(dst, src, bytes, copy_kind(kind)) : 0;
}
int pdhcg_device_zero(void *dst, int value, size_t bytes)
{
    return bytes ? cudaMemset(dst, value, bytes) : 0;
}
int pdhcg_device_zero_async(void *dst, int value, size_t bytes)
{
    return bytes ? cudaMemsetAsync(dst, value, bytes) : 0;
}
int pdhcg_device_synchronize()
{
    return cudaDeviceSynchronize();
}
int pdhcg_device_last_error()
{
    return cudaGetLastError();
}
int pdhcg_device_blas_create(pdhcg_device_blas_t *out)
{
    cublasHandle_t h = nullptr;
    int status = cublasCreate(&h);
    *out = reinterpret_cast<pdhcg_device_blas_t>(h);
    return status;
}
int pdhcg_device_blas_destroy(pdhcg_device_blas_t h)
{
    return cublasDestroy(reinterpret_cast<cublasHandle_t>(h));
}
int pdhcg_device_sparse_create(pdhcg_device_sparse_t *out)
{
    cusparseHandle_t h = nullptr;
    int status = cusparseCreate(&h);
    *out = reinterpret_cast<pdhcg_device_sparse_t>(h);
    return status;
}
int pdhcg_device_sparse_destroy(pdhcg_device_sparse_t h)
{
    return cusparseDestroy(reinterpret_cast<cusparseHandle_t>(h));
}
int pdhcg_device_set_pointer_mode(pdhcg_device_blas_t handle, pdhcg_device_pointer_mode_t mode)
{
    return cublasSetPointerMode(reinterpret_cast<cublasHandle_t>(handle),
                                mode == PDHCG_POINTER_HOST ? CUBLAS_POINTER_MODE_HOST : CUBLAS_POINTER_MODE_DEVICE);
}
int pdhcg_device_get_pointer_mode(pdhcg_device_blas_t handle, pdhcg_device_pointer_mode_t *mode)
{
    cublasPointerMode_t m;
    int status = cublasGetPointerMode(reinterpret_cast<cublasHandle_t>(handle), &m);
    if (!status)
        *mode = m == CUBLAS_POINTER_MODE_HOST ? PDHCG_POINTER_HOST : PDHCG_POINTER_DEVICE;
    return status;
}
int pdhcg_device_dot(
    pdhcg_device_blas_t handle, int n, const double *x, int incx, const double *y, int incy, double *result)
{
    return cublasDdot(reinterpret_cast<cublasHandle_t>(handle), n, x, incx, y, incy, result);
}
int pdhcg_device_nrm2(pdhcg_device_blas_t handle, int64_t n, const double *x, int64_t incx, double *result)
{
    return cublasDnrm2_v2_64(reinterpret_cast<cublasHandle_t>(handle), n, x, incx, result);
}
int pdhcg_device_asum(pdhcg_device_blas_t handle, int n, const double *x, int incx, double *result)
{
    return cublasDasum(reinterpret_cast<cublasHandle_t>(handle), n, x, incx, result);
}
int pdhcg_device_iamax(pdhcg_device_blas_t handle, int n, const double *x, int incx, int *result)
{
    return cublasIdamax(reinterpret_cast<cublasHandle_t>(handle), n, x, incx, result);
}
int pdhcg_device_scal(pdhcg_device_blas_t handle, int n, const double *alpha, double *x, int incx)
{
    return cublasDscal(reinterpret_cast<cublasHandle_t>(handle), n, alpha, x, incx);
}
int pdhcg_device_axpy(
    pdhcg_device_blas_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy)
{
    return cublasDaxpy(reinterpret_cast<cublasHandle_t>(handle), n, alpha, x, incx, y, incy);
}
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
                      int incy)
{
    return cublasDsymv(reinterpret_cast<cublasHandle_t>(handle),
                       (triangle == PDHCG_TRIANGLE_LOWER ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER),
                       n,
                       alpha,
                       a,
                       lda,
                       x,
                       incx,
                       beta,
                       y,
                       incy);
}
int pdhcg_device_vector_create(pdhcg_device_vector_t *out, int64_t size, double *values)
{
    cusparseDnVecDescr_t v = nullptr;
    int status = cusparseCreateDnVec(&v, size, values, CUDA_R_64F);
    *out = reinterpret_cast<pdhcg_device_vector_t>(v);
    return status;
}
int pdhcg_device_vector_destroy(pdhcg_device_vector_t v)
{
    return cusparseDestroyDnVec(reinterpret_cast<cusparseDnVecDescr_t>(v));
}
int pdhcg_device_vector_set_values(pdhcg_device_vector_t v, double *values)
{
    return cusparseDnVecSetValues(reinterpret_cast<cusparseDnVecDescr_t>(v), values);
}
int pdhcg_device_csr_transpose(pdhcg_device_sparse_t handle,
                               int m,
                               int n,
                               int nnz,
                               const double *values,
                               const int *rows,
                               const int *columns,
                               double *out_values,
                               int *out_rows,
                               int *out_columns)
{
    if (nnz == 0)
        return cudaMemset(out_rows, 0, ((size_t)n + 1) * sizeof(int));
    auto h = reinterpret_cast<cusparseHandle_t>(handle);
    size_t bytes = 0;
    int status = cusparseCsr2cscEx2_bufferSize(h,
                                               m,
                                               n,
                                               nnz,
                                               values,
                                               rows,
                                               columns,
                                               out_values,
                                               out_rows,
                                               out_columns,
                                               CUDA_R_64F,
                                               CUSPARSE_ACTION_NUMERIC,
                                               CUSPARSE_INDEX_BASE_ZERO,
                                               CUSPARSE_CSR2CSC_ALG_DEFAULT,
                                               &bytes);
    if (status)
        return status;
    void *buffer = nullptr;
    status = cudaMalloc(&buffer, bytes ? bytes : 1);
    if (status)
        return status;
    status = cusparseCsr2cscEx2(h,
                                m,
                                n,
                                nnz,
                                values,
                                rows,
                                columns,
                                out_values,
                                out_rows,
                                out_columns,
                                CUDA_R_64F,
                                CUSPARSE_ACTION_NUMERIC,
                                CUSPARSE_INDEX_BASE_ZERO,
                                CUSPARSE_CSR2CSC_ALG_DEFAULT,
                                buffer);
    int release_status = cudaFree(buffer);
    return status ? status : release_status;
}
