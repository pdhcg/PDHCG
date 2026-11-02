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

#include "checks.h"
#include "device_linearized.h"
#include "pdhcg_kernels.h"
#include "utils.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static __global__ void concatenate_primal_matrix_kernel(device_sparse_matrix_csr_t Q,
                                                        device_sparse_matrix_csr_t AT,
                                                        device_sparse_matrix_csr_t R,
                                                        device_sparse_matrix_csr_t result)
{
    int n = AT.num_rows;
    int r_start = result.num_rows - R.num_rows;
    int offset = Q.num_nonzeros + AT.num_nonzeros;
    int64_t thread = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    /* R is appended unchanged; distribute its entries across all threads. */
    for (int64_t entry = thread; entry < R.num_nonzeros; entry += stride)
    {
        result.col_ind[offset + entry] = R.col_ind[entry];
        result.val[offset + entry] = R.val[entry];
    }

    int64_t row = thread / warpSize;
    int lane = thread % warpSize;
    if (row > result.num_rows)
        return;
    if (row >= n)
    {
        if (lane == 0)
            result.row_ptr[row] = offset + (row >= r_start && R.row_ptr ? R.row_ptr[row - r_start] : 0);
        return;
    }

    int q_start = Q.row_ptr ? Q.row_ptr[row] : 0;
    int q_end = Q.row_ptr ? Q.row_ptr[row + 1] : 0;
    int a_start = AT.row_ptr[row];
    int a_end = AT.row_ptr[row + 1];
    if (lane == 0)
        result.row_ptr[row] = q_start + a_start;
    for (int64_t entry = (int64_t)q_start + lane; entry < q_end; entry += warpSize)
    {
        int64_t out = a_start + entry;
        result.col_ind[out] = Q.col_ind[entry];
        result.val[out] = Q.val[entry];
    }
    for (int64_t entry = (int64_t)a_start + lane; entry < a_end; entry += warpSize)
    {
        int64_t out = q_end + entry;
        result.col_ind[out] = n + AT.col_ind[entry];
        result.val[out] = -AT.val[entry];
    }
}

static __global__ void linearized_primal_kernel(int n,
                                                double *x,
                                                const double *initial_x,
                                                const double *gradient,
                                                const double *objective,
                                                const double *variable_lower,
                                                const double *variable_upper,
                                                double *reflected_x,
                                                double *pdhg_x,
                                                const double *primal_step_size,
                                                const int *base_count,
                                                int iteration,
                                                double reflection)
{
    int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n)
        return;

    double current = x[row];
    double next = current - *primal_step_size * (gradient[row] + objective[row]);
    next = fmax(variable_lower[row], fmin(next, variable_upper[row]));
    double reflected = 2.0 * next - current;
    reflected_x[row] = reflected;
    if (pdhg_x)
        pdhg_x[row] = next;
    double current_k = (double)*base_count + iteration;
    double weight = current_k / (current_k + 1.0);
    x[row] = weight * (reflection * reflected + (1.0 - reflection) * current) +
        (1.0 - weight) * initial_x[row];
}

static __global__ void linearized_dual_kernel(int m,
                                              const double *primal_product,
                                              double *y,
                                              double *reflected_y,
                                              double *pdhg_y,
                                              const double *initial_y,
                                              const double *constraint_lower,
                                              const double *constraint_upper,
                                              const double *constraint_offset,
                                              const double *dual_step_size,
                                              const int *base_count,
                                              int iteration,
                                              double reflection)
{
    int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m)
        return;
    double current_k = (double)*base_count + iteration;
    double weight = current_k / (current_k + 1.0);

    double dual_step = *dual_step_size;
    double current = y[row];
    double value = primal_product[row] + constraint_offset[row];
    double projected = fmax(constraint_lower[row],
                            fmin(value - current / dual_step, constraint_upper[row]));
    double next = current - dual_step * value + dual_step * projected;
    double reflected = 2.0 * next - current;
    reflected_y[row] = reflected;
    if (pdhg_y)
        pdhg_y[row] = next;
    y[row] = weight * (reflection * reflected + (1.0 - reflection) * current) +
        (1.0 - weight) * initial_y[row];
}

static void pdhcg_linearized_iterations(pdhg_solver_state_t *state,
                                        const pdhg_parameters_t *params,
                                        int iterations)
{
    linearized_context_t *context = state->inner_solver->linearized_context;
    cudaStream_t stream = (cudaStream_t)context->stream;
    quadratic_objective_term_t *term = state->quadratic_objective_term;
    int n = state->num_variables;
    int m = state->num_constraints;
    int primal_blocks = n / THREADS_PER_BLOCK + (n % THREADS_PER_BLOCK != 0);
    int dual_blocks = m / THREADS_PER_BLOCK + (m % THREADS_PER_BLOCK != 0);

    for (int iteration = 1; iteration <= iterations; ++iteration)
    {
        double *pdhg_x = iteration == iterations ? state->pdhg_primal_solution : NULL;
        double *pdhg_y = iteration == iterations ? state->pdhg_dual_solution : NULL;
        pdhcg_spmv_execute(state->sparse_handle, context->spmv_ctx, &HOST_ONE, &HOST_ZERO,
                            state->current_primal_solution, state->dual_product);
        if (term->objective_lowrank_matrix && term->num_rank_lowrank_obj > 0)
        {
            int rank = term->num_rank_lowrank_obj;
            const double *product = term->Rx_product;
            if (term->lowrank_middle_type == PDHCG_D_DIAG)
            {
                int blocks = rank / THREADS_PER_BLOCK + (rank % THREADS_PER_BLOCK != 0);
                element_wise_mul_inplace_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
                    term->Rx_product, term->d_middle_diag, rank);
            }
            else if (term->lowrank_middle_type == PDHCG_D_DENSE)
            {
                DEVICE_CHECK(pdhcg_device_symv(state->blas_handle, PDHCG_TRIANGLE_LOWER,
                                               rank, &HOST_ONE, term->d_middle_dense, rank,
                                               term->Rx_product, 1, &HOST_ZERO, term->Rx_buffer, 1));
                product = term->Rx_buffer;
            }
            pdhcg_spmv_execute(state->sparse_handle, term->spmv_ctx_Rt, &HOST_ONE, &HOST_ONE,
                                product, state->dual_product);
        }
        linearized_primal_kernel<<<primal_blocks, THREADS_PER_BLOCK, 0, stream>>>(
            n,
            state->current_primal_solution,
            state->initial_primal_solution,
            state->dual_product,
            state->objective_vector,
            state->variable_lower_bound,
            state->variable_upper_bound,
            state->reflected_primal_solution,
            pdhg_x,
            state->d_primal_step_size, state->d_inner_count,
            iteration, params->reflection_coefficient);

        pdhcg_spmv_execute(state->sparse_handle, state->spmv_ctx_A, &HOST_ONE, &HOST_ZERO,
                            state->reflected_primal_solution, state->primal_product);
        if (m > 0)
            linearized_dual_kernel<<<dual_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                m,
                state->primal_product,
                state->current_dual_solution,
                state->reflected_dual_solution,
                pdhg_y,
                state->initial_dual_solution,
                state->constraint_lower_bound,
                state->constraint_upper_bound,
                state->affine_cone_offset,
                state->d_dual_step_size, state->d_inner_count,
                iteration, params->reflection_coefficient);
    }
}

static void capture_iterations(pdhg_solver_state_t *state,
                               const pdhg_parameters_t *params,
                               int iterations)
{
    linearized_context_t *context = state->inner_solver->linearized_context;
    cudaStream_t stream = (cudaStream_t)context->stream;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    pdhcg_linearized_iterations(state, params, iterations);
    cudaGraph_t graph = NULL;
    cudaGraphExec_t executable = NULL;
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, NULL, NULL, 0));
    context->graph = executable;
    CUDA_CHECK(cudaGraphDestroy(graph));
}

extern "C" linearized_context_t *pdhcg_device_linearized_create(const pdhg_solver_state_t *state)
{
    const quadratic_objective_term_t *term = state->quadratic_objective_term;
    if (state->grid_context || state->has_variable_cones || state->affine_cones.num_blocks ||
        state->affine_cones.split || state->num_variables <= 0 ||
        (!quadratic_type_has_sparse_component(term->quad_obj_type) &&
         !quadratic_type_has_lowrank_component(term->quad_obj_type)) ||
        !isfinite(term->nonconvexity) || term->nonconvexity < 0)
        return NULL;

    const device_sparse_matrix_csr_t *Q = term->objective_sparse_matrix;
    const device_sparse_matrix_csr_t *R = term->objective_lowrank_matrix;
    const device_sparse_matrix_csr_t *AT = state->constraint_matrix_t;
    int rank = R ? R->num_rows : 0;
    int64_t rows = rank > 0
        ? (int64_t)(term->Rx_product - state->dual_product) + rank : state->num_variables;
    int64_t nonzeros = (int64_t)(Q ? Q->num_nonzeros : 0) + AT->num_nonzeros + (R ? R->num_nonzeros : 0);
    if ((int64_t)state->num_variables + state->num_constraints > INT_MAX ||
        rows > INT_MAX || nonzeros > INT_MAX)
        return NULL;

    linearized_context_t *context =
        (linearized_context_t *)safe_calloc(1, sizeof(linearized_context_t));
    device_sparse_matrix_csr_t *matrix = &context->primal_matrix;
    matrix->num_rows = (int)rows;
    matrix->num_cols = state->num_variables + state->num_constraints;
    matrix->num_nonzeros = (int)nonzeros;
    DEVICE_CHECK(pdhcg_device_allocate((void **)&matrix->row_ptr,
                                       ((size_t)matrix->num_rows + 1) * sizeof(int)));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&matrix->col_ind, (size_t)matrix->num_nonzeros * sizeof(int)));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&matrix->val, (size_t)matrix->num_nonzeros * sizeof(double)));
    /* [Q, -A^T; R, 0] uses an empty row when needed to align the Rx output. */
    const device_sparse_matrix_csr_t empty = {};
    concatenate_primal_matrix_kernel<<<matrix->num_rows / (THREADS_PER_BLOCK / 32) + 1, THREADS_PER_BLOCK>>>(
        Q ? *Q : empty, *AT, R ? *R : empty, *matrix);
    DEVICE_CHECK(pdhcg_device_vector_create(
        &context->vec_primal_dual_solution, matrix->num_cols, state->current_primal_solution));
    DEVICE_CHECK(pdhcg_device_vector_create(&context->vec_product, matrix->num_rows, state->dual_product));
    context->spmv_ctx = pdhcg_spmv_ctx_create(state->sparse_handle,
                                             matrix->num_rows,
                                             matrix->num_cols,
                                             matrix->num_nonzeros,
                                             matrix->row_ptr,
                                             matrix->col_ind,
                                             matrix->val,
                                             context->vec_primal_dual_solution,
                                             context->vec_product);
    cudaStream_t stream = NULL;
    cudaEvent_t ready = NULL;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    context->stream = stream;
    context->ready = ready;
    return context;
}

extern "C" void pdhcg_device_linearized_execute(pdhg_solver_state_t *state,
                                                const pdhg_parameters_t *params,
                                                int iterations)
{
    linearized_context_t *context = state->inner_solver->linearized_context;
    if (!context || iterations <= 0)
        return;
    if (iterations > params->termination_evaluation_frequency)
        CUDA_CHECK(cudaErrorInvalidValue);

    cudaStream_t stream = (cudaStream_t)context->stream;
    cudaEvent_t ready = (cudaEvent_t)context->ready;
    cusparseHandle_t sparse_handle = (cusparseHandle_t)state->sparse_handle;
    cudaStream_t previous_stream = NULL;
    CUSPARSE_CHECK(cusparseGetStream(sparse_handle, &previous_stream));
    CUDA_CHECK(cudaEventRecord(ready, 0));
    CUDA_CHECK(cudaStreamWaitEvent(stream, ready, 0));
    CUSPARSE_CHECK(cusparseSetStream(sparse_handle, stream));
    cublasHandle_t blas_handle = (cublasHandle_t)state->blas_handle;
    cudaStream_t previous_blas_stream = NULL;
    cublasPointerMode_t previous_pointer_mode = CUBLAS_POINTER_MODE_HOST;
    if (state->quadratic_objective_term->d_middle_dense)
    {
        CUBLAS_CHECK(cublasGetStream(blas_handle, &previous_blas_stream));
        CUBLAS_CHECK(cublasGetPointerMode(blas_handle, &previous_pointer_mode));
        CUBLAS_CHECK(cublasSetStream(blas_handle, stream));
        CUBLAS_CHECK(cublasSetPointerMode(blas_handle, CUBLAS_POINTER_MODE_HOST));
    }
    CUDA_CHECK(cudaMemcpyAsync(state->d_inner_count,
                               &state->inner_count,
                               sizeof(int),
                               cudaMemcpyHostToDevice,
                               stream));

    if (iterations == params->termination_evaluation_frequency - 1)
    {
        if (context->graph == NULL)
            capture_iterations(state, params, iterations);
        CUDA_CHECK(cudaGraphLaunch((cudaGraphExec_t)context->graph, stream));
    }
    else
        pdhcg_linearized_iterations(state, params, iterations);

    /* Complete before the caller advances inner_count or evaluates the iterates. */
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUSPARSE_CHECK(cusparseSetStream(sparse_handle, previous_stream));
    if (state->quadratic_objective_term->d_middle_dense)
    {
        CUBLAS_CHECK(cublasSetPointerMode(blas_handle, previous_pointer_mode));
        CUBLAS_CHECK(cublasSetStream(blas_handle, previous_blas_stream));
    }
}

extern "C" void pdhcg_device_linearized_destroy(linearized_context_t *context)
{
    if (!context)
        return;
    if (context->graph != NULL)
        CUDA_CHECK(cudaGraphExecDestroy((cudaGraphExec_t)context->graph));
    pdhcg_spmv_ctx_destroy(context->spmv_ctx);
    DEVICE_CHECK(pdhcg_device_vector_destroy(context->vec_primal_dual_solution));
    DEVICE_CHECK(pdhcg_device_vector_destroy(context->vec_product));
    DEVICE_CHECK(pdhcg_device_free(context->primal_matrix.row_ptr));
    DEVICE_CHECK(pdhcg_device_free(context->primal_matrix.col_ind));
    DEVICE_CHECK(pdhcg_device_free(context->primal_matrix.val));
    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)context->ready));
    CUDA_CHECK(cudaStreamDestroy((cudaStream_t)context->stream));
    free(context);
}
