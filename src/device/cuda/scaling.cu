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
#include "cone_utils.h"
#include "device_scaling.h"
#include "distributed_interface.h"
#include "solver_state.h"
#include "utils.h"
#ifdef PDHCG_COMPILE_DISTRIBUTED
#include "distributed_types.h"
#endif
#include <float.h>
#include <math.h>

#define SCALING_EPSILON 1e-12
#define LOG_SCALING_MIN_ABS 1e-300
#define LOG_SCALING_LIMIT 69.07755278982137
#define PSD_LOG_FACTOR_LIMIT (0.5 * LOG_SCALING_LIMIT)
#define PHASE_TAPER_CONE_THRESHOLD 8
#define SCALING_THREADS 256
#define SCALING_WARPS (SCALING_THREADS / 32)

typedef struct
{
    int start;
    int length;
    int total_length;
    int order;
    int factor_offset;
    cone_type_t type;
} scaling_cone_t;

typedef struct
{
    scaling_cone_t *blocks;
    double *factor_log;
    double *statistics;
    int count;
} scaling_cones_t;

typedef struct
{
    double *row_scale;
    double *column_scale;
    double *quadratic_stat;
    double *rank_stat;
    double *middle_weight;
    double *global_column_scale;
    scaling_cones_t variable_cones;
    scaling_cones_t affine_cones;
    scaling_cones_t split_variable_cones;
    scaling_cones_t split_affine_cones;
    bool include_quadratic;
    bool preserve_cones;
    bool reduce_rows;
    bool reduce_columns;
} scaling_workspace_t;

static int vector_blocks(int count)
{
    return (int)(((size_t)count + SCALING_THREADS - 1) / SCALING_THREADS);
}

static int matrix_blocks(int rows)
{
    return (int)(((size_t)rows + SCALING_WARPS - 1) / SCALING_WARPS);
}

static scaling_cones_t upload_cones(const cone_blocks_t *source, bool preserve)
{
    scaling_cones_t result = {};
    result.count = source->num_cones;
    if (!result.count)
        return result;
    scaling_cone_t *blocks = (scaling_cone_t *)safe_malloc((size_t)result.count * sizeof(scaling_cone_t));
    int factor_count = 0;
    for (int block = 0; block < result.count; ++block)
    {
        blocks[block].start = source->start_idx[block];
        blocks[block].length = cone_block_length(source, block);
        blocks[block].total_length = blocks[block].length;
        blocks[block].order = source->v_dim[block];
        blocks[block].type = source->type[block];
        blocks[block].factor_offset = factor_count;
        if (!preserve && blocks[block].type == CONE_PSD)
            factor_count += blocks[block].order;
    }
    ALLOC_AND_COPY(result.blocks, blocks, (size_t)result.count * sizeof(scaling_cone_t));
    if (factor_count)
        DEVICE_CHECK(pdhcg_device_allocate((void **)&result.factor_log, (size_t)factor_count * sizeof(double)));
    free(blocks);
    return result;
}

#ifdef PDHCG_COMPILE_DISTRIBUTED
static scaling_cones_t upload_split_cones(const distributed_cone_partition_t *source)
{
    scaling_cones_t result = {};
    result.count = source->num_cones;
    if (!result.count)
        return result;
    scaling_cone_t *blocks = (scaling_cone_t *)safe_calloc((size_t)result.count, sizeof(scaling_cone_t));
    for (int block = 0; block < result.count; ++block)
    {
        blocks[block].start = source->local_start[block];
        blocks[block].length = source->local_count[block];
        blocks[block].total_length = cone_length(source->type[block], source->v_dim[block]);
        blocks[block].type = source->type[block];
    }
    ALLOC_AND_COPY(result.blocks, blocks, (size_t)result.count * sizeof(scaling_cone_t));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&result.statistics, (size_t)2 * result.count * sizeof(double)));
    free(blocks);
    return result;
}
#endif

static void free_cones(scaling_cones_t *cones)
{
    CUDA_CHECK(cudaFree(cones->blocks));
    CUDA_CHECK(cudaFree(cones->factor_log));
    CUDA_CHECK(cudaFree(cones->statistics));
}

__device__ static double warp_sum(double value)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

__device__ static double warp_max(double value)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmax(value, __shfl_down_sync(0xffffffff, value, offset));
    return value;
}

/* Every caller supplies one shared slot per warp and synchronizes before reuse. */
__device__ static double block_sum(double value, double *shared)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0)
        shared[warp] = value;
    __syncthreads();
    value = threadIdx.x < SCALING_WARPS ? shared[lane] : 0.0;
    if (warp == 0)
        value = warp_sum(value);
    if (threadIdx.x == 0)
        shared[0] = value;
    __syncthreads();
    return shared[0];
}

__device__ static double block_max(double value, double *shared)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    value = warp_max(value);
    if (lane == 0)
        shared[warp] = value;
    __syncthreads();
    value = threadIdx.x < SCALING_WARPS ? shared[lane] : 0.0;
    if (warp == 0)
        value = warp_max(value);
    if (threadIdx.x == 0)
        shared[0] = value;
    __syncthreads();
    return shared[0];
}

__global__ static void fill_ones_kernel(double *values, int count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
        values[index] = 1.0;
}

__device__ static double absolute_power(double value, double power)
{
    if (power == 1.0)
        return value;
    if (power == 2.0)
        return value * value;
    if (power == 0.5)
        return sqrt(value);
    return pow(value, power);
}

/* Row traversal of A and A' makes both statistics race-free. */
__global__ static void row_statistic_kernel(
    device_sparse_matrix_csr_t matrix, int rows, bool maximum, double power, bool take_square_root, double *statistic)
{
    int row = blockIdx.x * SCALING_WARPS + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= rows)
        return;
    double value = 0.0;
    for (int index = matrix.row_ptr[row] + lane; index < matrix.row_ptr[row + 1]; index += 32)
    {
        double absolute = fabs(matrix.val[index]);
        value = maximum ? fmax(value, absolute) : value + absolute_power(absolute, power);
    }
    value = maximum ? warp_max(value) : warp_sum(value);
    if (lane == 0)
        statistic[row] = take_square_root ? (value < SCALING_EPSILON ? 1.0 : sqrt(value)) : value;
}

__global__ static void
absolute_matvec_kernel(device_sparse_matrix_csr_t matrix, int rows, const double *vector, double *result, bool add)
{
    int row = blockIdx.x * SCALING_WARPS + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= rows)
        return;
    double value = 0.0;
    for (int index = matrix.row_ptr[row] + lane; index < matrix.row_ptr[row + 1]; index += 32)
        value += fabs(matrix.val[index]) * vector[matrix.col_ind[index]];
    value = warp_sum(value);
    if (lane == 0)
        result[row] = add ? result[row] + value : value;
}

__global__ static void sqrt_statistic_kernel(double *statistic, const double *quadratic, int count, bool maximum)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    double value = statistic[index];
    if (quadratic)
        value = maximum ? fmax(value, quadratic[index]) : value + quadratic[index];
    statistic[index] = value < SCALING_EPSILON ? 1.0 : sqrt(value);
}

__device__ static double clamped_log(double value)
{
    if (isnan(value))
        return 0.0;
    return fmax(-LOG_SCALING_LIMIT, fmin(LOG_SCALING_LIMIT, value));
}

/* Project svec logarithms onto log(d_i d_j), preserving PSD congruence. */
__device__ static void
project_psd_scaling(double *scaling, scaling_cone_t cone, double *factor_log, bool logarithmic, double *shared)
{
    double *factor = factor_log + cone.factor_offset;
    double local_sum = 0.0;
    for (int index = threadIdx.x; index < cone.order; index += blockDim.x)
    {
        double value = 0.0;
        for (int other = 0; other < cone.order; ++other)
        {
            int column = index < other ? index : other;
            int row = index > other ? index : other;
            size_t slot = (size_t)column * cone.order - (size_t)column * (column - 1) / 2 + row - column;
            double entry = scaling[cone.start + slot];
            entry = clamped_log(logarithmic ? entry : log(fmax(entry, LOG_SCALING_MIN_ABS)));
            value += index == other ? 2.0 * entry : entry;
        }
        factor[index] = value;
        local_sum += value;
    }
    double correction = block_sum(local_sum, shared) / (2.0 * cone.order + 2.0);
    __syncthreads();
    for (int index = threadIdx.x; index < cone.order; index += blockDim.x)
        factor[index] =
            fmax(-PSD_LOG_FACTOR_LIMIT, fmin(PSD_LOG_FACTOR_LIMIT, (factor[index] - correction) / (cone.order + 2.0)));
    __syncthreads();
    for (int column = 0; column < cone.order; ++column)
    {
        size_t start = (size_t)column * cone.order - (size_t)column * (column - 1) / 2;
        for (int row = column + threadIdx.x; row < cone.order; row += blockDim.x)
        {
            double value = factor[row] + factor[column];
            scaling[cone.start + start + row - column] = logarithmic ? value : exp(value);
        }
    }
}

__global__ static void
adjust_cone_scaling_kernel(double *scaling, const scaling_cone_t *cones, double *factor_log, bool preserve, bool ruiz)
{
    __shared__ double shared[SCALING_WARPS];
    scaling_cone_t cone = cones[blockIdx.x];
    if (!preserve)
    {
        if (cone.type == CONE_PSD)
            project_psd_scaling(scaling, cone, factor_log, false, shared);
        return;
    }
    double maximum = 0.0;
    double square_sum = 0.0;
    for (int index = threadIdx.x; index < cone.length; index += blockDim.x)
    {
        double value = scaling[cone.start + index];
        maximum = fmax(maximum, value);
        square_sum += value * value;
    }
    maximum = block_max(maximum, shared);
    __syncthreads();
    square_sum = block_sum(square_sum, shared);
    double rms = sqrt(square_sum / cone.length);
    double value = ruiz ? (cone.length <= PHASE_TAPER_CONE_THRESHOLD ? maximum : rms)
                        : (cone.length <= PHASE_TAPER_CONE_THRESHOLD ? rms : sqrt(maximum * rms));
    for (int index = threadIdx.x; index < cone.length; index += blockDim.x)
        scaling[cone.start + index] = value;
}

/* Only SOC/RSOC blocks with a common scalar factor may cross partitions. */
__global__ static void split_cone_statistics_kernel(
    const double *values, const double *counts, const scaling_cone_t *cones, double *statistics, int num_cones)
{
    __shared__ double shared[SCALING_WARPS];
    scaling_cone_t cone = cones[blockIdx.x];
    double first = 0.0;
    double second = 0.0;
    for (int index = threadIdx.x; index < cone.length; index += blockDim.x)
    {
        double value = values[cone.start + index];
        first = counts ? first + value : fmax(first, value);
        second += counts ? counts[cone.start + index] : value * value;
    }
    first = counts ? block_sum(first, shared) : block_max(first, shared);
    __syncthreads();
    second = block_sum(second, shared);
    if (threadIdx.x == 0)
    {
        statistics[blockIdx.x] = first;
        statistics[num_cones + blockIdx.x] = second;
    }
}

__global__ static void apply_split_cone_scaling_kernel(
    double *scaling, const scaling_cone_t *cones, const double *statistics, int num_cones, bool logarithmic, bool ruiz)
{
    scaling_cone_t cone = cones[blockIdx.x];
    double first = statistics[blockIdx.x];
    double second = statistics[num_cones + blockIdx.x];
    double value;
    if (logarithmic)
        value = second > 0.0 ? first / second : 0.0;
    else
    {
        double rms = sqrt(second / cone.total_length);
        value = ruiz ? (cone.total_length <= PHASE_TAPER_CONE_THRESHOLD ? first : rms)
                     : (cone.total_length <= PHASE_TAPER_CONE_THRESHOLD ? rms : sqrt(first * rms));
    }
    for (int index = threadIdx.x; index < cone.length; index += blockDim.x)
        scaling[cone.start + index] = value;
}

static void adjust_split_cones(double *scaling,
                               const double *residual,
                               const double *counts,
                               scaling_cones_t *cones,
                               grid_context_t *grid,
                               pdhcg_comm_scope_t scope,
                               bool ruiz)
{
    if (!cones->count)
        return;
    split_cone_statistics_kernel<<<cones->count, SCALING_THREADS>>>(
        counts ? residual : scaling, counts, cones->blocks, cones->statistics, cones->count);
    pdhcg_all_reduce_array(grid, cones->statistics, cones->count, counts ? PDHCG_OP_SUM : PDHCG_OP_MAX, scope, 0);
    pdhcg_all_reduce_array(grid, cones->statistics + cones->count, cones->count, PDHCG_OP_SUM, scope, 0);
    apply_split_cone_scaling_kernel<<<cones->count, SCALING_THREADS>>>(
        scaling, cones->blocks, cones->statistics, cones->count, counts != NULL, ruiz);
}

__global__ static void scale_variable_kernel(
    double *objective, double *lower, double *upper, const double *scaling, double *cumulative, int count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    double value = scaling[index];
    objective[index] /= value;
    lower[index] *= value;
    upper[index] *= value;
    cumulative[index] *= value;
}

__global__ static void scale_constraint_kernel(
    double *lower, double *upper, double *offset, const double *scaling, double *cumulative, int count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    double value = scaling[index];
    lower[index] /= value;
    upper[index] /= value;
    offset[index] /= value;
    cumulative[index] *= value;
}

__global__ static void
scale_matrix_kernel(device_sparse_matrix_csr_t matrix, int rows, const double *row_scale, const double *column_scale)
{
    int row = blockIdx.x * SCALING_WARPS + (threadIdx.x >> 5);
    if (row >= rows)
        return;
    for (int index = matrix.row_ptr[row] + (threadIdx.x & 31); index < matrix.row_ptr[row + 1]; index += 32)
    {
        double denominator = row_scale ? row_scale[row] : 1.0;
        if (column_scale)
            denominator *= column_scale[matrix.col_ind[index]];
        matrix.val[index] /= denominator;
    }
}

static void
scale_matrix(device_sparse_matrix_csr_t *matrix, int rows, const double *row_scale, const double *column_scale)
{
    if (matrix && rows)
        scale_matrix_kernel<<<matrix_blocks(rows), SCALING_THREADS>>>(*matrix, rows, row_scale, column_scale);
}

/* Q is column-partitioned, so its row factors span the global variable axis. */
static const double *
quadratic_row_scaling(pdhg_solver_state_t *state, scaling_workspace_t *workspace, const double *local_scaling)
{
    if (!workspace->global_column_scale)
        return local_scaling;
    int global_columns = pdhcg_get_global_num_variables(state->grid_context);
    int start = pdhcg_get_variable_start(state->grid_context);
    CUDA_CHECK(cudaMemset(workspace->global_column_scale, 0, (size_t)global_columns * sizeof(double)));
    if (state->num_variables)
        CUDA_CHECK(cudaMemcpy(workspace->global_column_scale + start,
                              local_scaling,
                              (size_t)state->num_variables * sizeof(double),
                              cudaMemcpyDeviceToDevice));
    pdhcg_all_reduce_array(
        state->grid_context, workspace->global_column_scale, global_columns, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, 0);
    return workspace->global_column_scale;
}

static void scale_problem(pdhg_solver_state_t *state, scaling_workspace_t *workspace)
{
    quadratic_objective_term_t *term = state->quadratic_objective_term;
    int rows = state->num_constraints;
    int columns = state->num_variables;
    if (columns)
        scale_variable_kernel<<<vector_blocks(columns), SCALING_THREADS>>>(state->objective_vector,
                                                                           state->variable_lower_bound,
                                                                           state->variable_upper_bound,
                                                                           workspace->column_scale,
                                                                           state->variable_rescaling,
                                                                           columns);
    if (rows)
        scale_constraint_kernel<<<vector_blocks(rows), SCALING_THREADS>>>(state->constraint_lower_bound,
                                                                          state->constraint_upper_bound,
                                                                          state->affine_cone_offset,
                                                                          workspace->row_scale,
                                                                          state->constraint_rescaling,
                                                                          rows);
    scale_matrix(state->constraint_matrix, rows, workspace->row_scale, workspace->column_scale);
    scale_matrix(state->constraint_matrix_t, columns, workspace->column_scale, workspace->row_scale);
    if (workspace->include_quadratic)
    {
        const double *q_row_scale = quadratic_row_scaling(state, workspace, workspace->column_scale);
        int q_rows = term->objective_sparse_matrix ? term->objective_sparse_matrix->num_rows : 0;
        scale_matrix(term->objective_sparse_matrix, q_rows, q_row_scale, workspace->column_scale);
        scale_matrix(term->objective_lowrank_matrix, term->num_rank_lowrank_obj, NULL, workspace->column_scale);
        scale_matrix(term->objective_lowrank_matrix_t, columns, workspace->column_scale, NULL);
    }
}

static void adjust_cones(double *scaling, const scaling_cones_t *cones, bool preserve, bool ruiz)
{
    if (cones->count)
        adjust_cone_scaling_kernel<<<cones->count, SCALING_THREADS>>>(
            scaling, cones->blocks, cones->factor_log, preserve, ruiz);
}

static void quadratic_statistics(pdhg_solver_state_t *state,
                                 const qp_problem_t *device_problem,
                                 scaling_workspace_t *workspace,
                                 bool maximum)
{
    int columns = state->num_variables;
    quadratic_objective_term_t *term = state->quadratic_objective_term;
    int rank = term->num_rank_lowrank_obj;
    const device_sparse_matrix_csr_t *Q = term->objective_sparse_matrix;
    if (Q)
    {
        /* Symmetry identifies column statistics with full row statistics. */
        double *statistics =
            workspace->global_column_scale ? workspace->global_column_scale : workspace->quadratic_stat;
        if (Q->num_rows)
            row_statistic_kernel<<<matrix_blocks(Q->num_rows), SCALING_THREADS>>>(
                *Q, Q->num_rows, maximum, 1.0, false, statistics);
        if (workspace->global_column_scale)
        {
            pdhcg_all_reduce_array(state->grid_context,
                                   statistics,
                                   Q->num_rows,
                                   maximum ? PDHCG_OP_MAX : PDHCG_OP_SUM,
                                   PDHCG_SCOPE_ROW,
                                   0);
            if (columns)
                CUDA_CHECK(cudaMemcpy(workspace->quadratic_stat,
                                      statistics + pdhcg_get_variable_start(state->grid_context),
                                      (size_t)columns * sizeof(double),
                                      cudaMemcpyDeviceToDevice));
        }
    }
    else
        CUDA_CHECK(cudaMemset(workspace->quadratic_stat, 0, (size_t)columns * sizeof(double)));
    if (rank <= 0)
        return;
    if (term->objective_lowrank_matrix)
        row_statistic_kernel<<<matrix_blocks(rank), SCALING_THREADS>>>(
            *term->objective_lowrank_matrix, rank, maximum, 1.0, false, workspace->rank_stat);
    else
        CUDA_CHECK(cudaMemset(workspace->rank_stat, 0, (size_t)rank * sizeof(double)));
    pdhcg_all_reduce_array(
        state->grid_context, workspace->rank_stat, rank, maximum ? PDHCG_OP_MAX : PDHCG_OP_SUM, PDHCG_SCOPE_ROW, 0);
    const double *weight = workspace->rank_stat;
    const CsrComponent *D = device_problem->objective_lowrank_middle_matrix;
    if (D && D->row_ptr && device_problem->objective_lowrank_middle_matrix_num_nonzeros > 0)
    {
        device_sparse_matrix_csr_t middle = {
            rank, rank, device_problem->objective_lowrank_middle_matrix_num_nonzeros, D->row_ptr, D->col_ind, D->val};
        absolute_matvec_kernel<<<matrix_blocks(rank), SCALING_THREADS>>>(
            middle, rank, workspace->rank_stat, workspace->middle_weight, false);
        weight = workspace->middle_weight;
    }
    if (columns && term->objective_lowrank_matrix_t)
        absolute_matvec_kernel<<<matrix_blocks(columns), SCALING_THREADS>>>(
            *term->objective_lowrank_matrix_t, columns, weight, workspace->quadratic_stat, true);
}

static void equilibrate(pdhg_solver_state_t *state,
                        const qp_problem_t *device_problem,
                        scaling_workspace_t *workspace,
                        bool ruiz,
                        double alpha)
{
    int rows = state->num_constraints;
    int columns = state->num_variables;
    if (rows)
        row_statistic_kernel<<<matrix_blocks(rows), SCALING_THREADS>>>(
            *state->constraint_matrix, rows, ruiz, alpha, !workspace->reduce_rows, workspace->row_scale);
    if (columns)
        row_statistic_kernel<<<matrix_blocks(columns), SCALING_THREADS>>>(*state->constraint_matrix_t,
                                                                          columns,
                                                                          ruiz,
                                                                          2.0 - alpha,
                                                                          !workspace->include_quadratic &&
                                                                              !workspace->reduce_columns,
                                                                          workspace->column_scale);
    grid_context_t *grid = state->grid_context;
    if (workspace->reduce_rows)
    {
        pdhcg_all_reduce_array(
            grid, workspace->row_scale, rows, ruiz ? PDHCG_OP_MAX : PDHCG_OP_SUM, PDHCG_SCOPE_ROW, 0);
        if (rows)
            sqrt_statistic_kernel<<<vector_blocks(rows), SCALING_THREADS>>>(workspace->row_scale, NULL, rows, ruiz);
    }
    if (workspace->reduce_columns)
        pdhcg_all_reduce_array(
            grid, workspace->column_scale, columns, ruiz ? PDHCG_OP_MAX : PDHCG_OP_SUM, PDHCG_SCOPE_COL, 0);
    if (workspace->include_quadratic)
        quadratic_statistics(state, device_problem, workspace, ruiz);
    if (columns && (workspace->include_quadratic || workspace->reduce_columns))
        sqrt_statistic_kernel<<<vector_blocks(columns), SCALING_THREADS>>>(
            workspace->column_scale, workspace->quadratic_stat, columns, ruiz);
    adjust_cones(workspace->row_scale, &workspace->affine_cones, workspace->preserve_cones, ruiz);
    adjust_cones(workspace->column_scale, &workspace->variable_cones, workspace->preserve_cones, ruiz);
    adjust_split_cones(workspace->row_scale, NULL, NULL, &workspace->split_affine_cones, grid, PDHCG_SCOPE_COL, ruiz);
    adjust_split_cones(
        workspace->column_scale, NULL, NULL, &workspace->split_variable_cones, grid, PDHCG_SCOPE_ROW, ruiz);
    scale_problem(state, workspace);
}

__global__ static void row_log_sum_kernel(device_sparse_matrix_csr_t matrix, int rows, double *log_sum)
{
    int row = blockIdx.x * SCALING_WARPS + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= rows)
        return;
    double value = 0.0;
    for (int index = matrix.row_ptr[row] + lane; index < matrix.row_ptr[row + 1]; index += 32)
        value += log(fmax(fabs(matrix.val[index]), LOG_SCALING_MIN_ABS));
    value = warp_sum(value);
    if (lane == 0)
        log_sum[row] = value;
}

__global__ static void curtis_reid_update_kernel(device_sparse_matrix_csr_t matrix,
                                                 int rows,
                                                 const double *log_sum,
                                                 const double *other_scale,
                                                 double *scale,
                                                 double *residual_sum)
{
    int row = blockIdx.x * SCALING_WARPS + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= rows)
        return;
    double sum = lane == 0 ? log_sum[row] : 0.0;
    int begin = matrix.row_ptr[row];
    int end = matrix.row_ptr[row + 1];
    for (int index = begin + lane; index < end; index += 32)
        sum -= other_scale[matrix.col_ind[index]];
    sum = warp_sum(sum);
    if (lane == 0)
    {
        residual_sum[row] = sum;
        scale[row] = end > begin ? sum / (end - begin) : 0.0;
    }
}

__global__ static void curtis_reid_cones_kernel(double *scale,
                                                const double *residual_sum,
                                                const int *row_ptr,
                                                const double *global_counts,
                                                const scaling_cone_t *cones,
                                                double *factor_log,
                                                bool preserve)
{
    __shared__ double shared[SCALING_WARPS];
    scaling_cone_t cone = cones[blockIdx.x];
    if (!preserve)
    {
        if (cone.type == CONE_PSD)
            project_psd_scaling(scale, cone, factor_log, true, shared);
        return;
    }
    double sum = 0.0;
    double count = 0.0;
    for (int index = threadIdx.x; index < cone.length; index += blockDim.x)
    {
        sum += residual_sum[cone.start + index];
        if (global_counts)
            count += global_counts[cone.start + index];
    }
    sum = block_sum(sum, shared);
    if (global_counts)
    {
        __syncthreads();
        count = block_sum(count, shared);
    }
    else
        count = row_ptr[cone.start + cone.length] - row_ptr[cone.start];
    double value = count > 0 ? sum / count : 0.0;
    for (int index = threadIdx.x; index < cone.length; index += blockDim.x)
        scale[cone.start + index] = value;
}

__global__ static void exponentiate_scaling_kernel(double *scale, int count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
        scale[index] = exp(clamped_log(scale[index]));
}

__global__ static void row_count_kernel(const int *row_ptr, int rows, double *counts)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows)
        counts[row] = row_ptr[row + 1] - row_ptr[row];
}

__global__ static void normalize_residual_kernel(const double *residual, const double *counts, int size, double *scale)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        scale[index] = counts[index] > 0.0 ? residual[index] / counts[index] : 0.0;
}

static void curtis_reid_distributed(pdhg_solver_state_t *state, scaling_workspace_t *workspace, int iterations)
{
    int rows = state->num_constraints;
    int columns = state->num_variables;
    grid_context_t *grid = state->grid_context;
    device_sparse_matrix_csr_t A = *state->constraint_matrix;
    device_sparse_matrix_csr_t At = *state->constraint_matrix_t;
    double *row_log_sum = NULL;
    DEVICE_CHECK(pdhcg_device_allocate((void **)&row_log_sum, (size_t)rows * sizeof(double)));
    double *column_log_sum = NULL;
    DEVICE_CHECK(pdhcg_device_allocate((void **)&column_log_sum, (size_t)columns * sizeof(double)));
    double *row_counts = NULL;
    DEVICE_CHECK(pdhcg_device_allocate((void **)&row_counts, (size_t)rows * sizeof(double)));
    double *column_counts = NULL;
    DEVICE_CHECK(pdhcg_device_allocate((void **)&column_counts, (size_t)columns * sizeof(double)));
    double *residual = NULL;
    DEVICE_CHECK(pdhcg_device_allocate((void **)&residual, (size_t)(rows > columns ? rows : columns) * sizeof(double)));
    CUDA_CHECK(cudaMemset(workspace->column_scale, 0, (size_t)columns * sizeof(double)));
    if (rows)
    {
        row_log_sum_kernel<<<matrix_blocks(rows), SCALING_THREADS>>>(A, rows, row_log_sum);
        row_count_kernel<<<vector_blocks(rows), SCALING_THREADS>>>(A.row_ptr, rows, row_counts);
    }
    if (columns)
    {
        row_log_sum_kernel<<<matrix_blocks(columns), SCALING_THREADS>>>(At, columns, column_log_sum);
        row_count_kernel<<<vector_blocks(columns), SCALING_THREADS>>>(At.row_ptr, columns, column_counts);
    }
    pdhcg_all_reduce_array(grid, row_counts, rows, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, 0);
    pdhcg_all_reduce_array(grid, column_counts, columns, PDHCG_OP_SUM, PDHCG_SCOPE_COL, 0);
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        if (rows)
            curtis_reid_update_kernel<<<matrix_blocks(rows), SCALING_THREADS>>>(
                A, rows, row_log_sum, workspace->column_scale, workspace->row_scale, residual);
        pdhcg_all_reduce_array(grid, residual, rows, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, 0);
        if (rows)
            normalize_residual_kernel<<<vector_blocks(rows), SCALING_THREADS>>>(
                residual, row_counts, rows, workspace->row_scale);
        if (workspace->affine_cones.count)
            curtis_reid_cones_kernel<<<workspace->affine_cones.count, SCALING_THREADS>>>(
                workspace->row_scale,
                residual,
                A.row_ptr,
                row_counts,
                workspace->affine_cones.blocks,
                workspace->affine_cones.factor_log,
                workspace->preserve_cones);
        adjust_split_cones(
            workspace->row_scale, residual, row_counts, &workspace->split_affine_cones, grid, PDHCG_SCOPE_COL, false);
        if (columns)
            curtis_reid_update_kernel<<<matrix_blocks(columns), SCALING_THREADS>>>(
                At, columns, column_log_sum, workspace->row_scale, workspace->column_scale, residual);
        pdhcg_all_reduce_array(grid, residual, columns, PDHCG_OP_SUM, PDHCG_SCOPE_COL, 0);
        if (columns)
            normalize_residual_kernel<<<vector_blocks(columns), SCALING_THREADS>>>(
                residual, column_counts, columns, workspace->column_scale);
        if (workspace->variable_cones.count)
            curtis_reid_cones_kernel<<<workspace->variable_cones.count, SCALING_THREADS>>>(
                workspace->column_scale,
                residual,
                At.row_ptr,
                column_counts,
                workspace->variable_cones.blocks,
                workspace->variable_cones.factor_log,
                workspace->preserve_cones);
        adjust_split_cones(workspace->column_scale,
                           residual,
                           column_counts,
                           &workspace->split_variable_cones,
                           grid,
                           PDHCG_SCOPE_ROW,
                           false);
    }
    if (rows)
        exponentiate_scaling_kernel<<<vector_blocks(rows), SCALING_THREADS>>>(workspace->row_scale, rows);
    if (columns)
        exponentiate_scaling_kernel<<<vector_blocks(columns), SCALING_THREADS>>>(workspace->column_scale, columns);
    scale_problem(state, workspace);
    CUDA_CHECK(cudaFree(row_log_sum));
    CUDA_CHECK(cudaFree(column_log_sum));
    CUDA_CHECK(cudaFree(row_counts));
    CUDA_CHECK(cudaFree(column_counts));
    CUDA_CHECK(cudaFree(residual));
}

static void curtis_reid(pdhg_solver_state_t *state, scaling_workspace_t *workspace, int iterations)
{
    if (state->grid_context)
    {
        curtis_reid_distributed(state, workspace, iterations);
        return;
    }
    int rows = state->num_constraints;
    int columns = state->num_variables;
    if (!rows || !columns || !state->constraint_matrix->num_nonzeros)
        return;
    double *row_log_sum = NULL;
    DEVICE_CHECK(pdhcg_device_allocate((void **)&row_log_sum, (size_t)rows * sizeof(double)));
    double *column_log_sum = NULL;
    DEVICE_CHECK(pdhcg_device_allocate((void **)&column_log_sum, (size_t)columns * sizeof(double)));
    double *residual_sum = NULL;
    DEVICE_CHECK(
        pdhcg_device_allocate((void **)&residual_sum, (size_t)(rows > columns ? rows : columns) * sizeof(double)));
    CUDA_CHECK(cudaMemset(workspace->column_scale, 0, (size_t)columns * sizeof(double)));
    device_sparse_matrix_csr_t A = *state->constraint_matrix;
    device_sparse_matrix_csr_t At = *state->constraint_matrix_t;
    row_log_sum_kernel<<<matrix_blocks(rows), SCALING_THREADS>>>(A, rows, row_log_sum);
    row_log_sum_kernel<<<matrix_blocks(columns), SCALING_THREADS>>>(At, columns, column_log_sum);
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        curtis_reid_update_kernel<<<matrix_blocks(rows), SCALING_THREADS>>>(
            A, rows, row_log_sum, workspace->column_scale, workspace->row_scale, residual_sum);
        if (workspace->affine_cones.count)
            curtis_reid_cones_kernel<<<workspace->affine_cones.count, SCALING_THREADS>>>(
                workspace->row_scale,
                residual_sum,
                A.row_ptr,
                NULL,
                workspace->affine_cones.blocks,
                workspace->affine_cones.factor_log,
                workspace->preserve_cones);
        curtis_reid_update_kernel<<<matrix_blocks(columns), SCALING_THREADS>>>(
            At, columns, column_log_sum, workspace->row_scale, workspace->column_scale, residual_sum);
        if (workspace->variable_cones.count)
            curtis_reid_cones_kernel<<<workspace->variable_cones.count, SCALING_THREADS>>>(
                workspace->column_scale,
                residual_sum,
                At.row_ptr,
                NULL,
                workspace->variable_cones.blocks,
                workspace->variable_cones.factor_log,
                workspace->preserve_cones);
    }
    exponentiate_scaling_kernel<<<vector_blocks(rows), SCALING_THREADS>>>(workspace->row_scale, rows);
    exponentiate_scaling_kernel<<<vector_blocks(columns), SCALING_THREADS>>>(workspace->column_scale, columns);
    scale_problem(state, workspace);
    CUDA_CHECK(cudaFree(row_log_sum));
    CUDA_CHECK(cudaFree(column_log_sum));
    CUDA_CHECK(cudaFree(residual_sum));
}

__global__ static void bound_norm_kernel(const double *lower_bound,
                                         const double *upper_bound,
                                         const double *offsets,
                                         const double *objective_vector,
                                         int rows,
                                         int columns,
                                         double *partials)
{
    __shared__ double shared[SCALING_WARPS];
    double bound_sum = 0.0;
    double objective_sum = 0.0;
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t row = index; row < (size_t)rows; row += stride)
    {
        double lower = lower_bound[row];
        double upper = upper_bound[row];
        double offset = offsets[row];
        if (isfinite(lower) && lower != upper)
            bound_sum += lower * lower;
        if (isfinite(upper))
            bound_sum += upper * upper;
        bound_sum += offset * offset;
    }
    for (size_t column = index; column < (size_t)columns; column += stride)
    {
        double objective = objective_vector[column];
        objective_sum += objective * objective;
    }
    bound_sum = block_sum(bound_sum, shared);
    if (threadIdx.x == 0)
        partials[blockIdx.x] = bound_sum;
    __syncthreads();
    objective_sum = block_sum(objective_sum, shared);
    if (threadIdx.x == 0)
        partials[gridDim.x + blockIdx.x] = objective_sum;
}

__global__ static void finalize_norm_kernel(const double *partials, int count, double *norms, bool normalize)
{
    __shared__ double shared[SCALING_WARPS];
    double value = 0.0;
    for (int index = threadIdx.x; index < count; index += blockDim.x)
        value += partials[blockIdx.x * count + index];
    value = block_sum(value, shared);
    if (threadIdx.x == 0)
        norms[blockIdx.x] = normalize ? 1.0 / (sqrt(value) + 1.0) : value;
}

__global__ static void normalize_bound_norms_kernel(double *norms)
{
    int index = threadIdx.x;
    if (index < 2)
        norms[index] = 1.0 / (sqrt(norms[index]) + 1.0);
}

__global__ static void scale_global_bounds_kernel(double *constraint_lower,
                                                  double *constraint_upper,
                                                  double *offset,
                                                  double *variable_lower,
                                                  double *variable_upper,
                                                  double *objective,
                                                  int rows,
                                                  int columns,
                                                  double bound_scale,
                                                  double objective_scale)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows)
    {
        constraint_lower[index] *= bound_scale;
        constraint_upper[index] *= bound_scale;
        offset[index] *= bound_scale;
    }
    if (index < columns)
    {
        variable_lower[index] *= bound_scale;
        variable_upper[index] *= bound_scale;
        objective[index] *= objective_scale;
    }
}

__global__ static void scale_matrix_values_kernel(double *values, int count, double scaling)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
        values[index] *= scaling;
}

static void scale_matrix_values(device_sparse_matrix_csr_t *matrix, int count, double scaling)
{
    if (matrix && count)
        scale_matrix_values_kernel<<<vector_blocks(count), SCALING_THREADS>>>(matrix->val, count, scaling);
}

static void scale_bounds_and_objective(pdhg_solver_state_t *state)
{
    int rows = state->num_constraints;
    int columns = state->num_variables;
    int count = rows > columns ? rows : columns;
    int blocks = vector_blocks(count);
    blocks = blocks > 4096 ? 4096 : blocks;
    if (!blocks && !state->grid_context)
        return;
    double *partials = NULL;
    DEVICE_CHECK(pdhcg_device_allocate((void **)&partials, (size_t)2 * blocks * sizeof(double)));
    double *norms = NULL;
    DEVICE_CHECK(pdhcg_device_allocate((void **)&norms, 2 * sizeof(double)));
    if (blocks)
    {
        bound_norm_kernel<<<blocks, SCALING_THREADS>>>(state->constraint_lower_bound,
                                                       state->constraint_upper_bound,
                                                       state->affine_cone_offset,
                                                       state->objective_vector,
                                                       rows,
                                                       columns,
                                                       partials);
        finalize_norm_kernel<<<2, SCALING_THREADS>>>(partials, blocks, norms, state->grid_context == NULL);
    }
    else
        CUDA_CHECK(cudaMemset(norms, 0, 2 * sizeof(double)));
    if (state->grid_context)
    {
        /* Bounds are replicated across grid columns; the objective across rows. */
        pdhcg_all_reduce_scalar(state->grid_context, norms, PDHCG_OP_SUM, PDHCG_SCOPE_COL, true);
        pdhcg_all_reduce_scalar(state->grid_context, norms + 1, PDHCG_OP_SUM, PDHCG_SCOPE_ROW, true);
        normalize_bound_norms_kernel<<<1, 32>>>(norms);
    }
    double factors[2];
    CUDA_CHECK(cudaMemcpy(factors, norms, sizeof(factors), cudaMemcpyDeviceToHost));
    state->constraint_bound_rescaling = factors[0];
    state->objective_vector_rescaling = factors[1];
    if (count)
        scale_global_bounds_kernel<<<vector_blocks(count), SCALING_THREADS>>>(state->constraint_lower_bound,
                                                                              state->constraint_upper_bound,
                                                                              state->affine_cone_offset,
                                                                              state->variable_lower_bound,
                                                                              state->variable_upper_bound,
                                                                              state->objective_vector,
                                                                              rows,
                                                                              columns,
                                                                              factors[0],
                                                                              factors[1]);
    CUDA_CHECK(cudaFree(partials));
    CUDA_CHECK(cudaFree(norms));
}

__global__ static void finalize_starts_kernel(double *primal,
                                              double *dual,
                                              double *lower,
                                              double *upper,
                                              const double *variable_scale,
                                              const double *constraint_scale,
                                              double bound_scale,
                                              double objective_scale,
                                              int columns,
                                              int rows,
                                              const char *fixed)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < columns)
    {
        double value = 0.0;
        if (primal)
        {
            value = primal[index] * variable_scale[index] * bound_scale;
            primal[index] = value;
        }
        if (fixed && fixed[index])
        {
            lower[index] = value;
            upper[index] = value;
        }
    }
    if (index < rows && dual)
        dual[index] = dual[index] * constraint_scale[index] * objective_scale;
}

/* Defer the quadratic scaling without introducing overflow or underflow in an
   intermediate quotient that later factors would cancel. The ordinary range
   uses direct arithmetic; only exceptional intermediates need exponent scaling. */
__device__ static double apply_accumulated_scaling(double value, double row, double column, double multiplier)
{
    double result = value / row;
    bool exceptional = !isfinite(result) || fabs(result) < DBL_MIN;
    result /= column;
    exceptional = exceptional || !isfinite(result) || fabs(result) < DBL_MIN;
    result *= multiplier;
    if (exceptional && value != 0.0 && isfinite(value) && row > 0.0 && isfinite(row) && column > 0.0 &&
        isfinite(column) && multiplier > 0.0 && isfinite(multiplier))
    {
        int value_exponent, row_exponent, column_exponent, multiplier_exponent;
        double fraction = frexp(value, &value_exponent) / frexp(row, &row_exponent);
        fraction = fraction / frexp(column, &column_exponent) * frexp(multiplier, &multiplier_exponent);
        return ldexp(fraction, value_exponent - row_exponent - column_exponent + multiplier_exponent);
    }
    return result;
}

/* Q and R do not influence this equilibration path, so apply their accumulated
   diagonal factors and final objective scale together in one pass. */
__global__ static void finalize_matrix_kernel(device_sparse_matrix_csr_t matrix,
                                              const double *row_scale,
                                              const double *column_scale,
                                              double multiplier)
{
    int row = blockIdx.x * SCALING_WARPS + (threadIdx.x >> 5);
    if (row >= matrix.num_rows)
        return;
    double row_factor = row_scale ? row_scale[row] : 1.0;
    for (int index = matrix.row_ptr[row] + (threadIdx.x & 31); index < matrix.row_ptr[row + 1]; index += 32)
    {
        double column_factor = column_scale ? column_scale[matrix.col_ind[index]] : 1.0;
        matrix.val[index] = apply_accumulated_scaling(matrix.val[index], row_factor, column_factor, multiplier);
    }
}

__global__ static void finalize_diagonal_kernel(
    double *diagonal, const int *row_ptr, const double *variable_scale, double multiplier, int columns, int start)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= columns)
        return;
    int row = start + column;
    if (row_ptr && row_ptr[row] < row_ptr[row + 1])
    {
        double factor = variable_scale[column];
        diagonal[column] = apply_accumulated_scaling(diagonal[column], factor, factor, multiplier) + 1e-12;
    }
    else
        diagonal[column] = 0.0;
}

static void
finalize_quadratic(pdhg_solver_state_t *state, const qp_problem_t *device_problem, scaling_workspace_t *workspace)
{
    quadratic_objective_term_t *term = state->quadratic_objective_term;
    double multiplier = state->objective_vector_rescaling / state->constraint_bound_rescaling;
    double rank_multiplier = sqrt(multiplier);
    if (term->quad_obj_type == PDHCG_DIAG_Q)
    {
        if (state->num_variables)
            finalize_diagonal_kernel<<<vector_blocks(state->num_variables), SCALING_THREADS>>>(
                term->diagonal_objective_matrix,
                device_problem->objective_sparse_matrix ? device_problem->objective_sparse_matrix->row_ptr : NULL,
                state->variable_rescaling,
                multiplier,
                state->num_variables,
                pdhcg_get_variable_start(state->grid_context));
        return;
    }
    if (workspace->include_quadratic)
    {
        scale_matrix_values(term->objective_sparse_matrix,
                            term->objective_sparse_matrix ? term->objective_sparse_matrix->num_nonzeros : 0,
                            multiplier);
        int rank_nonzeros = term->objective_lowrank_matrix ? term->objective_lowrank_matrix->num_nonzeros : 0;
        scale_matrix_values(term->objective_lowrank_matrix, rank_nonzeros, rank_multiplier);
        scale_matrix_values(term->objective_lowrank_matrix_t, rank_nonzeros, rank_multiplier);
    }
    else
    {
        const double *q_row_scale = quadratic_row_scaling(state, workspace, state->variable_rescaling);
        if (term->objective_sparse_matrix && term->objective_sparse_matrix->num_rows)
            finalize_matrix_kernel<<<matrix_blocks(term->objective_sparse_matrix->num_rows), SCALING_THREADS>>>(
                *term->objective_sparse_matrix, q_row_scale, state->variable_rescaling, multiplier);
        if (term->objective_lowrank_matrix && term->num_rank_lowrank_obj)
            finalize_matrix_kernel<<<matrix_blocks(term->num_rank_lowrank_obj), SCALING_THREADS>>>(
                *term->objective_lowrank_matrix, NULL, state->variable_rescaling, rank_multiplier);
        if (term->objective_lowrank_matrix_t && term->objective_lowrank_matrix_t->num_rows)
            finalize_matrix_kernel<<<matrix_blocks(term->objective_lowrank_matrix_t->num_rows), SCALING_THREADS>>>(
                *term->objective_lowrank_matrix_t, state->variable_rescaling, NULL, rank_multiplier);
    }
}

void pdhcg_device_rescale_problem(const pdhg_parameters_t *params,
                                  const qp_problem_t *device_problem,
                                  pdhg_solver_state_t *state)
{
    int rows = state->num_constraints;
    int columns = state->num_variables;
    quadratic_objective_term_t *term = state->quadratic_objective_term;
    int rank = term->num_rank_lowrank_obj;
    size_t row_bytes = (size_t)rows * sizeof(double);
    size_t column_bytes = (size_t)columns * sizeof(double);
    grid_context_t *grid_context = state->grid_context;
    bool equilibrating =
        params->curtis_reid_iterations > 0 || params->l_inf_ruiz_iterations > 0 || params->has_pock_chambolle_alpha;
    if (columns)
        fill_ones_kernel<<<vector_blocks(columns), SCALING_THREADS>>>(state->variable_rescaling, columns);
    if (rows)
        fill_ones_kernel<<<vector_blocks(rows), SCALING_THREADS>>>(state->constraint_rescaling, rows);
    scaling_workspace_t workspace = {};
    workspace.preserve_cones = params->use_cone_preserving_scaling;
    workspace.include_quadratic = (params->l_inf_ruiz_iterations > 0 || params->has_pock_chambolle_alpha) &&
        uses_linearized_quadratic_update(params->non_diagonal_quadratic_mode, term->quad_obj_type);
    if (equilibrating)
    {
        DEVICE_CHECK(pdhcg_device_allocate((void **)&workspace.row_scale, row_bytes));
        DEVICE_CHECK(pdhcg_device_allocate((void **)&workspace.column_scale, column_bytes));
        workspace.variable_cones = upload_cones(&device_problem->cones, workspace.preserve_cones);
        workspace.affine_cones = upload_cones(&device_problem->affine_cones, workspace.preserve_cones);
    }
#ifdef PDHCG_COMPILE_DISTRIBUTED
    if (grid_context)
    {
        workspace.reduce_rows = grid_context->dims[1] > 1;
        workspace.reduce_columns = grid_context->dims[0] > 1;
        if (equilibrating)
        {
            workspace.split_variable_cones = upload_split_cones(&grid_context->split_cones);
            workspace.split_affine_cones = upload_split_cones(&grid_context->split_affine_cones);
        }
        if (workspace.reduce_rows && quadratic_type_has_sparse_component(term->quad_obj_type))
            DEVICE_CHECK(pdhcg_device_allocate((void **)&workspace.global_column_scale,
                                               (size_t)grid_context->global_num_variables * sizeof(double)));
    }
#endif
    if (workspace.include_quadratic)
    {
        DEVICE_CHECK(pdhcg_device_allocate((void **)&workspace.quadratic_stat, column_bytes));
        DEVICE_CHECK(pdhcg_device_allocate((void **)&workspace.rank_stat, (size_t)rank * sizeof(double)));
        DEVICE_CHECK(pdhcg_device_allocate((void **)&workspace.middle_weight, (size_t)rank * sizeof(double)));
    }
    if (params->curtis_reid_iterations > 0)
        curtis_reid(state, &workspace, params->curtis_reid_iterations);
    for (int iteration = 0; iteration < params->l_inf_ruiz_iterations; ++iteration)
        equilibrate(state, device_problem, &workspace, true, 1.0);
    if (params->has_pock_chambolle_alpha)
        equilibrate(state, device_problem, &workspace, false, params->pock_chambolle_alpha);
    state->constraint_bound_rescaling = 1.0;
    state->objective_vector_rescaling = 1.0;
    if (params->bound_objective_rescaling)
        scale_bounds_and_objective(state);
    finalize_quadratic(state, device_problem, &workspace);
    char *fixed = NULL;
    if (device_problem->cones.is_fixed && columns)
    {
        ALLOC_AND_COPY(fixed, device_problem->cones.is_fixed, (size_t)columns * sizeof(char));
    }
    int count = columns > rows ? columns : rows;
    if (count && (fixed || device_problem->primal_start || device_problem->dual_start))
        finalize_starts_kernel<<<vector_blocks(count), SCALING_THREADS>>>(
            device_problem->primal_start ? state->initial_primal_solution : NULL,
            device_problem->dual_start ? state->initial_dual_solution : NULL,
            state->variable_lower_bound,
            state->variable_upper_bound,
            state->variable_rescaling,
            state->constraint_rescaling,
            state->constraint_bound_rescaling,
            state->objective_vector_rescaling,
            columns,
            rows,
            fixed);
    CUDA_CHECK(cudaFree(fixed));
    CUDA_CHECK(cudaFree(workspace.row_scale));
    CUDA_CHECK(cudaFree(workspace.column_scale));
    CUDA_CHECK(cudaFree(workspace.quadratic_stat));
    CUDA_CHECK(cudaFree(workspace.rank_stat));
    CUDA_CHECK(cudaFree(workspace.middle_weight));
    CUDA_CHECK(cudaFree(workspace.global_column_scale));
    free_cones(&workspace.variable_cones);
    free_cones(&workspace.affine_cones);
    free_cones(&workspace.split_variable_cones);
    free_cones(&workspace.split_affine_cones);
}

__global__ static void
finite_bounds_kernel(const double *lower, const double *upper, double *finite_lower, double *finite_upper, int count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
    {
        finite_lower[index] = isfinite(lower[index]) ? lower[index] : 0.0;
        finite_upper[index] = isfinite(upper[index]) ? upper[index] : 0.0;
    }
}

void pdhcg_device_initialize_finite_bounds(pdhg_solver_state_t *state)
{
    size_t row_bytes = (size_t)state->num_constraints * sizeof(double);
    size_t column_bytes = (size_t)state->num_variables * sizeof(double);
    DEVICE_CHECK(pdhcg_device_allocate((void **)&state->constraint_lower_bound_finite_val, row_bytes));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&state->constraint_upper_bound_finite_val, row_bytes));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&state->variable_lower_bound_finite_val, column_bytes));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&state->variable_upper_bound_finite_val, column_bytes));
    if (state->num_constraints)
        finite_bounds_kernel<<<vector_blocks(state->num_constraints), SCALING_THREADS>>>(
            state->constraint_lower_bound,
            state->constraint_upper_bound,
            state->constraint_lower_bound_finite_val,
            state->constraint_upper_bound_finite_val,
            state->num_constraints);
    if (state->num_variables)
        finite_bounds_kernel<<<vector_blocks(state->num_variables), SCALING_THREADS>>>(
            state->variable_lower_bound,
            state->variable_upper_bound,
            state->variable_lower_bound_finite_val,
            state->variable_upper_bound_finite_val,
            state->num_variables);
}
