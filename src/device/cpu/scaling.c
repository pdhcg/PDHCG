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

#include "device_scaling.h"
#include "cone_utils.h"
#include "solver_state.h"
#include "utils.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define SCALING_EPSILON 1e-12
#define LOG_SCALING_MIN_ABS 1e-300
#define LOG_SCALING_LIMIT 69.07755278982137
#define PSD_LOG_FACTOR_LIMIT (0.5 * LOG_SCALING_LIMIT)
#define PHASE_TAPER_CONE_THRESHOLD 8

typedef enum
{
    CONE_SCALING_RUIZ,
    CONE_SCALING_POCK_CHAMBOLLE,
} cone_scaling_phase_t;

typedef enum
{
    QUADRATIC_COLUMN_MAX,
    QUADRATIC_COLUMN_SUM,
} quadratic_column_stat_t;

static int max_psd_order(const cone_blocks_t *cones)
{
    int max_order = 0;
    for (int block = 0; block < cones->num_cones; ++block)
        if (cones->type[block] == CONE_PSD)
            max_order = cones->v_dim[block] > max_order ? cones->v_dim[block] : max_order;
    return max_order;
}

static void project_psd_log_scaling(double *log_scaling, int start, int order, double *factor_log)
{
    memset(factor_log, 0, (size_t)order * sizeof(double));

    int slot = start;
    for (int column = 0; column < order; ++column)
    {
        for (int row = column; row < order; ++row, ++slot)
        {
            double value = log_scaling[slot];
            if (isnan(value))
                value = 0.0;
            else if (value > LOG_SCALING_LIMIT)
                value = LOG_SCALING_LIMIT;
            else if (value < -LOG_SCALING_LIMIT)
                value = -LOG_SCALING_LIMIT;

            if (row == column)
                factor_log[row] += 2.0 * value;
            else
            {
                factor_log[row] += value;
                factor_log[column] += value;
            }
        }
    }

    double rhs_sum = 0.0;
    for (int index = 0; index < order; ++index)
        rhs_sum += factor_log[index];
    const double correction = rhs_sum / (2.0 * order + 2.0);
    const double diagonal = order + 2.0;
    for (int index = 0; index < order; ++index)
    {
        factor_log[index] = (factor_log[index] - correction) / diagonal;
        factor_log[index] = fmax(-PSD_LOG_FACTOR_LIMIT, fmin(PSD_LOG_FACTOR_LIMIT, factor_log[index]));
    }

    slot = start;
    for (int column = 0; column < order; ++column)
        for (int row = column; row < order; ++row, ++slot)
            log_scaling[slot] = factor_log[row] + factor_log[column];
}

static void project_psd_scaling(double *scaling, int start, int order, double *factor_log)
{
    const int length = cone_length(CONE_PSD, order);
    for (int slot = start; slot < start + length; ++slot)
        scaling[slot] = log(fmax(scaling[slot], LOG_SCALING_MIN_ABS));
    project_psd_log_scaling(scaling, start, order, factor_log);
    for (int slot = start; slot < start + length; ++slot)
        scaling[slot] = exp(scaling[slot]);
}

/*
 * Part of PDHCG's cone-preserving scaling mode follows HPR-SOCP's :phase_taper strategy:
 * https://github.com/PolyU-IOR/HPR-SOCP
 *
 * When that mode is disabled, PSD blocks follow CARDAL's per-element scaling:
 * https://github.com/Lhongpei/CARDAL
 */
static void apply_cone_scaling_adjustment(double *scaling,
                                          const cone_blocks_t *cones,
                                          cone_scaling_phase_t phase,
                                          bool use_cone_preserving_scaling)
{
    int workspace_order = use_cone_preserving_scaling ? 0 : max_psd_order(cones);
    double *factor_log = workspace_order > 0 ? safe_malloc((size_t)workspace_order * sizeof(double)) : NULL;

    for (int block = 0; block < cones->num_cones; ++block)
    {
        if (!use_cone_preserving_scaling)
        {
            if (cones->type[block] == CONE_PSD)
                project_psd_scaling(scaling, cones->start_idx[block], cones->v_dim[block], factor_log);
            continue;
        }

        int start = cones->start_idx[block];
        int length = cone_block_length(cones, block);
        double block_max = 0.0;
        double sum_sq = 0.0;
        for (int index = start; index < start + length; ++index)
        {
            block_max = fmax(block_max, scaling[index]);
            sum_sq += scaling[index] * scaling[index];
        }

        double rms = sqrt(sum_sq / (double)length);
        double block_scale = phase == CONE_SCALING_RUIZ
            ? (length <= PHASE_TAPER_CONE_THRESHOLD ? block_max : rms)
            : (length <= PHASE_TAPER_CONE_THRESHOLD ? rms : sqrt(block_max * rms));
        for (int index = start; index < start + length; ++index)
            scaling[index] = block_scale;
    }

    free(factor_log);
}

static double curtis_reid_exp_clamped(double value)
{
    if (isnan(value))
        return 1.0;
    if (value > LOG_SCALING_LIMIT)
        value = LOG_SCALING_LIMIT;
    else if (value < -LOG_SCALING_LIMIT)
        value = -LOG_SCALING_LIMIT;
    return exp(value);
}

static void scale_problem(pdhg_solver_state_t *state,
                          const double *constraint_rescaling,
                          const double *variable_rescaling,
                          bool include_quadratic_objective)
{
    device_sparse_matrix_csr_t *A = state->constraint_matrix;
    device_sparse_matrix_csr_t *Q = state->quadratic_objective_term->objective_sparse_matrix;
    device_sparse_matrix_csr_t *R = state->quadratic_objective_term->objective_lowrank_matrix;
#ifdef _OPENMP
    const bool use_parallel = state->num_variables > 4096 || state->num_constraints > 4096 ||
        (A->num_nonzeros > 65536 && state->num_constraints > 1) ||
        (include_quadratic_objective &&
         ((Q && Q->num_nonzeros > 65536 && state->num_variables > 1) ||
          (R && R->num_nonzeros > 65536 && R->num_rows > 1)));
#pragma omp parallel if (use_parallel)
#endif
    {
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int i = 0; i < state->num_variables; ++i)
        {
            state->objective_vector[i] /= variable_rescaling[i];
            state->variable_upper_bound[i] *= variable_rescaling[i];
            state->variable_lower_bound[i] *= variable_rescaling[i];
        }
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int i = 0; i < state->num_constraints; ++i)
        {
            state->constraint_lower_bound[i] /= constraint_rescaling[i];
            state->constraint_upper_bound[i] /= constraint_rescaling[i];
            state->affine_cone_offset[i] /= constraint_rescaling[i];
        }

        if (A->num_nonzeros > 0)
        {
#ifdef _OPENMP
#pragma omp for nowait
#endif
            for (int row = 0; row < state->num_constraints; ++row)
            {
                for (int nz_idx = A->row_ptr[row]; nz_idx < A->row_ptr[row + 1]; ++nz_idx)
                {
                    int col = A->col_ind[nz_idx];
                    A->val[nz_idx] /= (constraint_rescaling[row] * variable_rescaling[col]);
                }
            }
        }

        if (include_quadratic_objective)
        {
            if (Q)
            {
#ifdef _OPENMP
#pragma omp for nowait
#endif
                for (int q_row = 0; q_row < state->num_variables; ++q_row)
                {
                    for (int nz_idx = Q->row_ptr[q_row]; nz_idx < Q->row_ptr[q_row + 1]; ++nz_idx)
                    {
                        int q_col = Q->col_ind[nz_idx];
                        Q->val[nz_idx] /= (variable_rescaling[q_row] * variable_rescaling[q_col]);
                    }
                }
            }
            if (R && R->num_nonzeros > 0)
            {
#ifdef _OPENMP
#pragma omp for nowait
#endif
                for (int r = 0; r < R->num_rows; ++r)
                {
                    for (int nz_idx = R->row_ptr[r]; nz_idx < R->row_ptr[r + 1]; ++nz_idx)
                    {
                        int col = R->col_ind[nz_idx];
                        R->val[nz_idx] /= variable_rescaling[col];
                    }
                }
            }
        }
    }
}

/* Keep intermediate quotients representable when later factors cancel them. */
static double apply_accumulated_scaling(double value, double row, double column, double multiplier)
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

/* Q and R need no intermediate writes when they do not determine the factors. */
static void finalize_quadratic_rescaling(pdhg_solver_state_t *state)
{
    const double *variable_rescaling = state->variable_rescaling;
    const double multiplier = state->objective_vector_rescaling / state->constraint_bound_rescaling;
    device_sparse_matrix_csr_t *Q = state->quadratic_objective_term->objective_sparse_matrix;
    if (Q && Q->num_nonzeros > 0)
    {
#ifdef _OPENMP
#pragma omp parallel for if (Q->num_nonzeros > 65536 && state->num_variables > 1)
#endif
        for (int row = 0; row < state->num_variables; ++row)
            for (int nz = Q->row_ptr[row]; nz < Q->row_ptr[row + 1]; ++nz)
                Q->val[nz] = apply_accumulated_scaling(
                    Q->val[nz], variable_rescaling[row], variable_rescaling[Q->col_ind[nz]], multiplier);
    }
    device_sparse_matrix_csr_t *R = state->quadratic_objective_term->objective_lowrank_matrix;
    if (R && R->num_nonzeros > 0)
    {
        const double rank_multiplier = sqrt(multiplier);
#ifdef _OPENMP
#pragma omp parallel for if (R->num_nonzeros > 65536 && state->quadratic_objective_term->num_rank_lowrank_obj > 1)
#endif
        for (int row = 0; row < state->quadratic_objective_term->num_rank_lowrank_obj; ++row)
            for (int nz = R->row_ptr[row]; nz < R->row_ptr[row + 1]; ++nz)
                R->val[nz] =
                    apply_accumulated_scaling(R->val[nz], 1.0, variable_rescaling[R->col_ind[nz]], rank_multiplier);
    }
}

static void pin_scaled_fixed_cone_bounds(pdhg_solver_state_t *state, const qp_problem_t *source_problem)
{
    if (!source_problem->cones.is_fixed)
        return;

    for (int variable = 0; variable < source_problem->num_variables; ++variable)
    {
        if (!source_problem->cones.is_fixed[variable])
            continue;

        double value = source_problem->primal_start ? source_problem->primal_start[variable] : 0.0;
        double scaled_value = value * state->variable_rescaling[variable] * state->constraint_bound_rescaling;
        state->variable_lower_bound[variable] = scaled_value;
        state->variable_upper_bound[variable] = scaled_value;
    }
}

/*
 * A. R. Curtis and J. K. Reid, "On the Automatic Scaling of Matrices
 * for Gaussian Elimination", IMA J. Appl. Math. 10(1), 118-124 (1972).
 */
static void curtis_reid_rescaling(pdhg_solver_state_t *state,
                                  const qp_problem_t *problem,
                                  int num_iterations,
                                  bool include_quadratic_objective,
                                  bool use_cone_preserving_scaling,
                                  double *cum_constraint_rescaling,
                                  double *cum_variable_rescaling)
{
    const int num_cons = state->num_constraints;
    const int num_vars = state->num_variables;
    const int num_nonzeros = state->constraint_matrix->num_nonzeros;
    double *con_rescale = safe_malloc((size_t)num_cons * sizeof(double));
    double *var_rescale = safe_malloc((size_t)num_vars * sizeof(double));
    int psd_workspace_order = 0;
    if (!use_cone_preserving_scaling)
    {
        psd_workspace_order = max_psd_order(&problem->cones);
        int affine_psd_order = max_psd_order(&problem->affine_cones);
        psd_workspace_order = affine_psd_order > psd_workspace_order ? affine_psd_order : psd_workspace_order;
    }
    double *psd_factor_log = psd_workspace_order > 0 ? safe_malloc((size_t)psd_workspace_order * sizeof(double)) : NULL;

    for (int row = 0; row < num_cons; ++row)
        con_rescale[row] = 1.0;
    for (int col = 0; col < num_vars; ++col)
        var_rescale[col] = 1.0;

    if (num_iterations > 0 && num_cons > 0 && num_vars > 0 && num_nonzeros > 0)
    {
        const device_sparse_matrix_csr_t *matrix = state->constraint_matrix;
        double *row_log_scale = safe_calloc((size_t)num_cons, sizeof(double));
        double *col_log_scale = safe_calloc((size_t)num_vars, sizeof(double));
        double *row_log_abs_sum = safe_calloc((size_t)num_cons, sizeof(double));
        double *col_log_abs_sum = safe_calloc((size_t)num_vars, sizeof(double));
        double *col_sum = safe_calloc((size_t)num_vars, sizeof(double));
        int *col_count = safe_calloc((size_t)num_vars, sizeof(int));

        for (int row = 0; row < num_cons; ++row)
        {
            for (int nz = matrix->row_ptr[row]; nz < matrix->row_ptr[row + 1]; ++nz)
            {
                const int col = matrix->col_ind[nz];
                const double log_abs = log(fmax(fabs(matrix->val[nz]), LOG_SCALING_MIN_ABS));
                row_log_abs_sum[row] += log_abs;
                col_log_abs_sum[col] += log_abs;
                ++col_count[col];
            }
        }

        /*
         * Minimize sum_(i,j) in nz(A) (log|A_ij| - r_i - c_j)^2 by
         * alternating exact row and column least-squares updates.
         */
        for (int iter = 0; iter < num_iterations; ++iter)
        {
            for (int row = 0; row < num_cons; ++row)
            {
                const int begin = matrix->row_ptr[row];
                const int end = matrix->row_ptr[row + 1];
                double sum = row_log_abs_sum[row];
                for (int nz = begin; nz < end; ++nz)
                    sum -= col_log_scale[matrix->col_ind[nz]];
                row_log_scale[row] = (end > begin) ? sum / (double)(end - begin) : 0.0;
            }

            for (int block = 0; block < problem->affine_cones.num_cones; ++block)
            {
                if (!use_cone_preserving_scaling)
                {
                    if (problem->affine_cones.type[block] == CONE_PSD)
                    {
                        project_psd_log_scaling(row_log_scale,
                                                problem->affine_cones.start_idx[block],
                                                problem->affine_cones.v_dim[block],
                                                psd_factor_log);
                    }
                    continue;
                }

                int start = problem->affine_cones.start_idx[block];
                int length = cone_block_length(&problem->affine_cones, block);
                double block_sum = 0.0;
                int block_count = 0;
                for (int row = start; row < start + length; ++row)
                {
                    int begin = matrix->row_ptr[row];
                    int end = matrix->row_ptr[row + 1];
                    block_sum += row_log_abs_sum[row];
                    block_count += end - begin;
                    for (int nz = begin; nz < end; ++nz)
                        block_sum -= col_log_scale[matrix->col_ind[nz]];
                }
                double block_log_scale = block_count > 0 ? block_sum / (double)block_count : 0.0;
                for (int row = start; row < start + length; ++row)
                    row_log_scale[row] = block_log_scale;
            }

            memcpy(col_sum, col_log_abs_sum, (size_t)num_vars * sizeof(double));
            for (int row = 0; row < num_cons; ++row)
            {
                for (int nz = matrix->row_ptr[row]; nz < matrix->row_ptr[row + 1]; ++nz)
                    col_sum[matrix->col_ind[nz]] -= row_log_scale[row];
            }
            for (int col = 0; col < num_vars; ++col)
                col_log_scale[col] = col_count[col] > 0 ? col_sum[col] / (double)col_count[col] : 0.0;

            /*
             * Adding c_j = c_B for all j in cone block B gives the exact
             * block minimizer below. With block aggregation disabled, PSD
             * columns use the diagonal-congruence projection instead.
             */
            for (int block = 0; block < problem->cones.num_cones; ++block)
            {
                if (!use_cone_preserving_scaling)
                {
                    if (problem->cones.type[block] == CONE_PSD)
                    {
                        project_psd_log_scaling(col_log_scale,
                                                problem->cones.start_idx[block],
                                                problem->cones.v_dim[block],
                                                psd_factor_log);
                    }
                    continue;
                }

                const int start = problem->cones.start_idx[block];
                const int length = cone_block_length(&problem->cones, block);
                double block_sum = 0.0;
                int block_count = 0;
                for (int col = start; col < start + length; ++col)
                {
                    block_sum += col_sum[col];
                    block_count += col_count[col];
                }
                const double block_log_scale = block_count > 0 ? block_sum / (double)block_count : 0.0;
                for (int col = start; col < start + length; ++col)
                    col_log_scale[col] = block_log_scale;
            }
        }

        for (int row = 0; row < num_cons; ++row)
            con_rescale[row] = curtis_reid_exp_clamped(row_log_scale[row]);
        for (int col = 0; col < num_vars; ++col)
            var_rescale[col] = curtis_reid_exp_clamped(col_log_scale[col]);

        free(row_log_scale);
        free(col_log_scale);
        free(row_log_abs_sum);
        free(col_log_abs_sum);
        free(col_sum);
        free(col_count);
    }

    scale_problem(state, con_rescale, var_rescale, include_quadratic_objective);

    for (int row = 0; row < num_cons; ++row)
        cum_constraint_rescaling[row] *= con_rescale[row];
    for (int col = 0; col < num_vars; ++col)
        cum_variable_rescaling[col] *= var_rescale[col];

    free(con_rescale);
    free(var_rescale);
    free(psd_factor_log);
}

static void accumulate_lowrank_quadratic_column_bound(const pdhg_solver_state_t *state,
                                                      const qp_problem_t *problem,
                                                      quadratic_column_stat_t statistic,
                                                      double *column_stat)
{
    const int rank = state->quadratic_objective_term->num_rank_lowrank_obj;
    const device_sparse_matrix_csr_t *R = state->quadratic_objective_term->objective_lowrank_matrix;
    if (rank <= 0 || !R || R->num_nonzeros <= 0)
        return;

    double *row_stat = safe_calloc((size_t)rank, sizeof(double));
    double *middle_weight = safe_calloc((size_t)rank, sizeof(double));
    for (int row = 0; row < rank; ++row)
    {
        for (int nz = R->row_ptr[row]; nz < R->row_ptr[row + 1]; ++nz)
        {
            const double value = fabs(R->val[nz]);
            if (statistic == QUADRATIC_COLUMN_MAX)
                row_stat[row] = fmax(row_stat[row], value);
            else
                row_stat[row] += value;
        }
    }

    const CsrComponent *D = problem->objective_lowrank_middle_matrix;
    if (D && problem->objective_lowrank_middle_matrix_num_nonzeros > 0)
    {
        for (int row = 0; row < rank; ++row)
            for (int nz = D->row_ptr[row]; nz < D->row_ptr[row + 1]; ++nz)
                middle_weight[row] += fabs(D->val[nz]) * row_stat[D->col_ind[nz]];
    }
    else
    {
        memcpy(middle_weight, row_stat, (size_t)rank * sizeof(double));
    }

    /* Bound the requested column norm of R' D R without forming a dense Hessian. */
    for (int row = 0; row < rank; ++row)
        for (int nz = R->row_ptr[row]; nz < R->row_ptr[row + 1]; ++nz)
            column_stat[R->col_ind[nz]] += fabs(R->val[nz]) * middle_weight[row];

    free(row_stat);
    free(middle_weight);
}

/* Q-aware variable equilibration follows HPR-QP's scaling rule:
 * https://github.com/PolyU-IOR/HPR-QP */
static void compute_quadratic_column_statistics(const pdhg_solver_state_t *state,
                                                const qp_problem_t *problem,
                                                quadratic_column_stat_t statistic,
                                                double *column_stat)
{
    const device_sparse_matrix_csr_t *Q = state->quadratic_objective_term->objective_sparse_matrix;
    if (Q && Q->num_nonzeros > 0)
    {
        for (int row = 0; row < state->num_variables; ++row)
        {
            for (int nz = Q->row_ptr[row]; nz < Q->row_ptr[row + 1]; ++nz)
            {
                const double value = fabs(Q->val[nz]);
                if (statistic == QUADRATIC_COLUMN_MAX)
                    column_stat[row] = fmax(column_stat[row], value);
                else
                    column_stat[row] += value;
            }
        }
    }
    accumulate_lowrank_quadratic_column_bound(state, problem, statistic, column_stat);
}

static void ruiz_rescaling(pdhg_solver_state_t *state,
                           const qp_problem_t *problem,
                           int num_iterations,
                           bool include_quadratic_objective,
                           bool use_cone_preserving_scaling,
                           double *cum_constraint_rescaling,
                           double *cum_variable_rescaling)
{
    int num_cons = state->num_constraints;
    int num_vars = state->num_variables;
    double *con_rescale = safe_malloc(num_cons * sizeof(double));
    double *var_rescale = safe_malloc(num_vars * sizeof(double));
    double *quadratic_column_max = include_quadratic_objective ? safe_malloc((size_t)num_vars * sizeof(double)) : NULL;

    for (int iter = 0; iter < num_iterations; ++iter)
    {
        for (int i = 0; i < num_vars; ++i)
            var_rescale[i] = 0.0;
        for (int i = 0; i < num_cons; ++i)
            con_rescale[i] = 0.0;

        if (state->constraint_matrix->num_nonzeros > 0)
        {
            for (int row = 0; row < num_cons; ++row)
            {
                for (int nz_idx = state->constraint_matrix->row_ptr[row];
                     nz_idx < state->constraint_matrix->row_ptr[row + 1];
                     ++nz_idx)
                {
                    int col = state->constraint_matrix->col_ind[nz_idx];
                    if (col < 0 || col >= num_vars)
                    {
                        fprintf(stderr,
                                "Error: Invalid column index %d at nz_idx %d for row %d. "
                                "Must be in [0, %d).\n",
                                col,
                                nz_idx,
                                row,
                                num_vars);
                    }
                    double val = fabs(state->constraint_matrix->val[nz_idx]);
                    if (val > var_rescale[col])
                        var_rescale[col] = val;
                    if (val > con_rescale[row])
                        con_rescale[row] = val;
                }
            }
        }
        if (include_quadratic_objective)
        {
            memset(quadratic_column_max, 0, (size_t)num_vars * sizeof(double));
            compute_quadratic_column_statistics(state, problem, QUADRATIC_COLUMN_MAX, quadratic_column_max);
            for (int col = 0; col < num_vars; ++col)
                var_rescale[col] = fmax(var_rescale[col], quadratic_column_max[col]);
        }
        for (int i = 0; i < num_vars; ++i)
            var_rescale[i] = (var_rescale[i] < SCALING_EPSILON) ? 1.0 : sqrt(var_rescale[i]);
        for (int i = 0; i < num_cons; ++i)
            con_rescale[i] = (con_rescale[i] < SCALING_EPSILON) ? 1.0 : sqrt(con_rescale[i]);

        apply_cone_scaling_adjustment(var_rescale, &problem->cones, CONE_SCALING_RUIZ, use_cone_preserving_scaling);
        apply_cone_scaling_adjustment(
            con_rescale, &problem->affine_cones, CONE_SCALING_RUIZ, use_cone_preserving_scaling);

        scale_problem(state, con_rescale, var_rescale, include_quadratic_objective);
        for (int i = 0; i < num_vars; ++i)
            cum_variable_rescaling[i] *= var_rescale[i];
        for (int i = 0; i < num_cons; ++i)
            cum_constraint_rescaling[i] *= con_rescale[i];
    }
    free(quadratic_column_max);
    free(con_rescale);
    free(var_rescale);
}

static void pock_chambolle_rescaling(pdhg_solver_state_t *state,
                                     const qp_problem_t *problem,
                                     double alpha,
                                     bool include_quadratic_objective,
                                     bool use_cone_preserving_scaling,
                                     double *cum_constraint_rescaling,
                                     double *cum_variable_rescaling)
{
    int num_cons = state->num_constraints;
    int num_vars = state->num_variables;
    double *con_rescale = safe_calloc(num_cons, sizeof(double));
    double *var_rescale = safe_calloc(num_vars, sizeof(double));

    if (state->constraint_matrix->num_nonzeros > 0)
    {
        for (int row = 0; row < num_cons; ++row)
        {
            for (int nz_idx = state->constraint_matrix->row_ptr[row];
                 nz_idx < state->constraint_matrix->row_ptr[row + 1];
                 ++nz_idx)
            {
                int col = state->constraint_matrix->col_ind[nz_idx];
                double val = fabs(state->constraint_matrix->val[nz_idx]);
                var_rescale[col] += pow(val, 2.0 - alpha);
                con_rescale[row] += pow(val, alpha);
            }
        }
    }

    if (include_quadratic_objective)
    {
        double *quadratic_column_sum = safe_calloc((size_t)num_vars, sizeof(double));
        compute_quadratic_column_statistics(state, problem, QUADRATIC_COLUMN_SUM, quadratic_column_sum);
        for (int col = 0; col < num_vars; ++col)
            var_rescale[col] += quadratic_column_sum[col];
        free(quadratic_column_sum);
    }

    for (int i = 0; i < num_vars; ++i)
        var_rescale[i] = (var_rescale[i] < SCALING_EPSILON) ? 1.0 : sqrt(var_rescale[i]);
    for (int i = 0; i < num_cons; ++i)
        con_rescale[i] = (con_rescale[i] < SCALING_EPSILON) ? 1.0 : sqrt(con_rescale[i]);

    apply_cone_scaling_adjustment(
        var_rescale, &problem->cones, CONE_SCALING_POCK_CHAMBOLLE, use_cone_preserving_scaling);
    apply_cone_scaling_adjustment(
        con_rescale, &problem->affine_cones, CONE_SCALING_POCK_CHAMBOLLE, use_cone_preserving_scaling);

    scale_problem(state, con_rescale, var_rescale, include_quadratic_objective);
    for (int i = 0; i < num_vars; ++i)
        cum_variable_rescaling[i] *= var_rescale[i];
    for (int i = 0; i < num_cons; ++i)
        cum_constraint_rescaling[i] *= con_rescale[i];

    free(con_rescale);
    free(var_rescale);
}

static void bound_obj_rescaling(pdhg_solver_state_t *state, bool include_quadratic_objective)
{
    device_sparse_matrix_csr_t *Q = state->quadratic_objective_term->objective_sparse_matrix;
    device_sparse_matrix_csr_t *R = state->quadratic_objective_term->objective_lowrank_matrix;
    double b_norm_sq = 0.0;
    for (int i = 0; i < state->num_constraints; ++i)
    {
        if (isfinite(state->constraint_lower_bound[i]) &&
            (state->constraint_lower_bound[i] != state->constraint_upper_bound[i]))
        {
            b_norm_sq += state->constraint_lower_bound[i] * state->constraint_lower_bound[i];
        }
        if (isfinite(state->constraint_upper_bound[i]))
        {
            b_norm_sq += state->constraint_upper_bound[i] * state->constraint_upper_bound[i];
        }
    }
    for (int i = 0; i < state->num_constraints; ++i)
        b_norm_sq += state->affine_cone_offset[i] * state->affine_cone_offset[i];
    double c_norm_sq = 0.0;
    for (int i = 0; i < state->num_variables; ++i)
    {
        c_norm_sq += state->objective_vector[i] * state->objective_vector[i];
    }
    state->constraint_bound_rescaling = 1.0 / (sqrt(b_norm_sq) + 1.0);
    state->objective_vector_rescaling = 1.0 / (sqrt(c_norm_sq) + 1.0);

    for (int i = 0; i < state->num_constraints; ++i)
    {
        state->constraint_lower_bound[i] *= state->constraint_bound_rescaling;
        state->constraint_upper_bound[i] *= state->constraint_bound_rescaling;
        state->affine_cone_offset[i] *= state->constraint_bound_rescaling;
    }
    for (int i = 0; i < state->num_variables; ++i)
    {
        state->variable_lower_bound[i] *= state->constraint_bound_rescaling;
        state->variable_upper_bound[i] *= state->constraint_bound_rescaling;
        state->objective_vector[i] *= state->objective_vector_rescaling;
    }
    if (!include_quadratic_objective)
        return;

    for (int nnz_idx = 0; nnz_idx < (Q ? Q->num_nonzeros : 0); ++nnz_idx)
    {
        Q->val[nnz_idx] *= state->objective_vector_rescaling / state->constraint_bound_rescaling;
    }
    if (R && R->num_nonzeros > 0)
    {
        double R_scale_factor = sqrt(state->objective_vector_rescaling / state->constraint_bound_rescaling);
        for (int nnz_idx = 0; nnz_idx < R->num_nonzeros; ++nnz_idx)
        {
            R->val[nnz_idx] *= R_scale_factor;
        }
    }
}

void pdhcg_device_rescale_problem(const pdhg_parameters_t *params,
                                  const qp_problem_t *device_problem,
                                  pdhg_solver_state_t *state)
{
    int num_cons = state->num_constraints;
    int num_vars = state->num_variables;
    quadratic_objective_term_t *term = state->quadratic_objective_term;

    for (int i = 0; i < num_cons; ++i)
        state->constraint_rescaling[i] = 1.0;
    for (int i = 0; i < num_vars; ++i)
        state->variable_rescaling[i] = 1.0;

    bool use_cone_preserving_scaling = params->use_cone_preserving_scaling;
    bool include_quadratic_objective =
        uses_linearized_quadratic_update(params->non_diagonal_quadratic_mode, term->quad_obj_type);
    bool has_scaling = params->curtis_reid_iterations > 0 || params->l_inf_ruiz_iterations > 0 ||
        params->has_pock_chambolle_alpha || params->bound_objective_rescaling;
    if (params->curtis_reid_iterations > 0)
    {
        curtis_reid_rescaling(state,
                              device_problem,
                              params->curtis_reid_iterations,
                              include_quadratic_objective,
                              use_cone_preserving_scaling,
                              state->constraint_rescaling,
                              state->variable_rescaling);
    }
    if (params->l_inf_ruiz_iterations > 0)
    {
        ruiz_rescaling(state,
                       device_problem,
                       params->l_inf_ruiz_iterations,
                       include_quadratic_objective,
                       use_cone_preserving_scaling,
                       state->constraint_rescaling,
                       state->variable_rescaling);
    }
    if (params->has_pock_chambolle_alpha)
    {
        pock_chambolle_rescaling(state,
                                 device_problem,
                                 params->pock_chambolle_alpha,
                                 include_quadratic_objective,
                                 use_cone_preserving_scaling,
                                 state->constraint_rescaling,
                                 state->variable_rescaling);
    }
    if (params->bound_objective_rescaling)
    {
        bound_obj_rescaling(state, include_quadratic_objective);
    }
    else
    {
        state->constraint_bound_rescaling = 1.0;
        state->objective_vector_rescaling = 1.0;
    }
    if (!include_quadratic_objective && has_scaling)
        finalize_quadratic_rescaling(state);
    if (term->quad_obj_type == PDHCG_DIAG_Q)
    {
        const CsrComponent *Q = device_problem->objective_sparse_matrix;
        const int variable_start = get_n_start(state->grid_context);
        const double multiplier = state->objective_vector_rescaling / state->constraint_bound_rescaling;
#pragma omp parallel for if (num_vars > 4096)
        for (int variable = 0; variable < num_vars; ++variable)
        {
            int row = variable_start + variable;
            if (!Q || !Q->row_ptr || Q->row_ptr[row] == Q->row_ptr[row + 1])
                continue;
            double value = term->diagonal_objective_matrix[variable];
            if (has_scaling)
                value = apply_accumulated_scaling(
                    value, state->variable_rescaling[variable], state->variable_rescaling[variable], multiplier);
            term->diagonal_objective_matrix[variable] = value + SCALING_EPSILON;
        }
    }
    if (has_scaling)
    {
        /* Preserve the transpose structure and copy each scaled entry once. */
        const device_sparse_matrix_csr_t *matrices[] = {state->constraint_matrix, term->objective_lowrank_matrix};
        device_sparse_matrix_csr_t *transposes[] = {state->constraint_matrix_t, term->objective_lowrank_matrix_t};
        for (int matrix_index = 0; matrix_index < 2; ++matrix_index)
        {
            const device_sparse_matrix_csr_t *matrix = matrices[matrix_index];
            if (!matrix || matrix->num_nonzeros == 0)
                continue;
            device_sparse_matrix_csr_t *transpose = transposes[matrix_index];
            int *next = safe_malloc((size_t)matrix->num_cols * sizeof(int));
            memcpy(next, transpose->row_ptr, (size_t)matrix->num_cols * sizeof(int));
            for (int row = 0; row < matrix->num_rows; ++row)
                for (int nz = matrix->row_ptr[row]; nz < matrix->row_ptr[row + 1]; ++nz)
                    transpose->val[next[matrix->col_ind[nz]]++] = matrix->val[nz];
            free(next);
        }
    }
    pin_scaled_fixed_cone_bounds(state, device_problem);

    if (device_problem->primal_start)
    {
#pragma omp parallel for if (num_vars > 4096)
        for (int i = 0; i < num_vars; ++i)
            state->initial_primal_solution[i] =
                device_problem->primal_start[i] * state->variable_rescaling[i] * state->constraint_bound_rescaling;
    }
    if (device_problem->dual_start)
    {
#pragma omp parallel for if (num_cons > 4096)
        for (int i = 0; i < num_cons; ++i)
            state->initial_dual_solution[i] =
                device_problem->dual_start[i] * state->constraint_rescaling[i] * state->objective_vector_rescaling;
    }
}

void pdhcg_device_initialize_finite_bounds(pdhg_solver_state_t *state)
{
    const int n = state->num_variables;
    const int m = state->num_constraints;
    DEVICE_CHECK(pdhcg_device_allocate((void **)&state->variable_lower_bound_finite_val, (size_t)n * sizeof(double)));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&state->variable_upper_bound_finite_val, (size_t)n * sizeof(double)));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&state->constraint_lower_bound_finite_val, (size_t)m * sizeof(double)));
    DEVICE_CHECK(pdhcg_device_allocate((void **)&state->constraint_upper_bound_finite_val, (size_t)m * sizeof(double)));
#pragma omp parallel for if (n > 4096)
    for (int i = 0; i < n; ++i)
    {
        double lower = state->variable_lower_bound[i];
        double upper = state->variable_upper_bound[i];
        state->variable_lower_bound_finite_val[i] = isfinite(lower) ? lower : 0.0;
        state->variable_upper_bound_finite_val[i] = isfinite(upper) ? upper : 0.0;
    }
#pragma omp parallel for if (m > 4096)
    for (int i = 0; i < m; ++i)
    {
        double lower = state->constraint_lower_bound[i];
        double upper = state->constraint_upper_bound[i];
        state->constraint_lower_bound_finite_val[i] = isfinite(lower) ? lower : 0.0;
        state->constraint_upper_bound_finite_val[i] = isfinite(upper) ? upper : 0.0;
    }
}
