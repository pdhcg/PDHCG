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

#include "pdhcg.h"

#ifdef PDHCG_COMPILE_DISTRIBUTED
#include "distributed_conic.h"
#include "distributed_utils.h"
#include "preprocess.h"
#include "solver_state.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int rank_id;
static const char *case_name;
static int phase_id;
static int failures;

#define CHECK(condition)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            fprintf(stderr,                                                                                            \
                    "rank %d %s phase %d: %s:%d: %s\n",                                                                \
                    rank_id,                                                                                           \
                    case_name,                                                                                         \
                    phase_id,                                                                                          \
                    __FILE__,                                                                                          \
                    __LINE__,                                                                                          \
                    #condition);                                                                                       \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                              \
        }                                                                                                              \
    } while (0)

static void *allocate(size_t count, size_t size)
{
    void *result = calloc(count ? count : 1, size);
    CHECK(result != NULL);
    return result;
}

static double *download(const double *device, int length)
{
    double *result = allocate((size_t)length, sizeof(double));
    if (length)
        CHECK(pdhcg_device_copy(result, device, (size_t)length * sizeof(double), PDHCG_COPY_DEVICE_TO_HOST) == 0);
    return result;
}

static double *dense_matrix(const device_sparse_matrix_csr_t *matrix, int rows, int columns)
{
    double *dense = allocate((size_t)rows * columns, sizeof(double));
    if (!matrix)
        return dense;
    CHECK(matrix->num_rows == rows && matrix->num_cols == columns);
    int *row_ptr = allocate((size_t)rows + 1, sizeof(int));
    int *col_ind = allocate((size_t)matrix->num_nonzeros, sizeof(int));
    double *values = download(matrix->val, matrix->num_nonzeros);
    CHECK(pdhcg_device_copy(row_ptr, matrix->row_ptr, ((size_t)rows + 1) * sizeof(int), PDHCG_COPY_DEVICE_TO_HOST) ==
          0);
    if (matrix->num_nonzeros)
        CHECK(pdhcg_device_copy(
                  col_ind, matrix->col_ind, (size_t)matrix->num_nonzeros * sizeof(int), PDHCG_COPY_DEVICE_TO_HOST) ==
              0);
    CHECK(row_ptr[0] == 0 && row_ptr[rows] == matrix->num_nonzeros);
    for (int row = 0; row < rows; ++row)
        for (int index = row_ptr[row]; index < row_ptr[row + 1]; ++index)
        {
            CHECK(col_ind[index] >= 0 && col_ind[index] < columns);
            dense[(size_t)row * columns + col_ind[index]] += values[index];
        }
    free(row_ptr);
    free(col_ind);
    free(values);
    return dense;
}

static void compare(const char *label, int index, double actual, double expected)
{
    if (actual == expected)
        return;
    if (!isfinite(actual) || !isfinite(expected) || fabs(actual - expected) > 2e-9 * fmax(1.0, fabs(expected)))
    {
        if (failures < 12)
            fprintf(stderr,
                    "rank %d %s phase %d: %s[%d] actual %.17g expected %.17g\n",
                    rank_id,
                    case_name,
                    phase_id,
                    label,
                    index,
                    actual,
                    expected);
        ++failures;
    }
}

static void compare_slice(const char *name, const double *device, int length, const double *reference, int first)
{
    double *host = download(device, length);
    for (int index = 0; index < length; ++index)
        compare(name, index, host[index], reference[first + index]);
    free(host);
}

static void compare_matrix(const char *name,
                           const device_sparse_matrix_csr_t *matrix,
                           int rows,
                           int columns,
                           const double *reference,
                           int reference_columns,
                           int row_first,
                           int col_first,
                           int transpose)
{
    double *host = dense_matrix(matrix, rows, columns);
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < columns; ++col)
        {
            size_t expected = transpose ? (size_t)(col + row_first) * reference_columns + row + col_first
                                        : (size_t)(row + row_first) * reference_columns + col + col_first;
            compare(name, row * columns + col, host[(size_t)row * columns + col], reference[expected]);
        }
    free(host);
}

typedef struct
{
    matrix_desc_t descriptor;
    int *rows;
    int *columns;
    double *values;
} host_matrix_t;

static host_matrix_t csr_matrix(const double *dense, int rows, int columns)
{
    host_matrix_t result = {0};
    result.rows = allocate((size_t)rows + 1, sizeof(int));
    result.columns = allocate((size_t)rows * columns, sizeof(int));
    result.values = allocate((size_t)rows * columns, sizeof(double));
    int nnz = 0;
    for (int row = 0; row < rows; ++row)
    {
        result.rows[row] = nnz;
        for (int col = 0; col < columns; ++col)
            if (dense[(size_t)row * columns + col] != 0.0)
            {
                result.columns[nnz] = col;
                result.values[nnz++] = dense[(size_t)row * columns + col];
            }
    }
    result.rows[rows] = nnz;
    result.descriptor.m = rows;
    result.descriptor.n = columns;
    result.descriptor.fmt = matrix_csr;
    result.descriptor.data.csr.nnz = nnz;
    result.descriptor.data.csr.row_ptr = result.rows;
    result.descriptor.data.csr.col_ind = result.columns;
    result.descriptor.data.csr.vals = result.values;
    return result;
}

static void free_matrix(host_matrix_t *matrix)
{
    free(matrix->rows);
    free(matrix->columns);
    free(matrix->values);
}

enum problem_kind
{
    LP_REGULAR,
    LP_EMPTY_TILES,
    Q_DIAGONAL,
    Q_SPARSE,
    R_DIAGONAL_MIDDLE,
    QR_DENSE_MIDDLE,
    VARIABLE_SOC,
    VARIABLE_RSOC,
    VARIABLE_SHORT_SOC,
    AFFINE_SOC,
    LOCAL_PSD,
    NUM_PROBLEM_KINDS
};

static const char *names[] = {"lp",
                              "empty_tiles",
                              "diagonal_q",
                              "sparse_q",
                              "lowrank_diagonal_middle",
                              "q_lowrank_dense_middle",
                              "split_variable_soc",
                              "split_variable_rsoc",
                              "split_endpoints_empty_ranks",
                              "split_affine_soc",
                              "local_psd_congruence"};

static qp_problem_t *make_problem(enum problem_kind kind)
{
    const int n = 12;
    int affine_rows = kind == AFFINE_SOC ? 10 : kind == LOCAL_PSD ? 6 : 0;
    int scalar_rows = 12 - affine_rows;
    double a[144] = {0}, f[144] = {0}, q[144] = {0}, r[36] = {0}, d[9] = {0};
    double objective[12], lower[12], upper[12], var_lower[12], var_upper[12], offset[12], primal[12], dual[12];
    for (int row = 0; row < n; ++row)
    {
        objective[row] = ldexp(row % 2 ? -1.0 : 1.0, row % 7 - 3);
        lower[row] = row % 3 ? -INFINITY : -ldexp(1.0, row % 5 - 2);
        upper[row] = row % 4 ? ldexp(1.0, row % 6 - 2) : INFINITY;
        var_lower[row] = row % 3 ? -INFINITY : -2.0;
        var_upper[row] = row % 4 ? INFINITY : 4.0;
        offset[row] = ldexp(row % 2 ? -1.0 : 1.0, row % 5 - 3);
        primal[row] = 0.0625;
        dual[row] = affine_rows ? 0.0 : 0.125 * (row % 3 - 1);
        for (int col = 0; col < n; ++col)
        {
            if (col == row || col == (row + 5) % n)
            {
                if (kind != LP_EMPTY_TILES || (row < 6 && col < 6))
                    a[row * n + col] = ldexp((row + col) % 2 ? -1.0 : 1.0, (row * 3 + col) % 13 - 6);
                f[row * n + col] = ldexp((row + col) % 2 ? 1.0 : -1.0, (row + col * 2) % 11 - 5);
            }
        }
    }
    int has_q = kind == Q_DIAGONAL || kind == Q_SPARSE || kind == QR_DENSE_MIDDLE;
    int has_r = kind == R_DIAGONAL_MIDDLE || kind == QR_DENSE_MIDDLE;
    if (has_q)
        for (int row = 0; row < n; ++row)
        {
            if (kind == Q_DIAGONAL)
                q[row * n + row] = row == 5 ? 0.0 : ldexp(1.0, row % 6 - 2);
            else
            {
                q[row * n + row] = 4.0 + row % 3;
                q[row * n + (row + 4) % n] = -0.5;
                q[row * n + (row + 8) % n] = -0.5;
            }
        }
    if (has_r)
    {
        /* The right column tile has no R entries, despite a global low-rank term. */
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 5; ++col)
                if ((row + col) % 2)
                    r[row * n + col] = ldexp(1.0, row + col - 3);
        d[0] = 2.0;
        d[4] = 0.5;
        d[8] = 4.0;
        if (kind == QR_DENSE_MIDDLE)
        {
            d[1] = d[3] = 0.125;
            d[5] = d[7] = -0.0625;
        }
    }
    cone_spec_t variable_cone = {0}, affine_cone = {0};
    int variable_cones = 0, affine_cones = affine_rows ? 1 : 0;
    if (kind == VARIABLE_SOC || kind == VARIABLE_RSOC)
    {
        variable_cones = 1;
        variable_cone.type = kind == VARIABLE_SOC ? CONE_STANDARD_SOC : CONE_ROTATED_SOC;
        variable_cone.start_idx = 1;
        variable_cone.v_dim = 8;
        for (int col = 1; col < 11; ++col)
            var_lower[col] = -INFINITY, var_upper[col] = INFINITY;
        primal[9] = kind == VARIABLE_SOC ? 0.5 : 1.0;
        primal[10] = 2.0;
    }
    if (kind == VARIABLE_SHORT_SOC)
    {
        variable_cones = 1;
        variable_cone.type = CONE_STANDARD_SOC;
        variable_cone.v_dim = 2;
        for (int col = 0; col < 4; ++col)
            var_lower[col] = -INFINITY, var_upper[col] = INFINITY;
    }
    if (kind == AFFINE_SOC)
    {
        affine_cone.type = CONE_STANDARD_SOC;
        affine_cone.v_dim = 8;
    }
    if (kind == LOCAL_PSD)
    {
        variable_cones = 1;
        variable_cone.type = CONE_PSD;
        variable_cone.start_idx = 2;
        variable_cone.v_dim = 3;
        affine_cone.type = CONE_PSD;
        affine_cone.v_dim = 3;
        for (int col = 2; col < 8; ++col)
        {
            var_lower[col] = -INFINITY;
            var_upper[col] = INFINITY;
            primal[col] = col == 2 || col == 5 || col == 7 ? 2.0 : 0.0;
        }
    }
    host_matrix_t A = csr_matrix(a, scalar_rows, n), F = csr_matrix(f, affine_rows, n);
    host_matrix_t Q = csr_matrix(q, n, n), R = csr_matrix(r, 3, n), D = csr_matrix(d, 3, 3);
    qp_problem_t *problem = create_qp_problem(objective,
                                              has_q ? &Q.descriptor : NULL,
                                              has_r ? &R.descriptor : NULL,
                                              has_r ? &D.descriptor : NULL,
                                              &A.descriptor,
                                              lower,
                                              upper,
                                              var_lower,
                                              var_upper,
                                              NULL,
                                              variable_cones,
                                              variable_cones ? &variable_cone : NULL,
                                              affine_rows ? &F.descriptor : NULL,
                                              affine_rows ? offset : NULL,
                                              affine_cones,
                                              affine_cones ? &affine_cone : NULL);
    free_matrix(&A);
    free_matrix(&F);
    free_matrix(&Q);
    free_matrix(&R);
    free_matrix(&D);
    CHECK(problem != NULL);
    if (kind == VARIABLE_SOC || kind == VARIABLE_RSOC)
    {
        CHECK(set_cone_fixed(problem, 0, 8, primal[9]) == 0);
        if (kind == VARIABLE_RSOC)
            CHECK(set_cone_fixed(problem, 0, 9, primal[10]) == 0);
    }
    set_start_values(problem, primal, dual);
    problem->objective_constant = 0.375;
    return problem;
}

typedef struct
{
    int n, m, rank, middle_type;
    quad_obj_type_t quad_type;
    double bound, objective, constant;
    double *A, *Q, *R, *diagonal, *middle;
    double *variable_scale, *constraint_scale, *c, *var_lower, *var_upper, *con_lower, *con_upper, *offset;
    double *initial_primal, *initial_dual;
} snapshot_t;

static snapshot_t snapshot(pdhg_solver_state_t *state)
{
    snapshot_t out = {0};
    out.n = state->num_variables;
    out.m = state->num_constraints;
    quadratic_objective_term_t *term = state->quadratic_objective_term;
    out.rank = term->num_rank_lowrank_obj;
    out.quad_type = term->quad_obj_type;
    out.middle_type = term->lowrank_middle_type;
    out.bound = state->constraint_bound_rescaling;
    out.objective = state->objective_vector_rescaling;
    out.constant = state->objective_constant;
    out.A = dense_matrix(state->constraint_matrix, out.m, out.n);
    out.Q = dense_matrix(term->objective_sparse_matrix, out.n, out.n);
    out.R = dense_matrix(term->objective_lowrank_matrix, out.rank, out.n);
    if (out.quad_type == PDHCG_DIAG_Q)
        out.diagonal = download(term->diagonal_objective_matrix, out.n);
    int middle_length = out.middle_type == PDHCG_D_DIAG ? out.rank
        : out.middle_type == PDHCG_D_DENSE              ? out.rank * out.rank
                                                        : 0;
    out.middle = download(out.middle_type == PDHCG_D_DIAG ? term->d_middle_diag : term->d_middle_dense, middle_length);
    out.variable_scale = download(state->variable_rescaling, out.n);
    out.constraint_scale = download(state->constraint_rescaling, out.m);
    out.c = download(state->objective_vector, out.n);
    out.var_lower = download(state->variable_lower_bound, out.n);
    out.var_upper = download(state->variable_upper_bound, out.n);
    out.con_lower = download(state->constraint_lower_bound, out.m);
    out.con_upper = download(state->constraint_upper_bound, out.m);
    out.offset = download(state->affine_cone_offset, out.m);
    out.initial_primal = download(state->initial_primal_solution, out.n);
    out.initial_dual = download(state->initial_dual_solution, out.m);
    return out;
}

static void free_snapshot(snapshot_t *s)
{
    free(s->A);
    free(s->Q);
    free(s->R);
    free(s->diagonal);
    free(s->middle);
    free(s->variable_scale);
    free(s->constraint_scale);
    free(s->c);
    free(s->var_lower);
    free(s->var_upper);
    free(s->con_lower);
    free(s->con_upper);
    free(s->offset);
    free(s->initial_primal);
    free(s->initial_dual);
}

static void compare_state(pdhg_solver_state_t *state, const snapshot_t *reference, const grid_context_t *grid)
{
    int n = state->num_variables, m = state->num_constraints;
    int n_start = grid->variable_cuts[grid->coords[1]], m_start = grid->constraint_cuts[grid->coords[0]];
    quadratic_objective_term_t *term = state->quadratic_objective_term;
    CHECK(n == grid->variable_cuts[grid->coords[1] + 1] - n_start);
    CHECK(m == grid->constraint_cuts[grid->coords[0] + 1] - m_start);
    CHECK(term->quad_obj_type == reference->quad_type);
    CHECK(term->num_rank_lowrank_obj == reference->rank);
    CHECK(term->lowrank_middle_type == reference->middle_type);
    compare("bound_scale", 0, state->constraint_bound_rescaling, reference->bound);
    compare("objective_scale", 0, state->objective_vector_rescaling, reference->objective);
    compare("objective_constant", 0, state->objective_constant, reference->constant);
    compare_slice("variable_scale", state->variable_rescaling, n, reference->variable_scale, n_start);
    compare_slice("constraint_scale", state->constraint_rescaling, m, reference->constraint_scale, m_start);
    compare_slice("objective", state->objective_vector, n, reference->c, n_start);
    compare_slice("variable_lower", state->variable_lower_bound, n, reference->var_lower, n_start);
    compare_slice("variable_upper", state->variable_upper_bound, n, reference->var_upper, n_start);
    compare_slice("constraint_lower", state->constraint_lower_bound, m, reference->con_lower, m_start);
    compare_slice("constraint_upper", state->constraint_upper_bound, m, reference->con_upper, m_start);
    compare_slice("affine_offset", state->affine_cone_offset, m, reference->offset, m_start);
    compare_slice("initial_primal", state->initial_primal_solution, n, reference->initial_primal, n_start);
    compare_slice("initial_dual", state->initial_dual_solution, m, reference->initial_dual, m_start);
    compare_matrix("A", state->constraint_matrix, m, n, reference->A, reference->n, m_start, n_start, 0);
    compare_matrix("At", state->constraint_matrix_t, n, m, reference->A, reference->n, m_start, n_start, 1);
    if (reference->quad_type == PDHCG_DIAG_Q)
        compare_slice("diagonal_q", term->diagonal_objective_matrix, n, reference->diagonal, n_start);
    else
        compare_matrix("Q", term->objective_sparse_matrix, reference->n, n, reference->Q, reference->n, 0, n_start, 0);
    compare_matrix("R", term->objective_lowrank_matrix, reference->rank, n, reference->R, reference->n, 0, n_start, 0);
    compare_matrix(
        "Rt", term->objective_lowrank_matrix_t, n, reference->rank, reference->R, reference->n, 0, n_start, 1);
    int middle_length = reference->middle_type == PDHCG_D_DIAG ? reference->rank
        : reference->middle_type == PDHCG_D_DENSE              ? reference->rank * reference->rank
                                                               : 0;
    compare_slice("D",
                  reference->middle_type == PDHCG_D_DIAG ? term->d_middle_diag : term->d_middle_dense,
                  middle_length,
                  reference->middle,
                  0);
}

static void configure(pdhg_parameters_t *params, enum problem_kind kind, int phase)
{
    set_default_parameters(params);
    params->verbose = 0;
    params->sv_max_iter = 2;
    params->partition_method = UNIFORM_PARTITION;
    params->permute_method = NO_PERMUTATION;
    params->presolve = false;
    params->curtis_reid_iterations = phase == 3 || phase >= 4 ? 3 : 0;
    params->l_inf_ruiz_iterations = phase == 1 || phase >= 4 ? 4 : 0;
    params->has_pock_chambolle_alpha = phase == 2 || phase >= 4;
    params->pock_chambolle_alpha = phase == 2 ? 1.25 : 0.75;
    params->bound_objective_rescaling = phase >= 4;
    params->non_diagonal_quadratic_mode = phase == 5 ? NON_DIAGONAL_QUADRATIC_LINEARIZED : NON_DIAGONAL_QUADRATIC_INNER;
    params->use_cone_preserving_scaling = kind != LOCAL_PSD;
}

static void test_grid(int rows, int columns)
{
    grid_context_t grid = initialize_parallel_context(rows, columns);
    int cases = 0;
    for (int kind = 0; kind < NUM_PROBLEM_KINDS; ++kind)
        for (int phase = 0; phase < 6; ++phase)
        {
            if (columns > 2 && kind != VARIABLE_SHORT_SOC)
                continue;
            if (phase == 5 && (kind < Q_DIAGONAL || kind > QR_DENSE_MIDDLE))
                continue;
            case_name = names[kind];
            phase_id = phase;
            pdhg_parameters_t params;
            configure(&params, (enum problem_kind)kind, phase);
            qp_problem_t *global = make_problem((enum problem_kind)kind);
            qp_problem_t *device_full = convert_problem_to_device(global, NULL);
            pdhg_solver_state_t *full = initialize_solver_state(&params, global, device_full, NULL);
            free_device_problem(device_full);
            CHECK(full != NULL);
            snapshot_t reference = snapshot(full);
            pdhg_solver_state_free(full);
            MPI_Barrier(MPI_COMM_WORLD);
            configure_partition_metadata(global, &grid, &params);
            qp_problem_t *local = partition_qp_problem(global, &grid, params.partition_method, NULL, NULL);
            CHECK(local != NULL);
            if ((kind == VARIABLE_SOC || kind == VARIABLE_RSOC) && columns > 1)
                CHECK(grid.split_cones.num_cones == 1);
            if (kind == VARIABLE_SHORT_SOC && columns == 4)
            {
                CHECK(grid.split_cones.num_cones == 1);
                CHECK(grid.split_cones.local_count[0] == (grid.coords[1] == 0 ? 3 : grid.coords[1] == 1 ? 1 : 0));
            }
            if (kind == AFFINE_SOC && rows > 1)
                CHECK(grid.split_affine_cones.num_cones == 1);
            if (kind == LP_EMPTY_TILES && (grid.coords[0] > 0 || grid.coords[1] > 0))
                CHECK(local->constraint_matrix_num_nonzeros == 0);
            if ((kind == R_DIAGONAL_MIDDLE || kind == QR_DENSE_MIDDLE) && grid.coords[1] > 0)
                CHECK(local->objective_lowrank_matrix_num_nonzeros == 0);
            qp_problem_t *device_state = convert_problem_to_device(local, &grid);
            pdhg_solver_state_t *state = initialize_solver_state(&params, local, device_state, &grid);
            free_device_problem(device_state);
            CHECK(state != NULL);
            compare_state(state, &reference, &grid);
            pdhg_solver_state_free(state);
            qp_problem_free(local);
            qp_problem_free(global);
            free_snapshot(&reference);
            MPI_Barrier(MPI_COMM_WORLD);
            ++cases;
        }
    int total_failures = 0;
    MPI_Allreduce(&failures, &total_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank_id == 0)
        printf("distributed scaling grid=%dx%d cases=%d failures=%d\n", rows, columns, cases, total_failures);
    destroy_parallel_context(&grid);
}

static void reject_heterogeneous_scaling(bool affine)
{
    grid_context_t grid = initialize_parallel_context(affine ? 2 : 1, affine ? 1 : 2);
    enum problem_kind kind = affine ? AFFINE_SOC : VARIABLE_SOC;
    pdhg_parameters_t params;
    configure(&params, kind, 4);
    qp_problem_t *global = make_problem(kind);
    configure_partition_metadata(global, &grid, &params);
    qp_problem_t *local = partition_qp_problem(global, &grid, params.partition_method, NULL, NULL);
    qp_problem_t *device = convert_problem_to_device(local, &grid);
    pdhg_solver_state_t *state = initialize_solver_state(&params, local, device, &grid);
    CHECK(state != NULL);
    free_device_problem(device);
    free_split_cones(state);
    const distributed_cone_partition_t *partition = affine ? &grid.split_affine_cones : &grid.split_cones;
    CHECK(partition->num_cones == 1);
    if (grid.coords[affine ? 0 : 1] == 0)
    {
        int count = partition->local_count[0];
        CHECK(count > 0);
        double *segment =
            (affine ? state->constraint_rescaling : state->variable_rescaling) + partition->local_start[0];
        double *values = download(segment, count);
        double changed = 2.0 * values[0];
        /* Each rank remains locally uniform; only the global range can reject it. */
        for (int i = 0; i < count; ++i)
            values[i] = changed;
        CHECK(pdhcg_device_copy(segment, values, (size_t)count * sizeof(double), PDHCG_COPY_HOST_TO_DEVICE) == 0);
        free(values);
    }
    initialize_split_cones(state);
    CHECK(0); /* The call must abort with the heterogeneous-scaling diagnostic. */
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 1 && size != 2 && size != 4)
    {
        if (rank_id == 0)
            fprintf(stderr, "test_distributed_scaling requires 1, 2, or 4 MPI ranks\n");
        MPI_Finalize();
        return 77;
    }
    case_name = "initialize";
    if (argc > 1)
    {
        case_name = argv[1];
        CHECK(size == 2 && argc == 2);
        CHECK(strcmp(argv[1], "--heterogeneous-variable") == 0 || strcmp(argv[1], "--heterogeneous-affine") == 0);
        reject_heterogeneous_scaling(strcmp(argv[1], "--heterogeneous-affine") == 0);
    }
    if (size == 1)
        test_grid(1, 1);
    else if (size == 2)
    {
        test_grid(1, 2);
        test_grid(2, 1);
    }
    else
    {
        test_grid(2, 2);
        test_grid(1, 4);
    }
    int total_failures = 0;
    MPI_Allreduce(&failures, &total_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Finalize();
    return total_failures ? 1 : 0;
}
#else
int main(void)
{
    return 0;
}
#endif
