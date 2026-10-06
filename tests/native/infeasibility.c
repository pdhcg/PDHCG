/* Copyright 2026 Hongpei Li. Licensed under the Apache License, Version 2.0. */
#include "infeasibility.h"
#include "device_general_op.h"
#include "preconditioner.h"
#include "solver_state.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            fprintf(stderr, "certificate check failed at line %d: %s\n", __LINE__, #condition);                        \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

typedef struct
{
    double rows[2], columns[2], bounds, objective;
} scaling_t;

typedef struct
{
    double a[4], c[2], lower[2], upper[2], h[4];
    int quadratic;
} problem_t;

static void near(double actual, double expected)
{
    CHECK(isfinite(actual));
    if (fabs(actual - expected) > 1e-11 * fmax(1.0, fabs(expected)))
    {
        fprintf(stderr, "expected %.17g, got %.17g\n", expected, actual);
        abort();
    }
}

static void upload(double *destination, const double values[2])
{
    CHECK(pdhcg_device_copy(destination, values, 2 * sizeof(double), PDHCG_COPY_HOST_TO_DEVICE) == 0);
}

/* Construct the scaled problem explicitly, independently of the solver's
   preconditioner. x_scaled = bounds * columns * x, and
   y_scaled = objective * rows * y. */
static qp_problem_t *make_problem(const problem_t *data, const scaling_t *scaling)
{
    const int row_ptr[] = {0, 2, 4}, col_ind[] = {0, 1, 0, 1};
    const double free_lower[] = {-INFINITY, -INFINITY};
    const double free_upper[] = {INFINITY, INFINITY};
    double a[4], h[4], c[2], lower[2], upper[2];
    for (int i = 0; i < 2; ++i)
    {
        c[i] = data->c[i] * scaling->objective / scaling->columns[i];
        lower[i] = data->lower[i] * scaling->bounds / scaling->rows[i];
        upper[i] = data->upper[i] * scaling->bounds / scaling->rows[i];
        for (int j = 0; j < 2; ++j)
        {
            a[2 * i + j] = data->a[2 * i + j] / (scaling->rows[i] * scaling->columns[j]);
            h[2 * i + j] =
                data->h[2 * i + j] * scaling->objective / (scaling->bounds * scaling->columns[i] * scaling->columns[j]);
        }
    }
    matrix_desc_t A = {0}, H = {0};
    A.m = A.n = H.m = H.n = 2;
    A.fmt = H.fmt = matrix_csr;
    A.data.csr.nnz = H.data.csr.nnz = 4;
    A.data.csr.row_ptr = H.data.csr.row_ptr = row_ptr;
    A.data.csr.col_ind = H.data.csr.col_ind = col_ind;
    A.data.csr.vals = a;
    H.data.csr.vals = h;
    qp_problem_t *problem = create_qp_problem(c,
                                              data->quadratic ? &H : NULL,
                                              NULL,
                                              NULL,
                                              &A,
                                              lower,
                                              upper,
                                              free_lower,
                                              free_upper,
                                              NULL,
                                              0,
                                              NULL,
                                              NULL,
                                              NULL,
                                              0,
                                              NULL);
    CHECK(problem != NULL);
    return problem;
}

static pdhg_solver_state_t *make_state(const problem_t *data, const scaling_t *scaling)
{
    const scaling_t identity = {{1, 1}, {1, 1}, 1, 1};
    qp_problem_t *original = make_problem(data, &identity);
    qp_problem_t *scaled = make_problem(data, scaling);
    double rows[] = {scaling->rows[0], scaling->rows[1]};
    double columns[] = {scaling->columns[0], scaling->columns[1]};
    rescale_info_t info = {0};
    info.scaled_problem = scaled;
    info.processed_problem = preprocess_qp_problem(scaled);
    CHECK(info.processed_problem != NULL);
    info.con_rescale = rows;
    info.var_rescale = columns;
    info.con_bound_rescale = scaling->bounds;
    info.obj_vec_rescale = scaling->objective;
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = 0;
    pdhg_solver_state_t *state = initialize_solver_state(&params, original, &info, NULL);
    CHECK(state != NULL);
    free_processed_qp_problem(info.processed_problem);
    qp_problem_free(scaled);
    qp_problem_free(original);
    return state;
}

static void evaluate(pdhg_solver_state_t *state, const scaling_t *scaling, const double primal[2], const double dual[2])
{
    const double sentinel[] = {13.25, -7.5};
    double scaled_primal[2], scaled_dual[2], after[2];
    for (int i = 0; i < 2; ++i)
    {
        scaled_primal[i] = primal[i] * scaling->bounds * scaling->columns[i];
        scaled_dual[i] = dual[i] * scaling->objective * scaling->rows[i];
    }
    upload(state->delta_primal_solution, scaled_primal);
    upload(state->delta_dual_solution, scaled_dual);
    upload(state->dual_slack, sentinel);
    compute_infeasibility_information(state);
    CHECK(pdhcg_device_copy(after, state->dual_slack, sizeof(after), PDHCG_COPY_DEVICE_TO_HOST) == 0);
    CHECK(after[0] == sentinel[0] && after[1] == sentinel[1]);
}

static void check_scaled_rays(const scaling_t *scaling, int quadratic)
{
    problem_t data = {{1, 0, 0, 2}, {2, -1}, {1, -INFINITY}, {INFINITY, 3}, {0, 0, 0, 0}, quadratic};
    if (quadratic == 1)
    {
        data.h[0] = 3;
        data.h[3] = 4;
    }
    else if (quadratic == 2)
    {
        data.h[0] = 3;
        data.h[1] = data.h[2] = -1;
        data.h[3] = 2;
    }
    pdhg_solver_state_t *state = make_state(&data, scaling);
    const double primal[] = {-2, 1}, dual[] = {1, -0.25}, zero[] = {0, 0};
    evaluate(state, scaling, primal, dual);

    /* In original coordinates, -c'd = 5, the row violation is 2, and
       ||H d||_inf is respectively 0, 6 or 7. For y, b'y = 1/4 and
       ||A'y||_inf = 1. These ratios must survive arbitrary rescaling.
       All three problems are feasible and have a finite optimum. */
    const double expected_primal[] = {2.0 / 5, 6.0 / 5, 7.0 / 5};
    CHECK(state->primal_ray_linear_objective < 0);
    CHECK(state->dual_ray_objective > 0);
    near(state->max_primal_ray_infeasibility / -state->primal_ray_linear_objective, expected_primal[quadratic]);
    near(state->max_dual_ray_infeasibility / state->dual_ray_objective, 4);
    CHECK(check_infeasibility_criteria(state, 1e-10) == TERMINATION_REASON_UNSPECIFIED);

    /* A zero direction cannot certify either status, even after a previous
       nonzero candidate populated the statistics. */
    evaluate(state, scaling, zero, zero);
    near(state->primal_ray_linear_objective, 0);
    near(state->dual_ray_objective, 0);
    near(state->max_primal_ray_infeasibility, 0);
    near(state->max_dual_ray_infeasibility, 0);
    CHECK(check_infeasibility_criteria(state, 1e-10) == TERMINATION_REASON_UNSPECIFIED);
    pdhg_solver_state_free(state);
}

static void check_exact_primal_certificate(const scaling_t *scaling)
{
    /* min -2 x_0 - x_1, x_0 >= 1, 2 x_1 <= 3 has recession direction (2, -1). */
    const problem_t data = {{1, 0, 0, 2}, {-2, -1}, {1, -INFINITY}, {INFINITY, 3}, {0, 0, 0, 0}, 0};
    const double primal[] = {2, -1}, dual[] = {0, 0};
    pdhg_solver_state_t *state = make_state(&data, scaling);
    evaluate(state, scaling, primal, dual);
    CHECK(state->primal_ray_linear_objective < 0);
    near(state->max_primal_ray_infeasibility, 0);
    CHECK(check_infeasibility_criteria(state, 1e-12) == TERMINATION_REASON_DUAL_INFEASIBLE);
    pdhg_solver_state_free(state);
}

static void check_exact_dual_certificate(const scaling_t *scaling)
{
    /* x_0 >= 1 and x_0 <= 0, with the exact certificate y = (1, -1). */
    const problem_t data = {{1, 0, 1, 0}, {0, 0}, {1, -INFINITY}, {INFINITY, 0}, {0, 0, 0, 0}, 0};
    const double primal[] = {0, 0}, dual[] = {1, -1};
    pdhg_solver_state_t *state = make_state(&data, scaling);
    evaluate(state, scaling, primal, dual);
    CHECK(state->dual_ray_objective > 0);
    near(state->max_dual_ray_infeasibility / state->dual_ray_objective, 0);
    CHECK(check_infeasibility_criteria(state, 1e-12) == TERMINATION_REASON_PRIMAL_INFEASIBLE);
    pdhg_solver_state_free(state);
}

static void check_invalid_certificates(void)
{
    pdhg_solver_state_t state = {0};
    const double invalid[] = {NAN, INFINITY};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i)
    {
        state.dual_ray_objective = invalid[i];
        CHECK(check_infeasibility_criteria(&state, 1e-10) == TERMINATION_REASON_UNSPECIFIED);
        state.dual_ray_objective = 1;
        state.max_dual_ray_infeasibility = invalid[i];
        CHECK(check_infeasibility_criteria(&state, 1e-10) == TERMINATION_REASON_UNSPECIFIED);
        state.dual_ray_objective = 0;
        state.max_dual_ray_infeasibility = 0;
        state.primal_ray_linear_objective = -invalid[i];
        CHECK(check_infeasibility_criteria(&state, 1e-10) == TERMINATION_REASON_UNSPECIFIED);
        state.primal_ray_linear_objective = -1;
        state.max_primal_ray_infeasibility = invalid[i];
        CHECK(check_infeasibility_criteria(&state, 1e-10) == TERMINATION_REASON_UNSPECIFIED);
        state.primal_ray_linear_objective = 0;
        state.max_primal_ray_infeasibility = 0;
    }
}

int main(void)
{
    CHECK(pdhcg_device_initialize() == 0);
    const scaling_t scalings[] = {
        {{1, 1}, {1, 1}, 1, 1},
        {{1, 1}, {1, 1}, 0x1p-40, 1},
        {{1, 1}, {1, 1}, 1, 0x1p-36},
        {{0.5, 16}, {8, 0.25}, 1, 1},
        {{0.5, 16}, {8, 0.25}, 0x1p-40, 0x1p-36},
    };
    for (size_t i = 0; i < sizeof(scalings) / sizeof(scalings[0]); ++i)
    {
        for (int quadratic = 0; quadratic <= 2; ++quadratic)
            check_scaled_rays(&scalings[i], quadratic);
        check_exact_primal_certificate(&scalings[i]);
        check_exact_dual_certificate(&scalings[i]);
    }
    check_invalid_certificates();
    CHECK(pdhcg_device_synchronize() == 0);
    puts("Infeasibility certificates preserve original-coordinate accuracy and solver slack");
    return 0;
}
