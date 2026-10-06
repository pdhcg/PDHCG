/* Copyright 2026 Hongpei Li. Licensed under the Apache License, Version 2.0. */
#include "device_general_op.h"
#include "infeasibility.h"
#include "preconditioner.h"
#include "solver_state.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            fprintf(stderr, "conic certificate check failed at line %d: %s\n", __LINE__, #condition);                  \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

typedef struct
{
    cone_type_t kind;
    double dual[3], ray[3], tangent[3], nondual[3];
} geometry_t;

typedef struct
{
    int n, m, variable_cone, affine_cone, quadratic;
    cone_type_t kind;
    double a[12], lower[4], upper[4], f[9], offset[3], c[3], anchor[3];
    char fixed[3];
} problem_t;

static matrix_desc_t csr(int m, int n, const double *values, int *rows, int *columns)
{
    for (int i = 0; i <= m; ++i)
        rows[i] = i * n;
    for (int i = 0; i < m * n; ++i)
        columns[i] = i % n;
    matrix_desc_t matrix = {0};
    matrix.m = m;
    matrix.n = n;
    matrix.fmt = matrix_csr;
    matrix.data.csr.nnz = m * n;
    matrix.data.csr.row_ptr = rows;
    matrix.data.csr.col_ind = columns;
    matrix.data.csr.vals = values;
    return matrix;
}

static qp_problem_t *make_problem(const problem_t *data)
{
    int a_rows[5], a_columns[12], f_rows[4], f_columns[9], h_rows[4], h_columns[9];
    const double identity[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    matrix_desc_t A = csr(data->m, data->n, data->a, a_rows, a_columns);
    matrix_desc_t F = csr(3, data->n, data->f, f_rows, f_columns);
    matrix_desc_t H = csr(data->n, data->n, identity, h_rows, h_columns);
    cone_spec_t cone = {0};
    cone.type = data->kind;
    cone.v_dim = data->kind == CONE_PSD ? 2 : 1;
    cone.power_alpha = 0.25;
    cone.is_fixed = data->fixed[0] || data->fixed[1] || data->fixed[2] ? data->fixed : NULL;
    qp_problem_t *problem = create_qp_problem(data->c,
                                              data->quadratic ? &H : NULL,
                                              NULL,
                                              NULL,
                                              &A,
                                              data->lower,
                                              data->upper,
                                              NULL,
                                              NULL,
                                              NULL,
                                              data->variable_cone,
                                              data->variable_cone ? &cone : NULL,
                                              data->affine_cone ? &F : NULL,
                                              data->affine_cone ? data->offset : NULL,
                                              data->affine_cone,
                                              data->affine_cone ? &cone : NULL);
    CHECK(problem != NULL);
    set_start_values(problem, data->anchor, NULL);
    return problem;
}

static void upload(double *destination, const double *source, int count)
{
    if (count)
        CHECK(pdhcg_device_copy(destination, source, (size_t)count * sizeof(double), PDHCG_COPY_HOST_TO_DEVICE) == 0);
}

/* Expected certificates are derived in original coordinates. The actual
   preconditioner is used only to express those candidates in solver units. */
static double
evaluate(const problem_t *data, int scaling, const double *primal, const double *dual, termination_reason_t expected)
{
    qp_problem_t *problem = make_problem(data);
    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = 0;
    if (scaling == 0)
    {
        params.l_inf_ruiz_iterations = 0;
        params.has_pock_chambolle_alpha = false;
        params.bound_objective_rescaling = false;
    }
    else if (scaling == 2)
        params.use_cone_preserving_scaling = false;
    rescale_info_t *info = rescale_problem(&params, problem);
    CHECK(info != NULL);
    pdhg_solver_state_t *state = initialize_solver_state(&params, problem, info, NULL);
    CHECK(state != NULL);
    double dx[3], dy[4], anchor[3], sentinel[3], after[3];
    for (int i = 0; i < data->n; ++i)
    {
        dx[i] = primal[i] * info->var_rescale[i] * info->con_bound_rescale;
        anchor[i] = data->anchor[i] * info->var_rescale[i] * info->con_bound_rescale;
        sentinel[i] = 17.25 + i;
    }
    for (int i = 0; i < problem->num_constraints; ++i)
        dy[i] = dual[i] * info->con_rescale[i] * info->obj_vec_rescale;
    upload(state->delta_primal_solution, dx, data->n);
    upload(state->delta_dual_solution, dy, problem->num_constraints);
    upload(state->pdhg_primal_solution, anchor, data->n);
    upload(state->dual_slack, sentinel, data->n);
    compute_infeasibility_information(state);
    termination_reason_t actual = check_infeasibility_criteria(state, 1e-7);
    if (actual != expected)
        fprintf(stderr,
                "kind=%d scaling=%d variable=%d affine=%d expected=%d got=%d; "
                "primal=(%.17g,%.17g) dual=(%.17g,%.17g)\n",
                data->kind,
                scaling,
                data->variable_cone,
                data->affine_cone,
                expected,
                actual,
                state->primal_ray_linear_objective,
                state->max_primal_ray_infeasibility,
                state->dual_ray_objective,
                state->max_dual_ray_infeasibility);
    CHECK(actual == expected);
    CHECK(pdhcg_device_copy(after, state->dual_slack, (size_t)data->n * sizeof(double), PDHCG_COPY_DEVICE_TO_HOST) ==
          0);
    for (int i = 0; i < data->n; ++i)
        CHECK(after[i] == sentinel[i]);
    const double objective = state->dual_ray_objective;

    if (expected == TERMINATION_REASON_PRIMAL_INFEASIBLE || expected == TERMINATION_REASON_DUAL_INFEASIBLE)
    {
        /* With no usable difference vector, the iterate must independently
           pass the complete certificate test. Clear the previous statistics
           so an old successful result cannot satisfy this check. */
        const double zero[] = {0, 0, 0, 0};
        upload(state->delta_primal_solution, zero, data->n);
        upload(state->delta_dual_solution, zero, problem->num_constraints);
        upload(state->pdhg_primal_solution, expected == TERMINATION_REASON_DUAL_INFEASIBLE ? dx : anchor, data->n);
        upload(state->pdhg_dual_solution,
               expected == TERMINATION_REASON_PRIMAL_INFEASIBLE ? dy : zero,
               problem->num_constraints);
        state->primal_ray_linear_objective = state->max_primal_ray_infeasibility = 0;
        state->dual_ray_objective = state->max_dual_ray_infeasibility = 0;
        compute_infeasibility_information(state);
        CHECK(check_infeasibility_criteria(state, 1e-7) == expected);
        CHECK(pdhcg_device_copy(
                  after, state->dual_slack, (size_t)data->n * sizeof(double), PDHCG_COPY_DEVICE_TO_HOST) == 0);
        for (int i = 0; i < data->n; ++i)
            CHECK(after[i] == sentinel[i]);
    }
    pdhg_solver_state_free(state);
    rescale_info_free(info);
    qp_problem_free(problem);
    return objective;
}

static void check_variable_certificates(const geometry_t *geometry, int scaling)
{
    problem_t data = {0};
    data.n = 3;
    data.m = 1;
    data.variable_cone = 1;
    data.kind = geometry->kind;
    data.lower[0] = -INFINITY;
    data.upper[0] = -1;
    const double zero[] = {0, 0, 0, 0}, dual[] = {-1};

    /* For w in K*, w'x <= -1 contradicts x in K, with certificate gap 1. */
    memcpy(data.a, geometry->dual, 3 * sizeof(double));
    evaluate(&data, scaling, zero, dual, TERMINATION_REASON_PRIMAL_INFEASIBLE);

    /* q is outside K*: each problem below has an explicit feasible point.
       SOC: (-1,0,1), RSOC: (-1,1,1), PSD: svec([[1,-1],[-1,1]]),
       EXP: (-1,0,0), POWER: (1,1,-1). A free-variable box test alone
       or a mistaken self-dual projection must not certify infeasibility. */
    memcpy(data.a, geometry->nondual, 3 * sizeof(double));
    evaluate(&data, scaling, zero, dual, TERMINATION_REASON_UNSPECIFIED);

    data.m = 0;
    for (int i = 0; i < 3; ++i)
        data.c[i] = -geometry->ray[i];
    /* d in K, c'd=-||d||^2<0: a valid recession certificate. */
    evaluate(&data, scaling, geometry->ray, zero, TERMINATION_REASON_DUAL_INFEASIBLE);
    /* H=I makes the same linear decrease insufficient: H*d is nonzero,
       and x=d is the exact finite minimizer of this cone-constrained QP. */
    data.quadratic = 1;
    evaluate(&data, scaling, geometry->ray, zero, TERMINATION_REASON_UNSPECIFIED);
}

static void check_affine_certificates(const geometry_t *geometry, int scaling)
{
    problem_t data = {0};
    data.n = 1;
    data.affine_cone = 1;
    data.kind = geometry->kind;
    memcpy(data.f, geometry->tangent, 3 * sizeof(double));
    const double zero[] = {0, 0, 0, 0};
    /* F'w=0, g=-w, w in K*: -g'w=||w||^2 is strictly positive. */
    for (int i = 0; i < 3; ++i)
        data.offset[i] = -geometry->dual[i];
    evaluate(&data, scaling, zero, geometry->dual, TERMINATION_REASON_PRIMAL_INFEASIBLE);
    /* Reversing the offset gives the explicit feasible point x=0. */
    memcpy(data.offset, geometry->ray, 3 * sizeof(double));
    evaluate(&data, scaling, zero, geometry->dual, TERMINATION_REASON_UNSPECIFIED);

    /* Strongly nonuniform F: y_linear=-1 and y_cone=w exactly cancel
       stationarity in [w'F; F], with certificate objective 1. */
    data.n = 3;
    data.m = 1;
    memset(data.f, 0, sizeof(data.f));
    memset(data.offset, 0, sizeof(data.offset));
    const double diagonal[] = {1e-4, 1, 1e4};
    double dual[4] = {-1, 0, 0, 0};
    for (int i = 0; i < 3; ++i)
    {
        data.f[3 * i + i] = diagonal[i];
        data.a[i] = geometry->dual[i] * diagonal[i];
        dual[i + 1] = geometry->dual[i];
    }
    data.lower[0] = -INFINITY;
    data.upper[0] = -1;
    evaluate(&data, scaling, zero, dual, TERMINATION_REASON_PRIMAL_INFEASIBLE);
}

static void check_fixed_section(cone_type_t kind, int scaling)
{
    problem_t data = {0};
    data.n = 3;
    data.m = 1;
    data.variable_cone = 1;
    data.kind = kind;
    int coordinate = 0;
    double support = 1;
    data.anchor[1] = data.anchor[2] = 1;
    data.fixed[1] = data.fixed[2] = 1;
    if (kind == CONE_STANDARD_SOC)
    {
        data.fixed[1] = 0;
        data.anchor[0] = 1;
        data.anchor[1] = 0;
    }
    else if (kind == CONE_ROTATED_SOC)
        data.anchor[0] = support = sqrt(2.0);
    else if (kind == CONE_EXPONENTIAL)
        support = 0;
    else
    {
        CHECK(kind == CONE_POWER);
        coordinate = 2;
        data.fixed[0] = 1;
        data.fixed[2] = 0;
        data.anchor[0] = 1;
    }
    data.a[coordinate] = 1;
    data.upper[0] = INFINITY;
    const double zero[] = {0, 0, 0, 0}, dual[] = {1};

    /* The anchors attain sup_C e_i'x = 1, sqrt(2), 0, 1 for SOC,
       RSOC, EXP, POWER. Thus Pi_C(p+e_i)=p and the normal -e_i
       contributes -support to the certificate objective. */
    data.lower[0] = support + 1;
    double first = evaluate(&data, scaling, zero, dual, TERMINATION_REASON_PRIMAL_INFEASIBLE);
    data.lower[0] = support + 2;
    double second = evaluate(&data, scaling, zero, dual, TERMINATION_REASON_PRIMAL_INFEASIBLE);
    CHECK(first > 0 && isfinite(first) && isfinite(second));
    CHECK(fabs(second / first - 2) <= 1e-6);
    /* Missing the support contribution would instead give an incorrect
       positive certificate for this feasible problem when support>0. */
    data.lower[0] = support - 0.25;
    evaluate(&data, scaling, zero, dual, TERMINATION_REASON_UNSPECIFIED);

    /* The fixed section has a finite optimum, even though its parent cone
       has nonzero recession directions. In particular fixed coordinates
       must be zero in a recession candidate, not pinned to the anchor. */
    data.m = 0;
    data.c[coordinate] = -1;
    double primal[] = {0, 0, 0};
    primal[coordinate] = 1;
    evaluate(&data, scaling, primal, zero, TERMINATION_REASON_UNSPECIFIED);
}

static void check_fixed_section_cancellation(void)
{
    problem_t data = {0};
    data.n = 3;
    data.m = 1;
    data.variable_cone = 1;
    data.fixed[1] = 1;
    data.lower[0] = 1;
    data.upper[0] = INFINITY;
    const double zero[] = {0, 0, 0, 0}, dual[] = {1};

    /* This feasible SOC section has w=0 and t>=1. At p=(0,0,2^54),
       p+A'y rounds back to p, so p-q alone falsely reports zero error.
       The computed normal is zero; its stationarity defect is still -A'y. */
    data.kind = CONE_STANDARD_SOC;
    data.a[2] = 1;
    data.anchor[2] = 0x1p54;
    evaluate(&data, 0, zero, dual, TERMINATION_REASON_UNSPECIFIED);

    /* Likewise y=1 and x>=1 is feasible in EXP at (1,1,e). The feasible
       anchor (-2^54,1,1) loses the unit x shift to rounding, but cannot
       turn the row multiplier into an infeasibility certificate. */
    data.kind = CONE_EXPONENTIAL;
    data.a[2] = 0;
    data.a[0] = 1;
    data.anchor[0] = -0x1p54;
    data.anchor[1] = data.anchor[2] = 1;
    evaluate(&data, 0, zero, dual, TERMINATION_REASON_UNSPECIFIED);
}

int main(void)
{
    CHECK(pdhcg_device_initialize() == 0);
    const double sqrt_two = sqrt(2.0);
    const geometry_t geometries[] = {
        {CONE_STANDARD_SOC, {1, 0, 1}, {0, 0, 1}, {1, 0, -1}, {1, 0, 0}},
        {CONE_ROTATED_SOC, {1, 1, 1}, {0, 1, 1}, {1, -1, 0}, {1, 0, 0}},
        {CONE_PSD, {1, sqrt_two, 1}, {1, 0, 1}, {1, 0, -1}, {0, 1, 0}},
        {CONE_EXPONENTIAL, {-1, -1, 1}, {0, 1, 2}, {1, -1, 0}, {1, 0, 0}},
        {CONE_POWER, {0.25, 0.75, -1}, {1, 1, 0}, {3, -1, 0}, {0, 0, 1}},
    };
    for (int scaling = 0; scaling < 3; ++scaling)
        for (size_t i = 0; i < sizeof(geometries) / sizeof(geometries[0]); ++i)
        {
            check_variable_certificates(&geometries[i], scaling);
            check_affine_certificates(&geometries[i], scaling);
            if (geometries[i].kind != CONE_PSD)
                check_fixed_section(geometries[i].kind, scaling);
        }
    check_fixed_section_cancellation();
    CHECK(pdhcg_device_synchronize() == 0);
    puts("Conic certificates, fixed-section support, and original-coordinate scaling passed");
    return 0;
}
