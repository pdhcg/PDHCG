/* Copyright 2026 Hongpei Li. Licensed under the Apache License, Version 2.0. */
#pragma once

#include "internal_types.h"

#ifdef __cplusplus
extern "C"
{
#endif
    /* Evaluate fixed-point differences, plus iterate candidates for conic problems,
       before ordinary residuals reuse the delta vectors. */
    void compute_infeasibility_information(pdhg_solver_state_t *state);
    termination_reason_t check_infeasibility_criteria(const pdhg_solver_state_t *state, double tolerance);
#ifdef __cplusplus
}
#endif
