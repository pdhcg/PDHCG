# C API Overview

Include `pdhcg.h` for the public API and type definitions. See
[Devices](../devices.md) for backend selection and execution settings.

## Linking

When using CMake's `add_subdirectory`, link the public target:

```cmake
target_link_libraries(my_app PRIVATE pdhcg::pdhcg)
```

## Quick Example

Solve `min x² + y² - 2x - 4y` with `x + y >= 4` and `x, y >= 0`.

```c
#include "pdhcg.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    double q[] = {2.0, 0.0, 0.0, 2.0};
    double a[] = {1.0, 1.0};
    double c[] = {-2.0, -4.0};
    double con_lb[] = {4.0};
    double var_lb[] = {0.0, 0.0};
    matrix_desc_t Q = {
        .m = 2, .n = 2, .fmt = matrix_dense, .data.dense.A = q
    };
    matrix_desc_t A = {
        .m = 1, .n = 2, .fmt = matrix_dense, .data.dense.A = a
    };

    qp_problem_t *prob = create_qp_problem(
        c, &Q, NULL, NULL, &A, con_lb, NULL, var_lb, NULL, NULL,
        0, NULL, NULL, NULL, 0, NULL
    );
    if (!prob)
        return EXIT_FAILURE;

    pdhg_parameters_t params;
    set_default_parameters(&params);
    pdhcg_result_t *result = solve_qp_problem(prob, &params);
    qp_problem_free(prob);
    if (!result)
        return EXIT_FAILURE;

    printf("Status: %d\n", result->termination_reason);
    printf("Objective: %f\n", result->primal_objective_value);
    printf("x = %f, y = %f\n", result->primal_solution[0], result->primal_solution[1]);
    pdhcg_result_free(result);
    return EXIT_SUCCESS;
}
```

The optimum is `(1.5, 2.5)`, with objective `-4.5`.
See [Functions](functions.md) for calls and ownership, and
[Types](types.md) for matrix, cone, parameter, and result layouts.
