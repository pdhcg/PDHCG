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

#include "device_linearized.h"

linearized_context_t *pdhcg_device_linearized_create(const pdhg_solver_state_t *state)
{
    (void)state;
    return NULL;
}

void pdhcg_device_linearized_execute(pdhg_solver_state_t *state,
                                     const pdhg_parameters_t *params,
                                     int iterations)
{
    (void)state;
    (void)params;
    (void)iterations;
}

void pdhcg_device_linearized_destroy(linearized_context_t *context)
{
    (void)context;
}
