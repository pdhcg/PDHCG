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

#include "backend.h"
#include "cbf_parser.h"
#include "cone_utils.h"
#include "mps_parser.h"
#include "utils.h"
#include <signal.h>
#include <stdio.h>
#include <string.h>

#if PDHCG_HAS_CPU
extern const pdhcg_backend_t *pdhcg_cpu_pdhcg_backend(void);
extern volatile sig_atomic_t pdhcg_cpu_g_pdhcg_cancel_request;
#endif
#if PDHCG_HAS_CUDA
extern const pdhcg_backend_t *pdhcg_cuda_pdhcg_backend(void);
extern volatile sig_atomic_t pdhcg_cuda_g_pdhcg_cancel_request;
#endif

void pdhcg_set_cancel_request(int requested)
{
#if PDHCG_HAS_CPU
    pdhcg_cpu_g_pdhcg_cancel_request = requested;
#endif
#if PDHCG_HAS_CUDA
    pdhcg_cuda_g_pdhcg_cancel_request = requested;
#endif
}

static const pdhcg_backend_t *host_backend(void)
{
#if PDHCG_HAS_CPU
    return pdhcg_cpu_pdhcg_backend();
#else
    return pdhcg_cuda_pdhcg_backend();
#endif
}

static const pdhcg_backend_t *select_backend(const char *device)
{
    if (device == NULL || strcmp(device, "auto") == 0)
        device = pdhcg_get_default_device();
#if PDHCG_HAS_CPU
    if (strcmp(device, "cpu") == 0)
        return pdhcg_cpu_pdhcg_backend();
#endif
#if PDHCG_HAS_CUDA
    if (strcmp(device, "cuda") == 0)
        return pdhcg_cuda_pdhcg_backend();
#endif
    return NULL;
}

const char *pdhcg_get_default_device(void)
{
    return PDHCG_DEFAULT_DEVICE_NAME;
}

const char *const *pdhcg_get_built_devices(size_t *count)
{
    static const char *const devices[] = {
#if PDHCG_HAS_CPU
        "cpu",
#endif
#if PDHCG_HAS_CUDA
        "cuda",
#endif
    };
    if (count)
        *count = sizeof(devices) / sizeof(devices[0]);
    return devices;
}

int pdhcg_validate_parameters(const pdhg_parameters_t *params, char *error_message, size_t error_message_size)
{
    if (params && !select_backend(params->device))
    {
        if (error_message && error_message_size > 0)
            snprintf(error_message, error_message_size, "device '%s' is not compiled into this build", params->device);
        return -1;
    }
    return host_backend()->validate(params, error_message, error_message_size);
}

void set_default_parameters(pdhg_parameters_t *params)
{
    host_backend()->defaults(params);
}

pdhcg_result_t *solve_qp_problem(const qp_problem_t *prob, const pdhg_parameters_t *params)
{
    if (prob == NULL)
    {
        fprintf(stderr, "[solve_qp_problem] problem must not be NULL.\n");
        return NULL;
    }
    pdhg_parameters_t defaults;
    if (params == NULL)
    {
        set_default_parameters(&defaults);
        params = &defaults;
    }
    char message[256];
    if (pdhcg_validate_parameters(params, message, sizeof(message)) != 0)
    {
        fprintf(stderr, "[solve_qp_problem] invalid parameters: %s.\n", message);
        return NULL;
    }
    const pdhcg_backend_t *backend = select_backend(params->device);
    int status = backend->initialize();
    if (status != 0)
    {
        const char *device = params->device;
        if (device == NULL || strcmp(device, "auto") == 0)
            device = pdhcg_get_default_device();
        fprintf(stderr, "[solve_qp_problem] failed to initialize device '%s' (status %d).\n", device, status);
        return NULL;
    }
    return backend->solve(prob, params);
}

pdhcg_result_t *solve_qp_problem_distributed(const pdhg_parameters_t *params, const qp_problem_t *prob)
{
    pdhg_parameters_t defaults;
    if (params == NULL)
    {
        set_default_parameters(&defaults);
        params = &defaults;
    }
    char message[256];
    if (pdhcg_validate_parameters(params, message, sizeof(message)) != 0)
    {
        fprintf(stderr, "[solve_qp_problem_distributed] invalid parameters: %s.\n", message);
        return NULL;
    }
    const char *device = params->device;
    if (device == NULL || strcmp(device, "auto") == 0)
        device = pdhcg_get_default_device();
    if (strcmp(device, "cuda") != 0)
    {
        fprintf(stderr, "[solve_qp_problem_distributed] distributed solves require the cuda device.\n");
        return NULL;
    }
    /* The distributed solver selects each rank's GPU before initialization. */
    return select_backend(device)->solve_distributed(params, prob);
}

qp_problem_t *create_qp_problem(const double *objective_c,
                                const matrix_desc_t *Q_desc,
                                const matrix_desc_t *R_desc,
                                const matrix_desc_t *D_desc,
                                const matrix_desc_t *A_desc,
                                const double *con_lb,
                                const double *con_ub,
                                const double *var_lb,
                                const double *var_ub,
                                const double *objective_constant,
                                int num_var_cones,
                                const cone_spec_t *var_cones,
                                const matrix_desc_t *affine_cone_matrix_desc,
                                const double *affine_cone_offset,
                                int num_affine_cones,
                                const cone_spec_t *affine_cones)
{
    return host_backend()->create(objective_c, Q_desc, R_desc, D_desc, A_desc, con_lb, con_ub,
                                  var_lb, var_ub, objective_constant, num_var_cones, var_cones,
                                  affine_cone_matrix_desc, affine_cone_offset, num_affine_cones,
                                  affine_cones);
}

void set_start_values(qp_problem_t *prob, const double *primal, const double *dual)
{
    host_backend()->set_start(prob, primal, dual);
}

int set_cone_fixed(qp_problem_t *prob, int cone_idx, int slot, double value)
{
    return host_backend()->set_fixed(prob, cone_idx, slot, value);
}

qp_problem_t *qcqp_to_socp_qp(const qp_problem_t *prob, cone_type_t default_type)
{
    return host_backend()->to_socp(prob, default_type);
}

void pdhcg_result_free(pdhcg_result_t *result)
{
    host_backend()->free_result(result);
}

void qp_problem_free(qp_problem_t *prob)
{
    host_backend()->free_problem(prob);
}

qp_problem_t *read_mps_file(const char *filename)
{
    return host_backend()->read_mps(filename);
}

qp_problem_t *read_cbf_file(const char *filename)
{
    return host_backend()->read_cbf(filename);
}

const char *pdhcg_presolve_version(void)
{
    return host_backend()->presolve_version();
}

int pdhcg_presolve_available(void)
{
    return host_backend()->presolve_available();
}

const char *pdhcg_get_presolve_status_str(int status)
{
    return host_backend()->presolve_status(status);
}

pdhcg_presolve_info_t *pdhcg_presolve(const qp_problem_t *prob, const pdhg_parameters_t *params)
{
    return host_backend()->presolve(prob, params);
}

pdhcg_result_t *pdhcg_create_result_from_presolve(const pdhcg_presolve_info_t *info, const qp_problem_t *prob)
{
    return host_backend()->presolve_result(info, prob);
}

int pdhcg_postsolve(const pdhcg_presolve_info_t *info, pdhcg_result_t *result, const qp_problem_t *prob)
{
    return host_backend()->postsolve(info, result, prob);
}

void pdhcg_presolve_info_free(pdhcg_presolve_info_t *info)
{
    host_backend()->free_presolve(info);
}

void *safe_malloc(size_t size)
{
    return host_backend()->allocate_host(size);
}

const char *termination_reason_to_string(termination_reason_t reason)
{
    return host_backend()->termination_name(reason);
}

int cone_length(cone_type_t type, int v_dim)
{
    return host_backend()->cone_size(type, v_dim);
}

int cone_block_length(const cone_blocks_t *blocks, int block)
{
    return host_backend()->cone_block_size(blocks, block);
}
