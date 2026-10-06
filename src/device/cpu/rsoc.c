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

#include "cone_kernel_ops.h"
#include "cone_projection_utils.h"

static inline double cone_section_negative_rsoc_upper(double omega_s,
                                                      double omega_t,
                                                      double s,
                                                      double t,
                                                      double fixed_norm2,
                                                      double polar_norm2,
                                                      double max_vector_metric)
{
    const double inv_sqrt2 = 0.70710678118654752440;
    double sqrt_omega_s = sqrt(omega_s);
    double sqrt_omega_t = sqrt(omega_t);
    double root_metric = sqrt_omega_s * sqrt_omega_t;
    double scaled_s = sqrt_omega_s * s;
    double scaled_t = sqrt_omega_t * t;
    double transformed_w = (scaled_s - scaled_t) * inv_sqrt2;
    double endpoint_polar = -(scaled_s + scaled_t) * inv_sqrt2;
    double transformed_fixed_norm2 = root_metric * fixed_norm2;
    double transformed_polar_norm2 = polar_norm2 / root_metric + transformed_w * transformed_w;
    double transformed_max_metric = fmax(1.0, max_vector_metric / root_metric);
    double transformed_upper = cone_section_negative_soc_upper(
        1.0, endpoint_polar, transformed_fixed_norm2, transformed_polar_norm2, transformed_max_metric);
    return root_metric * transformed_upper;
}

static inline double rotated_soc_smooth_objective(const double *point,
                                                  const double *rescaling,
                                                  const double *q_diag,
                                                  double tau,
                                                  int start,
                                                  int k,
                                                  const char *is_fixed,
                                                  double lambda,
                                                  double s,
                                                  double t)
{
    double objective = 0.0;
    for (int slot = 0; slot < k; ++slot)
    {
        int index = start + slot;
        if (is_fixed[index])
            continue;
        double omega = cone_section_weight(rescaling, q_diag, tau, index);
        double input = cone_section_actual(point, rescaling, index);
        double value = input * omega / (omega + lambda);
        double delta = value - input;
        objective += omega * delta * delta;
    }
    int s_index = start + k;
    int t_index = s_index + 1;
    double omega_s = cone_section_weight(rescaling, q_diag, tau, s_index);
    double omega_t = cone_section_weight(rescaling, q_diag, tau, t_index);
    double ds = s - cone_section_actual(point, rescaling, s_index);
    double dt = t - cone_section_actual(point, rescaling, t_index);
    return objective + omega_s * ds * ds + omega_t * dt * dt;
}

static inline void project_rotated_soc_section_serial(double *point,
                                                      const double *rescaling,
                                                      const double *q_diag,
                                                      double tau,
                                                      double *warm_start,
                                                      int start,
                                                      int k,
                                                      const char *is_fixed)
{
    int s_index = start + k;
    int t_index = s_index + 1;
    bool fixed_s = is_fixed[s_index] != 0;
    bool fixed_t = is_fixed[t_index] != 0;
    double s_input = cone_section_actual(point, rescaling, s_index);
    double t_input = cone_section_actual(point, rescaling, t_index);
    double fixed_norm2 = 0.0;
    double free_norm2 = 0.0;
    double polar_norm2 = 0.0;
    double max_omega = 0.0;
    int free_count = 0;
    for (int slot = 0; slot < k; ++slot)
    {
        int index = start + slot;
        double value = cone_section_actual(point, rescaling, index);
        if (is_fixed[index])
            fixed_norm2 += value * value;
        else
        {
            double omega = cone_section_weight(rescaling, q_diag, tau, index);
            free_norm2 += value * value;
            polar_norm2 += (omega * value) * (omega * value);
            max_omega = fmax(max_omega, omega);
            ++free_count;
        }
    }

    if (fixed_s && fixed_t)
    {
        double radius2 = fmax(0.0, 2.0 * s_input * t_input - fixed_norm2);
        if (free_count == 0 || free_norm2 <= radius2)
            return;
        if (!(radius2 > 0.0))
        {
            for (int slot = 0; slot < k; ++slot)
                if (!is_fixed[start + slot])
                    point[start + slot] = 0.0;
            return;
        }

        double lo = 0.0;
        double hi = sqrt(polar_norm2) / sqrt(radius2) * (1.0 + 64.0 * DBL_EPSILON);
        if (!(hi > 0.0) || !isfinite(hi))
        {
            hi = warm_start && *warm_start > 0.0 && isfinite(*warm_start) ? *warm_start : 1.0;
            for (int expansion = 0; expansion < 100; ++expansion)
            {
                double norm2 = 0.0;
                for (int slot = 0; slot < k; ++slot)
                {
                    int index = start + slot;
                    if (is_fixed[index])
                        continue;
                    double omega = cone_section_weight(rescaling, q_diag, tau, index);
                    double value = cone_section_actual(point, rescaling, index) * omega / (omega + hi);
                    norm2 += value * value;
                }
                if (norm2 <= radius2)
                    break;
                hi *= 2.0;
            }
        }
        for (int iteration = 0; iteration < 80; ++iteration)
        {
            double lambda = 0.5 * (lo + hi);
            double norm2 = 0.0;
            for (int slot = 0; slot < k; ++slot)
            {
                int index = start + slot;
                if (is_fixed[index])
                    continue;
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                double value = cone_section_actual(point, rescaling, index) * omega / (omega + lambda);
                norm2 += value * value;
            }
            if (norm2 > radius2)
                lo = lambda;
            else
                hi = lambda;
            if ((hi - lo) <= 1e-13 * (1.0 + hi + lo))
                break;
        }
        double lambda = 0.5 * (lo + hi);
        if (warm_start)
            *warm_start = lambda;
        for (int slot = 0; slot < k; ++slot)
        {
            int index = start + slot;
            if (!is_fixed[index])
            {
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                point[index] *= omega / (omega + lambda);
            }
        }
        return;
    }

    if (fixed_s || fixed_t)
    {
        int free_endpoint_index = fixed_s ? t_index : s_index;
        double fixed_endpoint = fixed_s ? s_input : t_input;
        double free_endpoint_input = fixed_s ? t_input : s_input;
        double omega_endpoint = cone_section_weight(rescaling, q_diag, tau, free_endpoint_index);
        if (!(fixed_endpoint > 0.0))
        {
            for (int slot = 0; slot < k; ++slot)
                if (!is_fixed[start + slot])
                    point[start + slot] = 0.0;
            point[free_endpoint_index] = fmax(free_endpoint_input, 0.0) * rescaling[free_endpoint_index];
            return;
        }
        if (free_endpoint_input >= 0.0 && fixed_norm2 + free_norm2 <= 2.0 * fixed_endpoint * free_endpoint_input)
            return;
        if (free_count == 0)
        {
            double lower_bound = fixed_norm2 / (2.0 * fixed_endpoint);
            point[free_endpoint_index] = fmax(free_endpoint_input, lower_bound) * rescaling[free_endpoint_index];
            return;
        }

        double lo = 0.0;
        double violation = fixed_norm2 + free_norm2 - 2.0 * fixed_endpoint * free_endpoint_input;
        double hi = omega_endpoint * violation / (2.0 * fixed_endpoint * fixed_endpoint);
        hi *= 1.0 + 64.0 * DBL_EPSILON;
        if (!(hi > 0.0) || !isfinite(hi))
        {
            hi = warm_start && *warm_start > 0.0 && isfinite(*warm_start) ? *warm_start : omega_endpoint;
            for (int expansion = 0; expansion < 100; ++expansion)
            {
                double norm2 = fixed_norm2;
                for (int slot = 0; slot < k; ++slot)
                {
                    int index = start + slot;
                    if (is_fixed[index])
                        continue;
                    double omega = cone_section_weight(rescaling, q_diag, tau, index);
                    double value = cone_section_actual(point, rescaling, index) * omega / (omega + hi);
                    norm2 += value * value;
                }
                double endpoint = free_endpoint_input + hi * fixed_endpoint / omega_endpoint;
                if (norm2 <= 2.0 * fixed_endpoint * endpoint)
                    break;
                hi *= 2.0;
            }
        }
        for (int iteration = 0; iteration < 80; ++iteration)
        {
            double lambda = 0.5 * (lo + hi);
            double norm2 = fixed_norm2;
            for (int slot = 0; slot < k; ++slot)
            {
                int index = start + slot;
                if (is_fixed[index])
                    continue;
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                double value = cone_section_actual(point, rescaling, index) * omega / (omega + lambda);
                norm2 += value * value;
            }
            double endpoint = free_endpoint_input + lambda * fixed_endpoint / omega_endpoint;
            if (norm2 > 2.0 * fixed_endpoint * endpoint)
                lo = lambda;
            else
                hi = lambda;
            if ((hi - lo) <= 1e-13 * (1.0 + hi + lo))
                break;
        }
        double lambda = 0.5 * (lo + hi);
        if (warm_start)
            *warm_start = lambda;
        for (int slot = 0; slot < k; ++slot)
        {
            int index = start + slot;
            if (!is_fixed[index])
            {
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                point[index] *= omega / (omega + lambda);
            }
        }
        point[free_endpoint_index] =
            (free_endpoint_input + lambda * fixed_endpoint / omega_endpoint) * rescaling[free_endpoint_index];
        return;
    }

    double total_norm2 = fixed_norm2 + free_norm2;
    if (s_input >= 0.0 && t_input >= 0.0 && total_norm2 <= 2.0 * s_input * t_input)
        return;

    double omega_s = cone_section_weight(rescaling, q_diag, tau, s_index);
    double omega_t = cone_section_weight(rescaling, q_diag, tau, t_index);
    if (fixed_norm2 == 0.0)
    {
        double bs = omega_s * s_input;
        double bt = omega_t * t_input;
        if (bs <= 0.0 && bt <= 0.0 && polar_norm2 <= 2.0 * bs * bt)
        {
            for (int slot = 0; slot < k; ++slot)
                if (!is_fixed[start + slot])
                    point[start + slot] = 0.0;
            point[s_index] = 0.0;
            point[t_index] = 0.0;
            return;
        }
    }

    double root_metric = sqrt(omega_s) * sqrt(omega_t);
    double balance = sqrt(omega_s) * s_input + sqrt(omega_t) * t_input;
    double balance_scale = 1.0 + fabs(sqrt(omega_s) * s_input) + fabs(sqrt(omega_t) * t_input);
    double lambda = root_metric;
    double projected_s = 0.0;
    double projected_t = 0.0;
    bool smooth_valid = true;

    if (fabs(balance) <= 64.0 * DBL_EPSILON * balance_scale)
    {
        double norm2 = fixed_norm2;
        for (int slot = 0; slot < k; ++slot)
        {
            int index = start + slot;
            if (is_fixed[index])
                continue;
            double omega = cone_section_weight(rescaling, q_diag, tau, index);
            double value = cone_section_actual(point, rescaling, index) * omega / (omega + lambda);
            norm2 += value * value;
        }
        double product = 0.5 * root_metric * norm2;
        double delta = sqrt(omega_s) * s_input;
        double scaled_t = 0.5 * (-delta + sqrt(fmax(0.0, delta * delta + 4.0 * product)));
        double scaled_s = scaled_t + delta;
        projected_s = scaled_s / sqrt(omega_s);
        projected_t = scaled_t / sqrt(omega_t);
        smooth_valid = projected_s >= 0.0 && projected_t >= 0.0;
    }
    else
    {
        bool lower_branch = balance > 0.0;
        double lo = lower_branch ? 0.0 : root_metric * (1.0 + 1e-14);
        double hi = lower_branch ? root_metric * (1.0 - 1e-14) : 2.0 * root_metric;

        if (!lower_branch)
        {
            hi = cone_section_negative_rsoc_upper(
                omega_s, omega_t, s_input, t_input, fixed_norm2, polar_norm2, max_omega);
            if (!(hi > lo) || !isfinite(hi))
            {
                hi = 2.0 * root_metric;
                for (int expansion = 0; expansion < 100; ++expansion)
                {
                    double determinant = omega_s * omega_t - hi * hi;
                    double s = omega_t * (omega_s * s_input + hi * t_input) / determinant;
                    double t = omega_s * (omega_t * t_input + hi * s_input) / determinant;
                    double f = INFINITY;
                    if (s >= 0.0 && t >= 0.0)
                    {
                        double norm2 = fixed_norm2;
                        for (int slot = 0; slot < k; ++slot)
                        {
                            int index = start + slot;
                            if (is_fixed[index])
                                continue;
                            double omega = cone_section_weight(rescaling, q_diag, tau, index);
                            double value = cone_section_actual(point, rescaling, index) * omega / (omega + hi);
                            norm2 += value * value;
                        }
                        f = norm2 - 2.0 * s * t;
                    }
                    if (f >= 0.0)
                        break;
                    hi *= 2.0;
                }
            }
        }

        for (int iteration = 0; iteration < 90; ++iteration)
        {
            double trial = 0.5 * (lo + hi);
            double determinant = omega_s * omega_t - trial * trial;
            double s = omega_t * (omega_s * s_input + trial * t_input) / determinant;
            double t = omega_s * (omega_t * t_input + trial * s_input) / determinant;
            double f = INFINITY;
            if (s >= 0.0 && t >= 0.0)
            {
                double norm2 = fixed_norm2;
                for (int slot = 0; slot < k; ++slot)
                {
                    int index = start + slot;
                    if (is_fixed[index])
                        continue;
                    double omega = cone_section_weight(rescaling, q_diag, tau, index);
                    double value = cone_section_actual(point, rescaling, index) * omega / (omega + trial);
                    norm2 += value * value;
                }
                f = norm2 - 2.0 * s * t;
            }
            if ((lower_branch && f > 0.0) || (!lower_branch && f < 0.0))
                lo = trial;
            else
                hi = trial;
            if ((hi - lo) <= 1e-13 * (1.0 + hi + lo))
                break;
        }
        lambda = 0.5 * (lo + hi);
        double determinant = omega_s * omega_t - lambda * lambda;
        projected_s = omega_t * (omega_s * s_input + lambda * t_input) / determinant;
        projected_t = omega_s * (omega_t * t_input + lambda * s_input) / determinant;
        smooth_valid = isfinite(projected_s) && isfinite(projected_t) && projected_s >= 0.0 && projected_t >= 0.0;
    }

    double best_objective = smooth_valid
        ? rotated_soc_smooth_objective(
              point, rescaling, q_diag, tau, start, k, is_fixed, lambda, projected_s, projected_t)
        : INFINITY;
    int mode = smooth_valid ? 0 : 1;
    if (fixed_norm2 == 0.0)
    {
        double vector_objective = 0.0;
        for (int slot = 0; slot < k; ++slot)
        {
            int index = start + slot;
            if (!is_fixed[index])
            {
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                double value = cone_section_actual(point, rescaling, index);
                vector_objective += omega * value * value;
            }
        }
        double s_axis = fmax(s_input, 0.0);
        double s_axis_objective =
            vector_objective + omega_s * (s_axis - s_input) * (s_axis - s_input) + omega_t * t_input * t_input;
        if (s_axis_objective < best_objective)
        {
            best_objective = s_axis_objective;
            projected_s = s_axis;
            projected_t = 0.0;
            mode = 1;
        }
        double t_axis = fmax(t_input, 0.0);
        double t_axis_objective =
            vector_objective + omega_s * s_input * s_input + omega_t * (t_axis - t_input) * (t_axis - t_input);
        if (t_axis_objective < best_objective)
        {
            projected_s = 0.0;
            projected_t = t_axis;
            mode = 1;
        }
    }

    if (mode == 0)
    {
        for (int slot = 0; slot < k; ++slot)
        {
            int index = start + slot;
            if (!is_fixed[index])
            {
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                point[index] *= omega / (omega + lambda);
            }
        }
        if (warm_start)
            *warm_start = lambda;
    }
    else
    {
        for (int slot = 0; slot < k; ++slot)
            if (!is_fixed[start + slot])
                point[start + slot] = 0.0;
        if (warm_start)
            *warm_start = 0.0;
    }
    point[s_index] = projected_s * rescaling[s_index];
    point[t_index] = projected_t * rescaling[t_index];
}

static void rsoc_project(double *__restrict__ primal_solution,
                         const double *__restrict__ variable_rescaling,
                         double *__restrict__ workspace,
                         const int *__restrict__ start_idx,
                         const int *__restrict__ v_dim,
                         const double *__restrict__ power_alpha,
                         const char *__restrict__ is_fixed,
                         int num_blocks)
{
    (void)power_alpha;
#pragma omp parallel for schedule(static) if (num_blocks >= 16)
    for (int blk = 0; blk < num_blocks; ++blk)
    {
        const double INV_SQRT2 = 0.7071067811865475;

        int start = start_idx[blk];
        int k = v_dim[blk];
        if (cone_section_has_fixed(is_fixed, start, k + 2))
        {
            project_rotated_soc_section_serial(
                primal_solution, variable_rescaling, NULL, 0.0, workspace + blk, start, k, is_fixed);
            continue;
        }
        double *v = primal_solution + start;
        double *sptr = primal_solution + start + k;
        double *tptr = primal_solution + start + k + 1;

        double s = *sptr;
        double t = *tptr;

        double w = (s - t) * INV_SQRT2;
        double z = (s + t) * INV_SQRT2;

        double d_s = variable_rescaling[start + k];
        double d_t = variable_rescaling[start + k + 1];
        double d_st = sqrt(d_s * d_t);

        bool diag_uniform = true;
        for (int m = 0; m < k && diag_uniform; ++m)
        {
            if (variable_rescaling[start + m] != d_st)
                diag_uniform = false;
        }
        if (diag_uniform)
        {
            double sumsq = w * w;
            for (int m = 0; m < k; ++m)
                sumsq += v[m] * v[m];
            double r = sqrt(sumsq);
            if (r <= z)
                continue;
            if (r <= -z)
            {
                for (int m = 0; m < k; ++m)
                    v[m] = 0.0;
                *sptr = 0.0;
                *tptr = 0.0;
                continue;
            }
            double scale = (z + r) / (2.0 * r);
            for (int m = 0; m < k; ++m)
                v[m] *= scale;
            double w_new = scale * w;
            double z_new = scale * r;
            *sptr = (z_new + w_new) * INV_SQRT2;
            *tptr = (z_new - w_new) * INV_SQRT2;
            continue;
        }

        double r_inv_sq = w * w;
        double r_pos_sq = w * w;
        for (int m = 0; m < k; ++m)
        {
            double dh = variable_rescaling[start + m] / d_st;
            double v_m = v[m];
            r_inv_sq += (v_m / dh) * (v_m / dh);
            r_pos_sq += (v_m * dh) * (v_m * dh);
        }
        double r_inv = sqrt(r_inv_sq);
        if (r_inv <= z)
            continue;
        double r_pos = sqrt(r_pos_sq);
        if (r_pos <= -z)
        {
            for (int m = 0; m < k; ++m)
                v[m] = 0.0;
            *sptr = 0.0;
            *tptr = 0.0;
            continue;
        }

        double lo, hi;
        bool z_pos = (z > 0.0);
        if (z_pos)
        {
            lo = 0.0;
            hi = 0.5 - 1e-14;
        }
        else
        {
            lo = 0.5 + 1e-14;
            hi = 1.0;
            for (int doubling = 0; doubling < 60; ++doubling)
            {
                double sum_hi = 0.0;
                for (int m = 0; m < k; ++m)
                {
                    double dh = variable_rescaling[start + m] / d_st;
                    double dh2 = dh * dh;
                    double tt = v[m] * dh / (dh2 + 2.0 * hi);
                    sum_hi += tt * tt;
                }
                double tw_hi = w / (1.0 + 2.0 * hi);
                sum_hi += tw_hi * tw_hi;
                double zt_hi = z / (1.0 - 2.0 * hi);
                double f_hi = sum_hi - zt_hi * zt_hi;
                if (f_hi > 0.0)
                    break;
                lo = hi;
                hi *= 2.0;
            }
        }

        double warm_lam = workspace[blk];
        if (warm_lam > lo && warm_lam < hi)
        {
            double sum_w = 0.0;
            for (int m = 0; m < k; ++m)
            {
                double dh = variable_rescaling[start + m] / d_st;
                double dh2 = dh * dh;
                double tt = v[m] * dh / (dh2 + 2.0 * warm_lam);
                sum_w += tt * tt;
            }
            double tw = w / (1.0 + 2.0 * warm_lam);
            sum_w += tw * tw;
            double zt = z / (1.0 - 2.0 * warm_lam);
            double f = sum_w - zt * zt;
            if (fabs(f) < 1e-12)
            {
                double w_new = w / (1.0 + 2.0 * warm_lam);
                double z_new = z / (1.0 - 2.0 * warm_lam);
                for (int m = 0; m < k; ++m)
                {
                    double dh = variable_rescaling[start + m] / d_st;
                    double dh2 = dh * dh;
                    v[m] = v[m] * dh2 / (dh2 + 2.0 * warm_lam);
                }
                *sptr = (z_new + w_new) * INV_SQRT2;
                *tptr = (z_new - w_new) * INV_SQRT2;
                continue;
            }
            if (z_pos)
            {
                if (f > 0.0)
                    lo = warm_lam;
                else
                    hi = warm_lam;
            }
            else
            {
                if (f > 0.0)
                    hi = warm_lam;
                else
                    lo = warm_lam;
            }
        }

        for (int it = 0; it < 60; ++it)
        {
            double lam = 0.5 * (lo + hi);
            double sum = 0.0;
            for (int m = 0; m < k; ++m)
            {
                double dh = variable_rescaling[start + m] / d_st;
                double dh2 = dh * dh;
                double tt = v[m] * dh / (dh2 + 2.0 * lam);
                sum += tt * tt;
            }
            double tw = w / (1.0 + 2.0 * lam);
            sum += tw * tw;
            double zt = z / (1.0 - 2.0 * lam);
            double f = sum - zt * zt;
            if (z_pos)
            {
                if (f > 0.0)
                    lo = lam;
                else
                    hi = lam;
            }
            else
            {
                if (f > 0.0)
                    hi = lam;
                else
                    lo = lam;
            }
            if ((hi - lo) / (1.0 + hi + lo) < 1e-13)
                break;
        }
        double lam = 0.5 * (lo + hi);
        workspace[blk] = lam;

        double w_new = w / (1.0 + 2.0 * lam);
        double z_new = z / (1.0 - 2.0 * lam);
        for (int m = 0; m < k; ++m)
        {
            double dh = variable_rescaling[start + m] / d_st;
            double dh2 = dh * dh;
            v[m] = v[m] * dh2 / (dh2 + 2.0 * lam);
        }
        *sptr = (z_new + w_new) * INV_SQRT2;
        *tptr = (z_new - w_new) * INV_SQRT2;
    }
}

static void rsoc_diag(double *__restrict__ pdhg_primal,
                      double *__restrict__ reflected_primal,
                      const double *__restrict__ current_primal,
                      const double *__restrict__ variable_rescaling,
                      const double *__restrict__ Q_diag,
                      double tau,
                      double *__restrict__ workspace,
                      const int *__restrict__ start_idx,
                      const int *__restrict__ v_dim,
                      const double *__restrict__ power_alpha,
                      const char *__restrict__ is_fixed,
                      int num_blocks)
{
    (void)power_alpha;
#pragma omp parallel for schedule(static) if (num_blocks >= 16)
    for (int blk = 0; blk < num_blocks; ++blk)
    {
        const double W_FLOOR = 1e-300;

        int start = start_idx[blk];
        int k = v_dim[blk];
        int len = k + 2;

        if (cone_section_has_fixed(is_fixed, start, len))
        {
            project_rotated_soc_section_serial(
                pdhg_primal, variable_rescaling, Q_diag, tau, workspace + blk, start, k, is_fixed);
            for (int slot = 0; slot < len; ++slot)
            {
                int index = start + slot;
                reflected_primal[index] = 2.0 * pdhg_primal[index] - current_primal[index];
            }
            continue;
        }

        double r_s = pdhg_primal[start + k];
        double r_t = pdhg_primal[start + k + 1];

        double q_s = Q_diag[start + k];
        double q_t = Q_diag[start + k + 1];
        double w_s = 1.0 + tau * q_s;
        double w_t = 1.0 + tau * q_t;
        if (!(w_s > W_FLOOR))
            w_s = W_FLOOR;
        if (!(w_t > W_FLOOR))
            w_t = W_FLOOR;
        double sigma = sqrt(w_s * w_t);
        double alpha = sqrt(w_t / w_s);
        double inv_alpha = 1.0 / alpha;

        double d_s = variable_rescaling[start + k];
        double d_t = variable_rescaling[start + k + 1];
        double d_st = sqrt(d_s * d_t);

        const double INV_SQRT2 = 0.7071067811865475;
        /* Fast path: no Q on cone slots (w_s = w_t = 1 and all w_v_i = 1) and uniform d_v = d_st.
               This is the COMMON case for QCQP transform aux vars. Reduces to LP-style RSOC closed form. */
        if (q_s == 0.0 && q_t == 0.0)
        {
            bool no_cone_Q = true;
            bool d_uniform = true;
            for (int m = 0; m < k; ++m)
            {
                if (Q_diag[start + m] != 0.0)
                {
                    no_cone_Q = false;
                    break;
                }
                if (variable_rescaling[start + m] != d_st)
                {
                    d_uniform = false;
                    break;
                }
            }
            if (no_cone_Q && d_uniform)
            {
                double w_val = (r_s - r_t) * INV_SQRT2;
                double z_val = (r_s + r_t) * INV_SQRT2;
                double sumsq = w_val * w_val;
                for (int m = 0; m < k; ++m)
                {
                    double vm = pdhg_primal[start + m];
                    sumsq += vm * vm;
                }
                double rnorm = sqrt(sumsq);
                if (rnorm <= z_val)
                {
                    for (int m = 0; m < len; ++m)
                    {
                        int idx = start + m;
                        reflected_primal[idx] = 2.0 * pdhg_primal[idx] - current_primal[idx];
                    }
                    continue;
                }
                if (rnorm <= -z_val)
                {
                    for (int m = 0; m < k; ++m)
                    {
                        pdhg_primal[start + m] = 0.0;
                        int idx = start + m;
                        reflected_primal[idx] = -current_primal[idx];
                    }
                    pdhg_primal[start + k] = 0.0;
                    pdhg_primal[start + k + 1] = 0.0;
                    reflected_primal[start + k] = -current_primal[start + k];
                    reflected_primal[start + k + 1] = -current_primal[start + k + 1];
                    continue;
                }
                double scale = (z_val + rnorm) / (2.0 * rnorm);
                double w_new = scale * w_val;
                double z_new = scale * rnorm;
                for (int m = 0; m < k; ++m)
                {
                    double v_new = scale * pdhg_primal[start + m];
                    pdhg_primal[start + m] = v_new;
                    int idx = start + m;
                    reflected_primal[idx] = 2.0 * v_new - current_primal[idx];
                }
                double s_new = (z_new + w_new) * INV_SQRT2;
                double t_new = (z_new - w_new) * INV_SQRT2;
                pdhg_primal[start + k] = s_new;
                pdhg_primal[start + k + 1] = t_new;
                reflected_primal[start + k] = 2.0 * s_new - current_primal[start + k];
                reflected_primal[start + k + 1] = 2.0 * t_new - current_primal[start + k + 1];
                continue;
            }
        }

        {
            double lhs = 0.0;
            for (int m = 0; m < k; ++m)
            {
                double d_m = variable_rescaling[start + m];
                double Ds = d_st / d_m;
                double rv = pdhg_primal[start + m];
                double term = Ds * rv;
                lhs += term * term;
            }
            if (r_s >= 0.0 && r_t >= 0.0 && lhs <= 2.0 * r_s * r_t)
            {
                for (int m = 0; m < len; ++m)
                {
                    int idx = start + m;
                    double pv = pdhg_primal[idx];
                    reflected_primal[idx] = 2.0 * pv - current_primal[idx];
                }
                continue;
            }
        }

        if (r_s <= 0.0 && r_t <= 0.0)
        {
            double rhs = 2.0 * sigma * sigma * r_s * r_t;
            double lhs = 0.0;
            for (int m = 0; m < k; ++m)
            {
                double d_m = variable_rescaling[start + m];
                double q_m = Q_diag[start + m];
                double w_m = 1.0 + tau * q_m;
                if (!(w_m > W_FLOOR))
                    w_m = W_FLOOR;
                double rv = pdhg_primal[start + m];
                double term = d_m * w_m * rv / d_st;
                lhs += term * term;
            }
            if (lhs <= rhs)
            {
                for (int m = 0; m < k; ++m)
                    pdhg_primal[start + m] = 0.0;
                pdhg_primal[start + k] = 0.0;
                pdhg_primal[start + k + 1] = 0.0;
                for (int m = 0; m < len; ++m)
                {
                    int idx = start + m;
                    reflected_primal[idx] = -current_primal[idx];
                }
                continue;
            }
        }

        double lo, hi;
        int bracket_kind; /* 0: f increasing on bracket; 1: f decreasing. */
        bool need_doubling = false;
        double sum_alpha = r_s + alpha * r_t;

        if (r_s > 0.0 && r_t > 0.0)
        {
            lo = 0.0;
            hi = 1.0 - 1e-14;
            bracket_kind = 1;
        }
        else if (r_s < 0.0 && r_t < 0.0)
        {
            lo = 1.0 + 1e-14;
            hi = 2.0;
            bracket_kind = 0;
            need_doubling = true;
        }
        else if (r_s <= 0.0 && r_t >= 0.0)
        {
            if (sum_alpha <= 0.0)
            {
                lo = 1.0 + 1e-14;
                if (r_t == 0.0)
                {
                    hi = 2.0;
                    need_doubling = true;
                }
                else
                {
                    hi = -r_s / (alpha * r_t);
                    if (!(hi > lo))
                        hi = lo + 1.0;
                }
                bracket_kind = 0;
            }
            else
            {
                lo = (r_t > 0.0) ? (-r_s / (alpha * r_t)) : 0.0;
                if (!(lo >= 0.0))
                    lo = 0.0;
                hi = 1.0 - 1e-14;
                if (!(lo < hi))
                    lo = hi - 1e-7;
                bracket_kind = 1;
            }
        }
        else
        {
            if (sum_alpha <= 0.0)
            {
                lo = 1.0 + 1e-14;
                if (r_s == 0.0)
                {
                    hi = 2.0;
                    need_doubling = true;
                }
                else
                {
                    hi = -alpha * r_t / r_s;
                    if (!(hi > lo))
                        hi = lo + 1.0;
                }
                bracket_kind = 0;
            }
            else
            {
                lo = (r_s > 0.0) ? (-alpha * r_t / r_s) : 0.0;
                if (!(lo >= 0.0))
                    lo = 0.0;
                hi = 1.0 - 1e-14;
                if (!(lo < hi))
                    lo = hi - 1e-7;
                bracket_kind = 1;
            }
        }

#define ORACLE_EVAL(ZETA, F_OUT)                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        double _zeta = (ZETA);                                                                                         \
        double _denom = 1.0 - _zeta * _zeta;                                                                           \
        double _s = (r_s + _zeta * alpha * r_t) / _denom;                                                              \
        double _t = (r_t + _zeta * inv_alpha * r_s) / _denom;                                                          \
        double _sv = 0.0;                                                                                              \
        for (int _m = 0; _m < k; ++_m)                                                                                 \
        {                                                                                                              \
            double _dm = variable_rescaling[start + _m];                                                               \
            double _Ds = d_st / _dm;                                                                                   \
            double _qm = Q_diag[start + _m];                                                                           \
            double _wm = 1.0 + tau * _qm;                                                                              \
            if (!(_wm > W_FLOOR))                                                                                      \
                _wm = W_FLOOR;                                                                                         \
            double _Dh2 = _Ds * _Ds * sigma / _wm;                                                                     \
            double _rv = pdhg_primal[start + _m];                                                                      \
            double _vz = _rv / (1.0 + _zeta * _Dh2);                                                                   \
            double _tm = _Ds * _vz;                                                                                    \
            _sv += _tm * _tm;                                                                                          \
        }                                                                                                              \
        (F_OUT) = _sv - 2.0 * _s * _t;                                                                                 \
    } while (0)

        if (need_doubling)
        {
            double f_hi;
            for (int dbl = 0; dbl < 60; ++dbl)
            {
                ORACLE_EVAL(hi, f_hi);
                if (f_hi >= 0.0)
                    break;
                lo = hi;
                hi *= 2.0;
            }
        }

        double warm_zeta = workspace[blk];
        if (warm_zeta > lo && warm_zeta < hi)
        {
            double f_w;
            ORACLE_EVAL(warm_zeta, f_w);
            if (fabs(f_w) < 1e-12)
            {
                double zeta = warm_zeta;
                double denom = 1.0 - zeta * zeta;
                double s_new = (r_s + zeta * alpha * r_t) / denom;
                double t_new = (r_t + zeta * inv_alpha * r_s) / denom;
                for (int m = 0; m < k; ++m)
                {
                    double d_m = variable_rescaling[start + m];
                    double Ds = d_st / d_m;
                    double q_m = Q_diag[start + m];
                    double w_m = 1.0 + tau * q_m;
                    if (!(w_m > W_FLOOR))
                        w_m = W_FLOOR;
                    double Dh2 = Ds * Ds * sigma / w_m;
                    double rv = pdhg_primal[start + m];
                    pdhg_primal[start + m] = rv / (1.0 + zeta * Dh2);
                }
                pdhg_primal[start + k] = s_new;
                pdhg_primal[start + k + 1] = t_new;
                for (int m = 0; m < len; ++m)
                {
                    int idx = start + m;
                    double pv = pdhg_primal[idx];
                    reflected_primal[idx] = 2.0 * pv - current_primal[idx];
                }
                continue;
            }
            if (bracket_kind == 0)
            {
                if (f_w < 0.0)
                    lo = warm_zeta;
                else
                    hi = warm_zeta;
            }
            else
            {
                if (f_w > 0.0)
                    lo = warm_zeta;
                else
                    hi = warm_zeta;
            }
        }

        for (int it = 0; it < 80; ++it)
        {
            double mid = 0.5 * (lo + hi);
            double f_m;
            ORACLE_EVAL(mid, f_m);
            if (bracket_kind == 0)
            {
                if (f_m < 0.0)
                    lo = mid;
                else
                    hi = mid;
            }
            else
            {
                if (f_m > 0.0)
                    lo = mid;
                else
                    hi = mid;
            }
            if ((hi - lo) / (1.0 + fabs(hi) + fabs(lo)) < 1e-13)
                break;
        }
        double zeta = 0.5 * (lo + hi);
        workspace[blk] = zeta;

        double denom = 1.0 - zeta * zeta;
        double s_new = (r_s + zeta * alpha * r_t) / denom;
        double t_new = (r_t + zeta * inv_alpha * r_s) / denom;
        for (int m = 0; m < k; ++m)
        {
            double d_m = variable_rescaling[start + m];
            double Ds = d_st / d_m;
            double q_m = Q_diag[start + m];
            double w_m = 1.0 + tau * q_m;
            if (!(w_m > W_FLOOR))
                w_m = W_FLOOR;
            double Dh2 = Ds * Ds * sigma / w_m;
            double rv = pdhg_primal[start + m];
            pdhg_primal[start + m] = rv / (1.0 + zeta * Dh2);
        }
        pdhg_primal[start + k] = s_new;
        pdhg_primal[start + k + 1] = t_new;

        for (int m = 0; m < len; ++m)
        {
            int idx = start + m;
            double pv = pdhg_primal[idx];
            reflected_primal[idx] = 2.0 * pv - current_primal[idx];
        }
#undef ORACLE_EVAL
    }
}

static void rsoc_residual(double *__restrict__ dual_residual,
                          double *__restrict__ complementarity_residual,
                          const double *__restrict__ objective_vector,
                          const double *__restrict__ dual_product,
                          const double *__restrict__ variable_rescaling,
                          const double *__restrict__ primal_solution,
                          double *__restrict__ workspace,
                          const int *__restrict__ start_idx,
                          const int *__restrict__ v_dim,
                          const double *__restrict__ power_alpha,
                          const char *__restrict__ is_fixed,
                          int num_blocks)
{
    (void)power_alpha;
#pragma omp parallel for schedule(static) if (num_blocks >= 16)
    for (int blk = 0; blk < num_blocks; ++blk)
    {
        const double INV_SQRT2 = 0.7071067811865475;
        int start = start_idx[blk];
        int k = v_dim[blk];

        if (cone_section_has_fixed(is_fixed, start, k + 2))
        {
            for (int slot = 0; slot < k + 2; ++slot)
            {
                int index = start + slot;
                double residual = objective_vector[index] - dual_product[index];
                dual_residual[index] = is_fixed[index] ? primal_solution[index] : primal_solution[index] - residual;
            }
            project_rotated_soc_section_serial(
                dual_residual, variable_rescaling, NULL, 0.0, workspace + blk, start, k, is_fixed);
            for (int slot = 0; slot < k + 2; ++slot)
            {
                int index = start + slot;
                dual_residual[index] =
                    is_fixed[index] ? 0.0 : (primal_solution[index] - dual_residual[index]) * variable_rescaling[index];
            }
            complementarity_residual[blk] = 0.0;
            continue;
        }

        double r_s = objective_vector[start + k] - dual_product[start + k];
        double r_t = objective_vector[start + k + 1] - dual_product[start + k + 1];
        double r_w = (r_s - r_t) * INV_SQRT2;
        double r_z = (r_s + r_t) * INV_SQRT2;

        double d_s = variable_rescaling[start + k];
        double d_t = variable_rescaling[start + k + 1];
        double d_st = sqrt(d_s * d_t);

        bool diag_uniform = true;
        for (int m = 0; m < k && diag_uniform; ++m)
        {
            if (variable_rescaling[start + m] != d_st)
                diag_uniform = false;
        }

        if (diag_uniform)
        {
            double sumsq = r_w * r_w;
            for (int m = 0; m < k; ++m)
            {
                double v_m = objective_vector[start + m] - dual_product[start + m];
                sumsq += v_m * v_m;
            }
            double r_norm = sqrt(sumsq);

            double v_factor, p_s, p_t;
            if (r_norm <= r_z)
            {
                v_factor = 0.0;
                p_s = r_s;
                p_t = r_t;
            }
            else if (r_norm <= -r_z)
            {
                v_factor = 1.0;
                p_s = 0.0;
                p_t = 0.0;
            }
            else
            {
                double scale = (r_z + r_norm) / (2.0 * r_norm);
                v_factor = 1.0 - scale;
                double w_new = scale * r_w;
                double z_new = scale * r_norm;
                p_s = (z_new + w_new) * INV_SQRT2;
                p_t = (z_new - w_new) * INV_SQRT2;
            }
            for (int m = 0; m < k; ++m)
            {
                double v_m = objective_vector[start + m] - dual_product[start + m];
                dual_residual[start + m] = v_m * v_factor * variable_rescaling[start + m];
            }
            dual_residual[start + k] = (r_s - p_s) * variable_rescaling[start + k];
            dual_residual[start + k + 1] = (r_t - p_t) * variable_rescaling[start + k + 1];
            continue;
        }

        double r_inv_sq = r_w * r_w;
        double r_pos_sq = r_w * r_w;
        for (int m = 0; m < k; ++m)
        {
            double e_m = d_st / variable_rescaling[start + m];
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            r_inv_sq += (rc_m / e_m) * (rc_m / e_m);
            r_pos_sq += (rc_m * e_m) * (rc_m * e_m);
        }
        double r_inv = sqrt(r_inv_sq);
        double r_pos = sqrt(r_pos_sq);

        if (r_inv <= r_z)
        {
            for (int m = 0; m < k; ++m)
                dual_residual[start + m] = 0.0;
            dual_residual[start + k] = 0.0;
            dual_residual[start + k + 1] = 0.0;
            continue;
        }
        if (r_pos <= -r_z)
        {
            for (int m = 0; m < k; ++m)
            {
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                dual_residual[start + m] = rc_m * variable_rescaling[start + m];
            }
            dual_residual[start + k] = r_s * variable_rescaling[start + k];
            dual_residual[start + k + 1] = r_t * variable_rescaling[start + k + 1];
            continue;
        }

        double lo, hi;
        bool z_pos = (r_z > 0.0);
        if (z_pos)
        {
            lo = 0.0;
            hi = 0.5 - 1e-14;
        }
        else
        {
            lo = 0.5 + 1e-14;
            hi = 1.0;
            for (int doubling = 0; doubling < 60; ++doubling)
            {
                double sum_hi = 0.0;
                for (int m = 0; m < k; ++m)
                {
                    double e_m = d_st / variable_rescaling[start + m];
                    double e_m2 = e_m * e_m;
                    double rc_m = objective_vector[start + m] - dual_product[start + m];
                    double tt = rc_m * e_m / (e_m2 + 2.0 * hi);
                    sum_hi += tt * tt;
                }
                double tw_hi = r_w / (1.0 + 2.0 * hi);
                sum_hi += tw_hi * tw_hi;
                double zt_hi = r_z / (1.0 - 2.0 * hi);
                double f_hi = sum_hi - zt_hi * zt_hi;
                if (f_hi > 0.0)
                    break;
                lo = hi;
                hi *= 2.0;
            }
        }

        double warm_lam = workspace[blk];
        if (warm_lam > lo && warm_lam < hi)
        {
            double sum_w = 0.0;
            for (int m = 0; m < k; ++m)
            {
                double e_m = d_st / variable_rescaling[start + m];
                double e_m2 = e_m * e_m;
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                double tt = rc_m * e_m / (e_m2 + 2.0 * warm_lam);
                sum_w += tt * tt;
            }
            double tw = r_w / (1.0 + 2.0 * warm_lam);
            sum_w += tw * tw;
            double zt = r_z / (1.0 - 2.0 * warm_lam);
            double f = sum_w - zt * zt;
            if (fabs(f) < 1e-12)
            {
                double p_w_w = r_w / (1.0 + 2.0 * warm_lam);
                double p_z_w = r_z / (1.0 - 2.0 * warm_lam);
                double p_s_w = (p_z_w + p_w_w) * INV_SQRT2;
                double p_t_w = (p_z_w - p_w_w) * INV_SQRT2;
                for (int m = 0; m < k; ++m)
                {
                    double e_m = d_st / variable_rescaling[start + m];
                    double e_m2 = e_m * e_m;
                    double rc_m = objective_vector[start + m] - dual_product[start + m];
                    double p_m = rc_m * e_m2 / (e_m2 + 2.0 * warm_lam);
                    dual_residual[start + m] = (rc_m - p_m) * variable_rescaling[start + m];
                }
                dual_residual[start + k] = (r_s - p_s_w) * variable_rescaling[start + k];
                dual_residual[start + k + 1] = (r_t - p_t_w) * variable_rescaling[start + k + 1];
                continue;
            }
            if (z_pos)
            {
                if (f > 0.0)
                    lo = warm_lam;
                else
                    hi = warm_lam;
            }
            else
            {
                if (f > 0.0)
                    hi = warm_lam;
                else
                    lo = warm_lam;
            }
        }

        for (int it = 0; it < 60; ++it)
        {
            double lam = 0.5 * (lo + hi);
            double sum = 0.0;
            for (int m = 0; m < k; ++m)
            {
                double e_m = d_st / variable_rescaling[start + m];
                double e_m2 = e_m * e_m;
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                double tt = rc_m * e_m / (e_m2 + 2.0 * lam);
                sum += tt * tt;
            }
            double tw = r_w / (1.0 + 2.0 * lam);
            sum += tw * tw;
            double zt = r_z / (1.0 - 2.0 * lam);
            double f = sum - zt * zt;
            if (z_pos)
            {
                if (f > 0.0)
                    lo = lam;
                else
                    hi = lam;
            }
            else
            {
                if (f > 0.0)
                    hi = lam;
                else
                    lo = lam;
            }
            if ((hi - lo) / (1.0 + hi + lo) < 1e-13)
                break;
        }
        double lam = 0.5 * (lo + hi);
        workspace[blk] = lam;

        double p_w = r_w / (1.0 + 2.0 * lam);
        double p_z = r_z / (1.0 - 2.0 * lam);
        double p_s = (p_z + p_w) * INV_SQRT2;
        double p_t = (p_z - p_w) * INV_SQRT2;

        for (int m = 0; m < k; ++m)
        {
            double e_m = d_st / variable_rescaling[start + m];
            double e_m2 = e_m * e_m;
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            double p_m = rc_m * e_m2 / (e_m2 + 2.0 * lam);
            dual_residual[start + m] = (rc_m - p_m) * variable_rescaling[start + m];
        }
        dual_residual[start + k] = (r_s - p_s) * variable_rescaling[start + k];
        dual_residual[start + k + 1] = (r_t - p_t) * variable_rescaling[start + k + 1];
    }
}

const cone_kernel_ops_t pdhcg_rsoc_cone_kernel_ops = {{rsoc_project}, {rsoc_diag}, {rsoc_residual}};
