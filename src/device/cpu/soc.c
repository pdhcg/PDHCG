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

static inline void project_standard_soc_section_serial(double *point,
                                                       const double *rescaling,
                                                       const double *q_diag,
                                                       double tau,
                                                       double *warm_start,
                                                       int start,
                                                       int k,
                                                       const char *is_fixed)
{
    int u_length = k + 1;
    int z_index = start + u_length;
    bool fixed_z = is_fixed[z_index] != 0;
    double fixed_norm2 = 0.0;
    double free_norm2 = 0.0;
    double polar_norm2 = 0.0;
    double max_omega = 0.0;
    int free_count = 0;
    for (int slot = 0; slot < u_length; ++slot)
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

    double z_input = cone_section_actual(point, rescaling, z_index);
    if (fixed_z)
    {
        double radius2 = fmax(0.0, z_input * z_input - fixed_norm2);
        if (free_count == 0 || free_norm2 <= radius2)
            return;
        if (!(radius2 > 0.0))
        {
            for (int slot = 0; slot < u_length; ++slot)
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
                for (int slot = 0; slot < u_length; ++slot)
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
            for (int slot = 0; slot < u_length; ++slot)
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
        for (int slot = 0; slot < u_length; ++slot)
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

    double total_norm2 = fixed_norm2 + free_norm2;
    if (z_input >= 0.0 && total_norm2 <= z_input * z_input)
        return;
    if (free_count == 0)
    {
        double projected_z = fmax(z_input, sqrt(fixed_norm2));
        point[z_index] = projected_z * rescaling[z_index];
        return;
    }

    double omega_z = cone_section_weight(rescaling, q_diag, tau, z_index);
    if (fixed_norm2 == 0.0)
    {
        if (-omega_z * z_input >= sqrt(polar_norm2))
        {
            for (int slot = 0; slot < u_length; ++slot)
                if (!is_fixed[start + slot])
                    point[start + slot] = 0.0;
            point[z_index] = 0.0;
            return;
        }
    }

    double lambda;
    if (z_input == 0.0)
    {
        lambda = omega_z;
        double norm2 = fixed_norm2;
        for (int slot = 0; slot < u_length; ++slot)
        {
            int index = start + slot;
            if (is_fixed[index])
                continue;
            double omega = cone_section_weight(rescaling, q_diag, tau, index);
            double value = cone_section_actual(point, rescaling, index) * omega / (omega + lambda);
            norm2 += value * value;
        }
        point[z_index] = sqrt(norm2) * rescaling[z_index];
    }
    else
    {
        bool lower_branch = z_input > 0.0;
        double lo;
        double hi;
        if (lower_branch)
        {
            lo = 0.0;
            hi = omega_z * (1.0 - 1e-14);
        }
        else
        {
            lo = omega_z * (1.0 + 1e-14);
            hi = cone_section_negative_soc_upper(omega_z, -omega_z * z_input, fixed_norm2, polar_norm2, max_omega);
            if (!(hi > lo) || !isfinite(hi))
            {
                hi = 2.0 * omega_z;
                for (int expansion = 0; expansion < 100; ++expansion)
                {
                    double norm2 = fixed_norm2;
                    for (int slot = 0; slot < u_length; ++slot)
                    {
                        int index = start + slot;
                        if (is_fixed[index])
                            continue;
                        double omega = cone_section_weight(rescaling, q_diag, tau, index);
                        double value = cone_section_actual(point, rescaling, index) * omega / (omega + hi);
                        norm2 += value * value;
                    }
                    double z = omega_z * z_input / (omega_z - hi);
                    if (norm2 >= z * z)
                        break;
                    hi *= 2.0;
                }
            }
        }

        if (warm_start && *warm_start > lo && *warm_start < hi && isfinite(*warm_start))
        {
            double norm2 = fixed_norm2;
            for (int slot = 0; slot < u_length; ++slot)
            {
                int index = start + slot;
                if (is_fixed[index])
                    continue;
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                double value = cone_section_actual(point, rescaling, index) * omega / (omega + *warm_start);
                norm2 += value * value;
            }
            double z = omega_z * z_input / (omega_z - *warm_start);
            double f = norm2 - z * z;
            if ((lower_branch && f > 0.0) || (!lower_branch && f < 0.0))
                lo = *warm_start;
            else
                hi = *warm_start;
        }

        for (int iteration = 0; iteration < 80; ++iteration)
        {
            double trial = 0.5 * (lo + hi);
            double norm2 = fixed_norm2;
            for (int slot = 0; slot < u_length; ++slot)
            {
                int index = start + slot;
                if (is_fixed[index])
                    continue;
                double omega = cone_section_weight(rescaling, q_diag, tau, index);
                double value = cone_section_actual(point, rescaling, index) * omega / (omega + trial);
                norm2 += value * value;
            }
            double z = omega_z * z_input / (omega_z - trial);
            double f = norm2 - z * z;
            if ((lower_branch && f > 0.0) || (!lower_branch && f < 0.0))
                lo = trial;
            else
                hi = trial;
            if ((hi - lo) <= 1e-13 * (1.0 + hi + lo))
                break;
        }
        lambda = 0.5 * (lo + hi);
        point[z_index] *= omega_z / (omega_z - lambda);
    }

    if (warm_start)
        *warm_start = lambda;
    for (int slot = 0; slot < u_length; ++slot)
    {
        int index = start + slot;
        if (!is_fixed[index])
        {
            double omega = cone_section_weight(rescaling, q_diag, tau, index);
            point[index] *= omega / (omega + lambda);
        }
    }
}

static void soc_project(double *__restrict__ primal_solution,
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
        int start = start_idx[blk];
        int k = v_dim[blk];
        if (cone_section_has_fixed(is_fixed, start, k + 2))
        {
            project_standard_soc_section_serial(
                primal_solution, variable_rescaling, NULL, 0.0, workspace + blk, start, k, is_fixed);
            continue;
        }
        double *v = primal_solution + start;
        double *wptr = primal_solution + start + k;
        double *zptr = primal_solution + start + k + 1;

        double w = *wptr;
        double z = *zptr;

        double d_z = variable_rescaling[start + k + 1];
        double dhat_w = variable_rescaling[start + k] / d_z;
        double dhat_w2 = dhat_w * dhat_w;

        bool diag_uniform = (dhat_w == 1.0);
        for (int m = 0; m < k && diag_uniform; ++m)
        {
            if (variable_rescaling[start + m] != d_z)
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
                *wptr = 0.0;
                *zptr = 0.0;
                continue;
            }
            double scale = (z + r) / (2.0 * r);
            for (int m = 0; m < k; ++m)
                v[m] *= scale;
            *wptr = scale * w;
            *zptr = scale * r;
            continue;
        }

        double r_inv_sq = (w / dhat_w) * (w / dhat_w);
        double r_pos_sq = (w * dhat_w) * (w * dhat_w);
        for (int m = 0; m < k; ++m)
        {
            double dh = variable_rescaling[start + m] / d_z;
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
            *wptr = 0.0;
            *zptr = 0.0;
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
                    double dh = variable_rescaling[start + m] / d_z;
                    double dh2 = dh * dh;
                    double t = v[m] * dh / (dh2 + 2.0 * hi);
                    sum_hi += t * t;
                }
                double tw_hi = w * dhat_w / (dhat_w2 + 2.0 * hi);
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
                double dh = variable_rescaling[start + m] / d_z;
                double dh2 = dh * dh;
                double t = v[m] * dh / (dh2 + 2.0 * warm_lam);
                sum_w += t * t;
            }
            double tw = w * dhat_w / (dhat_w2 + 2.0 * warm_lam);
            sum_w += tw * tw;
            double zt = z / (1.0 - 2.0 * warm_lam);
            double f = sum_w - zt * zt;
            if (fabs(f) < 1e-12)
            {
                *zptr = z / (1.0 - 2.0 * warm_lam);
                *wptr = w * dhat_w2 / (dhat_w2 + 2.0 * warm_lam);
                for (int m = 0; m < k; ++m)
                {
                    double dh = variable_rescaling[start + m] / d_z;
                    double dh2 = dh * dh;
                    v[m] = v[m] * dh2 / (dh2 + 2.0 * warm_lam);
                }
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
                double dh = variable_rescaling[start + m] / d_z;
                double dh2 = dh * dh;
                double t = v[m] * dh / (dh2 + 2.0 * lam);
                sum += t * t;
            }
            double tw = w * dhat_w / (dhat_w2 + 2.0 * lam);
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

        *zptr = z / (1.0 - 2.0 * lam);
        *wptr = w * dhat_w2 / (dhat_w2 + 2.0 * lam);
        for (int m = 0; m < k; ++m)
        {
            double dh = variable_rescaling[start + m] / d_z;
            double dh2 = dh * dh;
            v[m] = v[m] * dh2 / (dh2 + 2.0 * lam);
        }
    }
}

static void soc_diag(double *__restrict__ pdhg_primal,
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
        int start = start_idx[blk];
        int k = v_dim[blk];
        int w_off = start + k;
        int z_off = start + k + 1;

        if (cone_section_has_fixed(is_fixed, start, k + 2))
        {
            project_standard_soc_section_serial(
                pdhg_primal, variable_rescaling, Q_diag, tau, workspace + blk, start, k, is_fixed);
            for (int slot = 0; slot < k + 2; ++slot)
            {
                int index = start + slot;
                reflected_primal[index] = 2.0 * pdhg_primal[index] - current_primal[index];
            }
            continue;
        }

        double r_w = pdhg_primal[w_off];
        double r_z = pdhg_primal[z_off];

        double d_z = variable_rescaling[z_off];
        double w_w = 1.0 + tau * Q_diag[w_off];
        double w_z = 1.0 + tau * Q_diag[z_off];
        double sqrt_w_w = sqrt(w_w);
        double sqrt_w_z = sqrt(w_z);
        double e_z = sqrt_w_z * d_z;
        double e_w = sqrt_w_w * variable_rescaling[w_off];
        double eh_w = e_w / e_z;
        double eh_w2 = eh_w * eh_w;

        double r_inv_sq = w_w * (r_w / eh_w) * (r_w / eh_w);
        double r_pos_sq = w_w * (r_w * eh_w) * (r_w * eh_w);
        for (int m = 0; m < k; ++m)
        {
            double w_m = 1.0 + tau * Q_diag[start + m];
            double e_m = sqrt(w_m) * variable_rescaling[start + m];
            double eh_m = e_m / e_z;
            double r_m = pdhg_primal[start + m];
            r_inv_sq += w_m * (r_m / eh_m) * (r_m / eh_m);
            r_pos_sq += w_m * (r_m * eh_m) * (r_m * eh_m);
        }
        double w_z_r_z_sq = w_z * r_z * r_z;

        if (r_inv_sq <= w_z_r_z_sq && r_z >= 0.0)
        {
            for (int m = 0; m < k; ++m)
            {
                int idx = start + m;
                reflected_primal[idx] = 2.0 * pdhg_primal[idx] - current_primal[idx];
            }
            reflected_primal[w_off] = 2.0 * r_w - current_primal[w_off];
            reflected_primal[z_off] = 2.0 * r_z - current_primal[z_off];
            continue;
        }

        if (r_pos_sq <= w_z_r_z_sq && r_z <= 0.0)
        {
            for (int m = 0; m < k; ++m)
            {
                int idx = start + m;
                pdhg_primal[idx] = 0.0;
                reflected_primal[idx] = -current_primal[idx];
            }
            pdhg_primal[w_off] = 0.0;
            pdhg_primal[z_off] = 0.0;
            reflected_primal[w_off] = -current_primal[w_off];
            reflected_primal[z_off] = -current_primal[z_off];
            continue;
        }

        /* Fast path: no Q on cone slots and uniform d_v = d_z (LP-style symmetric case). */
        if (Q_diag[w_off] == 0.0 && Q_diag[z_off] == 0.0)
        {
            bool no_cone_Q = true;
            bool d_uniform = (variable_rescaling[w_off] == d_z);
            for (int m = 0; m < k; ++m)
            {
                if (Q_diag[start + m] != 0.0)
                {
                    no_cone_Q = false;
                    break;
                }
                if (variable_rescaling[start + m] != d_z)
                {
                    d_uniform = false;
                    break;
                }
            }
            if (no_cone_Q && d_uniform)
            {
                double sumsq = r_w * r_w;
                for (int m = 0; m < k; ++m)
                {
                    double vm = pdhg_primal[start + m];
                    sumsq += vm * vm;
                }
                double rnorm = sqrt(sumsq);
                /* in-cone (rnorm <= r_z, r_z >= 0) and at-origin (rnorm <= -r_z, r_z <= 0) handled above */
                double scale = (r_z + rnorm) / (2.0 * rnorm);
                for (int m = 0; m < k; ++m)
                {
                    double v_new = scale * pdhg_primal[start + m];
                    pdhg_primal[start + m] = v_new;
                    int idx = start + m;
                    reflected_primal[idx] = 2.0 * v_new - current_primal[idx];
                }
                double w_new = scale * r_w;
                double z_new = scale * rnorm;
                pdhg_primal[w_off] = w_new;
                pdhg_primal[z_off] = z_new;
                reflected_primal[w_off] = 2.0 * w_new - current_primal[w_off];
                reflected_primal[z_off] = 2.0 * z_new - current_primal[z_off];
                continue;
            }
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
                    double w_m = 1.0 + tau * Q_diag[start + m];
                    double e_m = sqrt(w_m) * variable_rescaling[start + m];
                    double eh_m = e_m / e_z;
                    double eh_m2 = eh_m * eh_m;
                    double r_m = pdhg_primal[start + m];
                    double t = sqrt(w_m) * r_m * eh_m / (eh_m2 + 2.0 * hi);
                    sum_hi += t * t;
                }
                double tw_hi = sqrt_w_w * r_w * eh_w / (eh_w2 + 2.0 * hi);
                sum_hi += tw_hi * tw_hi;
                double tz_hi = sqrt_w_z * r_z / (1.0 - 2.0 * hi);
                double f_hi = sum_hi - tz_hi * tz_hi;
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
                double w_m = 1.0 + tau * Q_diag[start + m];
                double e_m = sqrt(w_m) * variable_rescaling[start + m];
                double eh_m = e_m / e_z;
                double eh_m2 = eh_m * eh_m;
                double r_m = pdhg_primal[start + m];
                double t = sqrt(w_m) * r_m * eh_m / (eh_m2 + 2.0 * warm_lam);
                sum_w += t * t;
            }
            double tw = sqrt_w_w * r_w * eh_w / (eh_w2 + 2.0 * warm_lam);
            sum_w += tw * tw;
            double tz = sqrt_w_z * r_z / (1.0 - 2.0 * warm_lam);
            double f = sum_w - tz * tz;
            if (fabs(f) < 1e-12)
            {
                double new_z = r_z / (1.0 - 2.0 * warm_lam);
                double new_w = r_w * eh_w2 / (eh_w2 + 2.0 * warm_lam);
                pdhg_primal[z_off] = new_z;
                pdhg_primal[w_off] = new_w;
                reflected_primal[z_off] = 2.0 * new_z - current_primal[z_off];
                reflected_primal[w_off] = 2.0 * new_w - current_primal[w_off];
                for (int m = 0; m < k; ++m)
                {
                    int idx = start + m;
                    double w_m = 1.0 + tau * Q_diag[idx];
                    double e_m = sqrt(w_m) * variable_rescaling[idx];
                    double eh_m = e_m / e_z;
                    double eh_m2 = eh_m * eh_m;
                    double r_m = pdhg_primal[idx];
                    double new_m = r_m * eh_m2 / (eh_m2 + 2.0 * warm_lam);
                    pdhg_primal[idx] = new_m;
                    reflected_primal[idx] = 2.0 * new_m - current_primal[idx];
                }
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
                double w_m = 1.0 + tau * Q_diag[start + m];
                double e_m = sqrt(w_m) * variable_rescaling[start + m];
                double eh_m = e_m / e_z;
                double eh_m2 = eh_m * eh_m;
                double r_m = pdhg_primal[start + m];
                double t = sqrt(w_m) * r_m * eh_m / (eh_m2 + 2.0 * lam);
                sum += t * t;
            }
            double tw = sqrt_w_w * r_w * eh_w / (eh_w2 + 2.0 * lam);
            sum += tw * tw;
            double tz = sqrt_w_z * r_z / (1.0 - 2.0 * lam);
            double f = sum - tz * tz;
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

        double new_z = r_z / (1.0 - 2.0 * lam);
        double new_w = r_w * eh_w2 / (eh_w2 + 2.0 * lam);
        pdhg_primal[z_off] = new_z;
        pdhg_primal[w_off] = new_w;
        reflected_primal[z_off] = 2.0 * new_z - current_primal[z_off];
        reflected_primal[w_off] = 2.0 * new_w - current_primal[w_off];
        for (int m = 0; m < k; ++m)
        {
            int idx = start + m;
            double w_m = 1.0 + tau * Q_diag[idx];
            double e_m = sqrt(w_m) * variable_rescaling[idx];
            double eh_m = e_m / e_z;
            double eh_m2 = eh_m * eh_m;
            double r_m = pdhg_primal[idx];
            double new_m = r_m * eh_m2 / (eh_m2 + 2.0 * lam);
            pdhg_primal[idx] = new_m;
            reflected_primal[idx] = 2.0 * new_m - current_primal[idx];
        }
    }
}

static void soc_residual(double *__restrict__ dual_residual,
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
            project_standard_soc_section_serial(
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

        double r_w = objective_vector[start + k] - dual_product[start + k];
        double r_z = objective_vector[start + k + 1] - dual_product[start + k + 1];

        double d_z = variable_rescaling[start + k + 1];
        double e_w = d_z / variable_rescaling[start + k];
        double e_w2 = e_w * e_w;

        bool diag_uniform = (e_w == 1.0);
        for (int m = 0; m < k && diag_uniform; ++m)
        {
            if (variable_rescaling[start + m] != d_z)
                diag_uniform = false;
        }

        if (diag_uniform)
        {
            double sumsq = r_w * r_w;
            for (int m = 0; m < k; ++m)
            {
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                sumsq += rc_m * rc_m;
            }
            double r = sqrt(sumsq);
            double v_factor, p_w, p_z;
            if (r <= r_z)
            {
                v_factor = 0.0;
                p_w = r_w;
                p_z = r_z;
            }
            else if (r <= -r_z)
            {
                v_factor = 1.0;
                p_w = 0.0;
                p_z = 0.0;
            }
            else
            {
                double scale = (r_z + r) / (2.0 * r);
                v_factor = 1.0 - scale;
                p_w = scale * r_w;
                p_z = scale * r;
            }
            for (int m = 0; m < k; ++m)
            {
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                dual_residual[start + m] = rc_m * v_factor * variable_rescaling[start + m];
            }
            dual_residual[start + k] = (r_w - p_w) * variable_rescaling[start + k];
            dual_residual[start + k + 1] = (r_z - p_z) * variable_rescaling[start + k + 1];
            continue;
        }

        double r_inv_sq = (r_w / e_w) * (r_w / e_w);
        double r_pos_sq = (r_w * e_w) * (r_w * e_w);
        for (int m = 0; m < k; ++m)
        {
            double e_m = d_z / variable_rescaling[start + m];
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
            dual_residual[start + k] = r_w * variable_rescaling[start + k];
            dual_residual[start + k + 1] = r_z * variable_rescaling[start + k + 1];
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
                    double e_m = d_z / variable_rescaling[start + m];
                    double e_m2 = e_m * e_m;
                    double rc_m = objective_vector[start + m] - dual_product[start + m];
                    double t = rc_m * e_m / (e_m2 + 2.0 * hi);
                    sum_hi += t * t;
                }
                double tw_hi = r_w * e_w / (e_w2 + 2.0 * hi);
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
                double e_m = d_z / variable_rescaling[start + m];
                double e_m2 = e_m * e_m;
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                double t = rc_m * e_m / (e_m2 + 2.0 * warm_lam);
                sum_w += t * t;
            }
            double tw = r_w * e_w / (e_w2 + 2.0 * warm_lam);
            sum_w += tw * tw;
            double zt = r_z / (1.0 - 2.0 * warm_lam);
            double f = sum_w - zt * zt;
            if (fabs(f) < 1e-12)
            {
                double p_z_w = r_z / (1.0 - 2.0 * warm_lam);
                double p_w_w = r_w * e_w2 / (e_w2 + 2.0 * warm_lam);
                for (int m = 0; m < k; ++m)
                {
                    double e_m = d_z / variable_rescaling[start + m];
                    double e_m2 = e_m * e_m;
                    double rc_m = objective_vector[start + m] - dual_product[start + m];
                    double p_m = rc_m * e_m2 / (e_m2 + 2.0 * warm_lam);
                    dual_residual[start + m] = (rc_m - p_m) * variable_rescaling[start + m];
                }
                dual_residual[start + k] = (r_w - p_w_w) * variable_rescaling[start + k];
                dual_residual[start + k + 1] = (r_z - p_z_w) * variable_rescaling[start + k + 1];
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
                double e_m = d_z / variable_rescaling[start + m];
                double e_m2 = e_m * e_m;
                double rc_m = objective_vector[start + m] - dual_product[start + m];
                double t = rc_m * e_m / (e_m2 + 2.0 * lam);
                sum += t * t;
            }
            double tw = r_w * e_w / (e_w2 + 2.0 * lam);
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

        double p_z = r_z / (1.0 - 2.0 * lam);
        double p_w = r_w * e_w2 / (e_w2 + 2.0 * lam);

        for (int m = 0; m < k; ++m)
        {
            double e_m = d_z / variable_rescaling[start + m];
            double e_m2 = e_m * e_m;
            double rc_m = objective_vector[start + m] - dual_product[start + m];
            double p_m = rc_m * e_m2 / (e_m2 + 2.0 * lam);
            dual_residual[start + m] = (rc_m - p_m) * variable_rescaling[start + m];
        }
        dual_residual[start + k] = (r_w - p_w) * variable_rescaling[start + k];
        dual_residual[start + k + 1] = (r_z - p_z) * variable_rescaling[start + k + 1];
    }
}

const cone_kernel_ops_t pdhcg_soc_cone_kernel_ops = {{soc_project}, {soc_diag}, {soc_residual}};
