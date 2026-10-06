/*
Copyright 2025-2026 Haihao Lu
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

#pragma once
#include <stdbool.h>
#ifdef __cplusplus
extern "C"
{
#endif
    void pdhcg_device_compute_and_rescale_reduced_cost(double *reduced_cost,
                                                       const double *objective,
                                                       const double *dual_product,
                                                       const double *variable_rescaling,
                                                       const double objective_vector_rescaling,
                                                       const double constraint_bound_rescaling,
                                                       int n_vars);
    void pdhcg_device_element_wise_mul(const double *__restrict__ A,
                                       const double *__restrict__ B,
                                       double *__restrict__ C,
                                       int n);
    void pdhcg_device_vector_sub(double *__restrict__ direction,
                                 const double *__restrict__ a,
                                 const double *__restrict__ b,
                                 int n);
    void pdhcg_device_vector_add(const double *__restrict__ a,
                                 const double *__restrict__ b,
                                 double *__restrict__ out,
                                 int n);
    void pdhcg_device_project_primal_onto_bounds(double *__restrict__ primal_solution,
                                                 const double *__restrict__ variable_lower_bound,
                                                 const double *__restrict__ variable_upper_bound,
                                                 int num_variables);
    void pdhcg_device_prepare_projected_gradient_point(double *__restrict__ projected_point,
                                                       const double *__restrict__ primal_solution,
                                                       const double *__restrict__ effective_objective,
                                                       const double *__restrict__ dual_product,
                                                       const double *__restrict__ variable_lower_bound,
                                                       const double *__restrict__ variable_upper_bound,
                                                       double step_size,
                                                       int num_variables);
    void pdhcg_device_augment_projected_gradient_residual(double *__restrict__ dual_residual,
                                                          const double *__restrict__ primal_solution,
                                                          const double *__restrict__ projected_point,
                                                          const double *__restrict__ variable_rescaling,
                                                          double step_size,
                                                          int num_variables);
    void pdhcg_device_compute_lp_next_pdhg_primal_solution(const double *current_primal,
                                                           double *reflected_primal,
                                                           const double *dual_product,
                                                           const double *objective,
                                                           const double *var_lb,
                                                           const double *var_ub,
                                                           int n,
                                                           double step_size);
    void pdhcg_device_compute_lp_next_pdhg_primal_solution_major(const double *current_primal,
                                                                 double *pdhg_primal,
                                                                 double *reflected_primal,
                                                                 const double *dual_product,
                                                                 const double *objective,
                                                                 const double *var_lb,
                                                                 const double *var_ub,
                                                                 int n,
                                                                 double step_size,
                                                                 double *dual_slack);
    void pdhcg_device_compute_diagonal_q_next_pdhg_primal_solution_major(const double *current_primal,
                                                                         double *pdhg_primal,
                                                                         double *reflected_primal,
                                                                         double *objective_product,
                                                                         const double *dual_product,
                                                                         const double *objective,
                                                                         const double *var_lb,
                                                                         const double *var_ub,
                                                                         int n,
                                                                         double step_size);
    void pdhcg_device_compute_diagonal_q_next_pdhg_primal_solution(const double *current_primal,
                                                                   double *reflected_primal,
                                                                   double *objective_product,
                                                                   const double *dual_product,
                                                                   const double *objective,
                                                                   const double *var_lb,
                                                                   const double *var_ub,
                                                                   int n,
                                                                   double step_size);
    void pdhcg_device_compute_next_pdhg_dual_solution(const double *current_dual,
                                                      double *reflected_dual,
                                                      const double *primal_product,
                                                      const double *affine_cone_offset,
                                                      const double *constraint_lower_bound,
                                                      const double *constraint_upper_bound,
                                                      int n,
                                                      double step_size);
    void pdhcg_device_compute_next_pdhg_dual_solution_major(const double *current_dual,
                                                            double *pdhg_dual,
                                                            double *reflected_dual,
                                                            const double *primal_product,
                                                            const double *affine_cone_offset,
                                                            const double *constraint_lower_bound,
                                                            const double *constraint_upper_bound,
                                                            int n,
                                                            double step_size);
    void pdhcg_device_prepare_constraint_dual_update(const double *current_dual,
                                                     const double *primal_product,
                                                     const double *affine_cone_offset,
                                                     const double *constraint_lower_bound,
                                                     const double *constraint_upper_bound,
                                                     double *projected_constraint_value,
                                                     int n,
                                                     double step_size);
    void pdhcg_device_finish_constraint_dual_update(const double *current_dual,
                                                    const double *primal_product,
                                                    const double *affine_cone_offset,
                                                    const double *projected_constraint_value,
                                                    double *pdhg_dual,
                                                    double *reflected_dual,
                                                    int n,
                                                    double step_size);
    void pdhcg_device_halpern_update(const double *initial_primal,
                                     double *current_primal,
                                     const double *reflected_primal,
                                     const double *initial_dual,
                                     double *current_dual,
                                     const double *reflected_dual,
                                     int n_vars,
                                     int n_cons,
                                     double weight,
                                     double reflection_coeff);
    void pdhcg_device_rescale_solution(double *primal_solution,
                                       double *dual_solution,
                                       const double *variable_rescaling,
                                       const double *constraint_rescaling,
                                       const double objective_vector_rescaling,
                                       const double constraint_bound_rescaling,
                                       int n_vars,
                                       int n_cons);
    void pdhcg_device_compute_delta_solution(const double *initial_primal,
                                             const double *pdhg_primal,
                                             double *delta_primal,
                                             const double *initial_dual,
                                             const double *pdhg_dual,
                                             double *delta_dual,
                                             int n_vars,
                                             int n_cons);
    void pdhcg_device_primal_gradient_descent(const double *dual_product,
                                              const double *current_primal_solution,
                                              double *reflected_primal,
                                              const double *objective_vector,
                                              const double *objective_product,
                                              const double *var_lb,
                                              const double *var_ub,
                                              const double stepsize,
                                              const int n_vars);
    void pdhcg_device_primal_gradient_descent_kernel_major(const double *dual_product,
                                                           const double *current_primal_solution,
                                                           double *reflected_primal,
                                                           double *pdhg_primal_solution,
                                                           const double *objective_vector,
                                                           const double *objective_product,
                                                           const double *var_lb,
                                                           const double *var_ub,
                                                           const double stepsize,
                                                           const int n_vars);
    void pdhcg_device_compute_bb_alpha_safeguard(const double *d_norm_gtg, const double *d_tmp, double *d_alpha);
    void pdhcg_device_compute_bb_alpha_M(const double *d_stMs, const double *d_tmp, double *d_alpha);
    void pdhcg_device_scalar_sqrt_copy(const double *src, double *dst);
    void pdhcg_device_compute_csr_diag(
        const int *row_ptr, const int *col_ind, const double *val, double *diag, int num_rows);
    void pdhcg_device_compute_csr_row_sq_norm(const int *row_ptr, const double *val, double *out, int num_rows);
    void pdhcg_device_element_wise_mul_inplace(double *__restrict__ x, const double *__restrict__ d, int n);
    void pdhcg_device_compute_csr_row_sq_norm_weighted(
        const int *row_ptr, const int *col_ind, const double *val, const double *weights, double *out, int num_rows);
    void pdhcg_device_compute_csr_row_quad_form_dense(const int *row_ptr,
                                                      const int *col_ind,
                                                      const double *val,
                                                      const double *D_dense,
                                                      int rank,
                                                      double *out,
                                                      int num_rows);
    void pdhcg_device_refresh_inner_precond(
        const double *diag_h_static, double inv_tau, double *m_diag, double *m_inv, int n_vars);
    void pdhcg_device_primal_gradient_descent_kernel_bb_init(const double *dual_product,
                                                             double *gradient,
                                                             double *direction,
                                                             const double *current_primal_solution,
                                                             double *pdhg_primal_solution,
                                                             const double *objective_vector,
                                                             const double *objective_product,
                                                             const double *var_lb,
                                                             const double *var_ub,
                                                             const double stepsize,
                                                             const int n_vars);
    void pdhcg_device_primal_bb_update_gradient(double *pdhg_primal_solution,
                                                const double *current_primal_solution,
                                                const double *objective_vector,
                                                const double *dual_product,
                                                const double *objective_product,
                                                double *gradient,
                                                double *delta_gradient,
                                                const double inv_step_size,
                                                const int n_vars);
    void pdhcg_device_primal_bb_update_direction(double *pdhg_primal_solution,
                                                 const double *gradient,
                                                 double *direction,
                                                 const double *var_lb,
                                                 const double *var_ub,
                                                 const double *d_alpha,
                                                 const int n_vars);
    void pdhcg_device_primal_bb_final(const double *current_primal_solution,
                                      const double *pdhg_primal_solution,
                                      double *reflected_primal_solution,
                                      const int n_vars);
    void pdhcg_device_primal_gradient_descent_kernel_bb_init_precond(const double *dual_product,
                                                                     double *gradient,
                                                                     double *direction,
                                                                     const double *current_primal_solution,
                                                                     double *pdhg_primal_solution,
                                                                     const double *objective_vector,
                                                                     const double *objective_product,
                                                                     const double *var_lb,
                                                                     const double *var_ub,
                                                                     const double *m_inv,
                                                                     const double stepsize,
                                                                     const int n_vars);
    void pdhcg_device_primal_bb_update_direction_kernel_precond(double *pdhg_primal_solution,
                                                                const double *gradient,
                                                                double *direction,
                                                                const double *var_lb,
                                                                const double *var_ub,
                                                                const double *m_inv,
                                                                const double *d_alpha,
                                                                const int n_vars);
    void pdhcg_device_compute_lp_residual(double *primal_residual,
                                          const double *primal_product,
                                          const double *affine_cone_offset,
                                          const double *constraint_lower_bound,
                                          const double *constraint_upper_bound,
                                          const double *dual_solution,
                                          double *dual_residual,
                                          const double *dual_product,
                                          const double *dual_slack,
                                          const double *objective_vector,
                                          const double *constraint_rescaling,
                                          const double *variable_rescaling,
                                          double *affine_dual_membership,
                                          double *dual_obj_contribution,
                                          const double *const_lb_finite,
                                          const double *const_ub_finite,
                                          bool defer_constraint_projection,
                                          int num_constraints,
                                          int num_variables);
    void pdhcg_device_compute_qp_residual(double *primal_residual,
                                          const double *primal_product,
                                          const double *affine_cone_offset,
                                          const double *primal_obj_product,
                                          const double *primal_solution,
                                          const double *constraint_lower_bound,
                                          const double *constraint_upper_bound,
                                          const double *variable_lower_bound,
                                          const double *variable_upper_bound,
                                          const double *dual_solution,
                                          double *dual_residual,
                                          const double *dual_product,
                                          double *dual_slack,
                                          const double *objective_vector,
                                          const double *constraint_rescaling,
                                          const double *variable_rescaling,
                                          double *affine_dual_membership,
                                          double *dual_obj_contribution,
                                          const double *const_lb_finite,
                                          const double *const_ub_finite,
                                          const double step_size,
                                          bool defer_constraint_projection,
                                          int num_constraints,
                                          int num_variables);
    void pdhcg_device_recover_primal_obj_dual_product(double *dual_product,
                                                      double *primal_obj_product,
                                                      const double *variable_rescaling,
                                                      int num_variables);
    void pdhcg_device_prepare_infeasibility_projection(double *projection,
                                                       const double *dual_product,
                                                       const double *primal,
                                                       const double *lower,
                                                       const double *upper,
                                                       const unsigned char *cone_type,
                                                       int n);
    void pdhcg_device_finish_infeasibility_projection(double *dual_product,
                                                      double *projection,
                                                      const double *primal,
                                                      const double *lower,
                                                      const double *upper,
                                                      const unsigned char *cone_type,
                                                      const double *rescaling,
                                                      int n);
    void pdhcg_device_projection_residual(double *projection, const double *point, const double *rescaling, int n);
    void pdhcg_device_primal_infeasibility_project(double *primal_ray_estimate,
                                                   const double *variable_lower_bound,
                                                   const double *variable_upper_bound,
                                                   const unsigned char *cone_type,
                                                   int num_variables);
    void pdhcg_device_dual_infeasibility_project(double *dual_ray_estimate,
                                                 const double *constraint_lower_bound,
                                                 const double *constraint_upper_bound,
                                                 int num_constraints);
    void pdhcg_device_compute_primal_infeasibility(const double *primal_product,
                                                   const double *const_lb,
                                                   const double *const_ub,
                                                   int num_constraints,
                                                   double *primal_infeasibility,
                                                   const double *constraint_rescaling);
    void pdhcg_device_compute_dual_infeasibility(const double *dual_product,
                                                 const double *var_lb,
                                                 const double *var_ub,
                                                 int num_variables,
                                                 double *dual_infeasibility,
                                                 const double *variable_rescaling);
    void
    pdhcg_device_dual_solution_dual_objective_contribution(const double *constraint_lower_bound_finite_val,
                                                           const double *constraint_upper_bound_finite_val,
                                                           const double *affine_cone_offset,
                                                           const double *dual_solution,
                                                           int num_constraints,
                                                           double *dual_objective_dual_solution_contribution_array);
    void pdhcg_device_dual_objective_dual_slack_contribution_array(const double *dual_slack,
                                                                   double *dual_objective_dual_slack_contribution_array,
                                                                   const double *variable_lower_bound_finite_val,
                                                                   const double *variable_upper_bound_finite_val,
                                                                   int num_variables);
    void pdhcg_device_compute_and_rescale_reduced_cost_qp(double *__restrict__ reduced_cost,
                                                          const double *__restrict__ objective,
                                                          const double *__restrict__ quadratic_product,
                                                          const double *__restrict__ dual_product,
                                                          const double *__restrict__ variable_rescaling,
                                                          const double objective_vector_rescaling,
                                                          const double constraint_bound_rescaling,
                                                          const double *__restrict__ variable_lower_bound,
                                                          const double *__restrict__ variable_upper_bound,
                                                          int n_vars);
#ifdef __cplusplus
}
#endif
