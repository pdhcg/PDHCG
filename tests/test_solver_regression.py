# SPDX-License-Identifier: Apache-2.0

"""Small deterministic problems with independently known solutions/statuses."""

import numpy as np
import pytest
import scipy.sparse as sp

from pdhcg import Model


def configure(model):
    model.setParams(
        Presolve=False, LogLevel=0, TimeLimit=15.0,
        OptimalityTol=1e-7, FeasibilityTol=1e-7,
    )
    return model


@pytest.mark.parametrize("mode", ["inner", "linearized"])
def test_sparse_constrained_qp(solver_device, mode):
    # Q is SPD and non-diagonal. Choose c using a known KKT point:
    # x* = 1, Ax* = 2, row multipliers = 1, Qx* + c = A^T y*.
    n = 32
    q = sp.diags([-np.ones(n - 1), 4 * np.ones(n), -np.ones(n - 1)],
                 [-1, 0, 1], format="csr")
    a = sp.hstack([sp.eye(n // 2), sp.eye(n // 2)], format="csr")
    expected_x = np.ones(n)
    expected_y = np.ones(n // 2)
    c = np.asarray(a.T @ expected_y - q @ expected_x)
    model = configure(Model(
        objective_vector=c, objective_matrix=q, constraint_matrix=a,
        constraint_lower_bound=2 * np.ones(n // 2),
        variable_lower_bound=np.zeros(n),
    ))
    model.setParam("non_diagonal_quadratic_mode", mode)
    model.optimize(device=solver_device)

    assert model.Status == "OPTIMAL"
    np.testing.assert_allclose(model.X, expected_x, rtol=0, atol=5e-5)
    np.testing.assert_allclose(model.Pi, expected_y, rtol=0, atol=5e-5)
    assert np.min(a @ model.X - 2) >= -5e-5
    assert np.min(model.X) >= -5e-5
    np.testing.assert_allclose(q @ model.X + c - a.T @ model.Pi, 0, atol=5e-5)
    expected_objective = 0.5 * expected_x @ q @ expected_x + c @ expected_x
    actual_objective = 0.5 * model.X @ q @ model.X + c @ model.X
    assert model.ObjVal == pytest.approx(expected_objective, abs=5e-5)
    assert model.ObjVal == pytest.approx(actual_objective, abs=1e-9)
    assert model.RelPrimalResidual <= 1e-7
    assert model.RelDualResidual <= 1e-7
    assert model.RelGap <= 1e-7


def infeasible_lp():
    # Individually valid rows, jointly inconsistent: x >= 1 and x <= 0.
    return configure(Model(
        objective_vector=[0.0], constraint_matrix=sp.csr_matrix([[1.0], [1.0]]),
        constraint_lower_bound=[1.0, -np.inf], constraint_upper_bound=[np.inf, 0.0],
    ))


def test_infeasible_lp(solver_device):
    model = infeasible_lp()
    model.setParams(InfeasibleTol=1e-6, iteration_limit=2000)
    model.optimize(device=solver_device)
    assert model.Status == "PRIMAL_INFEASIBLE"
    assert model.DualRayObj > 0
    assert model.MaxDualRayInfeas / model.DualRayObj <= 1e-6


@pytest.mark.parametrize("tolerance", [None, 1e-12], ids=["default", "strict"])
def test_infeasible_lp_certificate_tolerance(solver_device, tolerance):
    model = infeasible_lp()
    model.setParam("iteration_limit", 2000)
    if tolerance is not None:
        model.setParam("InfeasibleTol", tolerance)
    else:
        assert model.getParam("InfeasibleTol") == 1e-10
    model.optimize(device=solver_device)
    assert model.Status == "PRIMAL_INFEASIBLE"
    assert model.DualRayObj > 0
    assert model.MaxDualRayInfeas / model.DualRayObj <= model.getParam("InfeasibleTol")


def test_unbounded_lp(solver_device):
    # min -x, x >= 0.
    model = configure(Model(
        objective_vector=[-1.0], constraint_matrix=sp.csr_matrix([[1.0]]),
        constraint_lower_bound=[0.0],
    ))
    model.setParam("iteration_limit", 2000)
    model.optimize(device=solver_device)
    assert model.Status == "DUAL_INFEASIBLE"
    assert model.PrimalRayLinObj < 0
    assert model.MaxPrimalRayInfeas / -model.PrimalRayLinObj <= model.getParam("InfeasibleTol")


def test_scaled_feasible_lp(solver_device):
    # Both global rescaling factors are far from one. This feasible bounded LP
    # must not be rejected by a certificate measured in the scaled units.
    model = configure(Model(
        objective_vector=[1e4], constraint_matrix=sp.csr_matrix([[1.0]]),
        constraint_lower_bound=[1e4], constraint_upper_bound=[2e4],
    ))
    model.setParam("iteration_limit", 10000)
    model.optimize(device=solver_device)
    assert model.Status == "OPTIMAL"
    np.testing.assert_allclose(model.X, [1e4], rtol=1e-6, atol=0)
    assert model.ObjVal == pytest.approx(1e8, rel=1e-6)
    assert model.RelPrimalResidual <= 1e-7
    assert model.RelDualResidual <= 1e-7
    assert model.RelGap <= 1e-7


def test_infeasible_qp(solver_device):
    # Adding convex curvature cannot make the contradictory rows feasible.
    model = configure(Model(
        objective_vector=[0.0], objective_matrix=sp.csr_matrix([[1.0]]),
        constraint_matrix=sp.csr_matrix([[1.0], [1.0]]),
        constraint_lower_bound=[1.0, -np.inf], constraint_upper_bound=[np.inf, 0.0],
    ))
    model.setParam("iteration_limit", 10000)
    model.optimize(device=solver_device)
    assert model.Status == "PRIMAL_INFEASIBLE"
    assert model.DualRayObj > 0
    assert model.MaxDualRayInfeas / model.DualRayObj <= model.getParam("InfeasibleTol")


def test_unbounded_qp(solver_device):
    # The direction (0, 1) satisfies H d = 0 and c'd = -1.
    model = configure(Model(
        objective_vector=[0.0, -1.0], objective_matrix=sp.diags([2.0, 0.0], format="csr"),
        constraint_matrix=sp.eye(2, format="csr"), constraint_lower_bound=[0.0, 0.0],
    ))
    model.setParam("iteration_limit", 2000)
    model.optimize(device=solver_device)
    assert model.Status == "DUAL_INFEASIBLE"
    assert model.PrimalRayLinObj < 0
    assert model.MaxPrimalRayInfeas / -model.PrimalRayLinObj <= model.getParam("InfeasibleTol")
