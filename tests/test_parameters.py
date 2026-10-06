# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from pdhcg._core import get_default_params, validate_params

from pdhcg import Model
from pdhcg import built_devices


def test_parameter_defaults_pass_core_validation() -> None:
    defaults = get_default_params()

    assert "eps_infeasible" in defaults
    validate_params(defaults)


def test_set_param_rejects_unknown_name() -> None:
    model = Model(objective_vector=np.zeros(1))

    with pytest.raises(ValueError, match="Unknown parameter"):
        model.setParam("does_not_exist", 1)


def test_set_param_is_transactional() -> None:
    model = Model(objective_vector=np.zeros(1))
    original_frequency = model.getParam("TermCheckFreq")

    with pytest.raises(ValueError, match="termination_evaluation_frequency"):
        model.setParam("TermCheckFreq", 0)

    assert model.getParam("TermCheckFreq") == original_frequency


def test_set_params_is_transactional() -> None:
    model = Model(objective_vector=np.zeros(1))
    original_time_limit = model.getParam("TimeLimit")

    with pytest.raises(ValueError, match="eps_optimal_relative"):
        model.setParams(TimeLimit=2.0, OptimalityTol=0.0)

    assert model.getParam("TimeLimit") == original_time_limit


def test_infeasible_tolerance_alias() -> None:
    model = Model(objective_vector=np.zeros(1))

    model.setParam("InfeasibleTol", 1e-10)

    assert model.getParam("eps_infeasible") == pytest.approx(1e-10)


def test_threads_validation_is_transactional() -> None:
    model = Model(objective_vector=np.zeros(1))
    assert model.getParam("Threads") == 0
    model.setParam("Threads", 2)
    assert model.getParam("num_threads") == 2
    with pytest.raises(ValueError, match="num_threads"):
        model.setParam("Threads", -1)
    assert model.getParam("Threads") == 2
    model.setParam("Threads", 0)


@pytest.mark.skipif("cpu" not in built_devices(), reason="CPU backend required")
def test_threads_can_change_between_cpu_solves() -> None:
    import scipy.sparse as sp

    n = 2048
    model = Model(objective_matrix=sp.eye(n, format="csr"),
                  objective_vector=-np.ones(n))
    model.setParams(LogLevel=0, OptimalityTol=1e-7, FeasibilityTol=1e-7)
    for threads in (1, 2, 0):
        model.setParam("Threads", threads)
        model.optimize(device="cpu")
        assert model.Status == "OPTIMAL"
        np.testing.assert_allclose(model.X, 1.0, atol=1e-5)


@pytest.mark.skipif("cpu" not in built_devices(), reason="CPU backend required")
def test_cvxpy_threads_option() -> None:
    cp = pytest.importorskip("cvxpy")
    import pdhcg.cvxpy_backend  # noqa: F401

    x = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)))
    problem.solve(solver="PDHCG", device="cpu", Threads=2,
                  eps_optimal_relative=1e-7, eps_feasible_relative=1e-7)
    assert problem.status == cp.OPTIMAL
    np.testing.assert_allclose(x.value, 1.0, atol=1e-4)
