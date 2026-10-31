# Copyright 2026 Hongpei Li
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
from pdhcg._core import get_default_params, validate_params

from pdhcg import Model
from pdhcg import built_devices


def test_parameter_defaults_pass_core_validation() -> None:
    defaults = get_default_params()

    assert "eps_infeasible" in defaults
    assert [defaults[key] for key in ("k_p", "k_i", "k_d", "i_smooth")] == pytest.approx(
        [0.99, 0.01, 0.0, 0.3]
    )
    validate_params(defaults)


@pytest.mark.parametrize("alias, native, value", [
    ("RestartKp", "k_p", -0.5),
    ("RestartKi", "k_i", -0.1),
    ("RestartKd", "k_d", -0.2),
    ("RestartISmooth", "i_smooth", 1.0),
])
def test_weight_parameter_aliases(alias, native, value) -> None:
    model = Model(objective_vector=np.zeros(1))
    model.setParam(alias, value)
    assert model.getParam(native) == pytest.approx(value)
    model.setParam(native, 0.0)
    assert model.getParam(alias) == 0.0


@pytest.mark.parametrize("name", ["RestartKp", "RestartKi", "RestartKd"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_weight_gains_reject_nonfinite_values(name, value) -> None:
    model = Model(objective_vector=np.zeros(1))
    before = model.getParam(name)
    with pytest.raises(ValueError, match="restart PID coefficients must be finite"):
        model.setParam(name, value)
    assert model.getParam(name) == before


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
def test_weight_smoothing_validation(value) -> None:
    model = Model(objective_vector=np.zeros(1))
    before = model.getParam("RestartISmooth")
    with pytest.raises(ValueError, match="restart i_smooth"):
        model.setParam("RestartISmooth", value)
    assert model.getParam("RestartISmooth") == before


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
