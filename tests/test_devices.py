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

"""Backend selection and isolation tests against installed native modules."""

import os
import subprocess
import sys

import numpy as np
import pytest

from pdhcg import Model, built_devices
from pdhcg._build_config import DEFAULT_DEVICE
from pdhcg._core import get_default_params, solve_once


def _qp():
    model = Model(
        objective_vector=np.array([-2.0, -4.0]),
        objective_matrix=np.diag([2.0, 2.0]),
        variable_lower_bound=np.zeros(2),
    )
    model.setParams(Presolve=False, LogLevel=0, OptimalityTol=1e-7, FeasibilityTol=1e-7)
    return model


def _run_python(code):
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, text=True, capture_output=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_import_and_listing_do_not_load_native_backends():
    _run_python("""
import contextlib
import io
import sys
import pdhcg
from pdhcg._build_config import DEFAULT_DEVICE
assert pdhcg.built_devices()
output = io.StringIO()
with contextlib.redirect_stdout(output):
    result = pdhcg.print_devices()
assert result is None
assert output.getvalue().splitlines() == [
    'Built devices: ' + ', '.join(pdhcg.built_devices()),
    'Default device: ' + DEFAULT_DEVICE,
]
assert not any('pdhcg._pdhcg_' in name for name in sys.modules)
""")


def test_unknown_and_unbuilt_devices_fail_before_loading():
    with pytest.raises(ValueError, match="not compiled"):
        Model(objective_vector=[0.0]).optimize(device="unknown")
    with pytest.raises(TypeError, match="backend name"):
        solve_once(device=0)
    for device in {"cpu", "cuda"} - set(built_devices()):
        with pytest.raises(ValueError, match="not compiled"):
            solve_once(device=device)


def test_default_matches_build_configuration():
    from pdhcg._core import _backend

    assert _backend().device == DEFAULT_DEVICE


@pytest.mark.skipif("cpu" not in built_devices(), reason="CPU was not compiled")
def test_cpu_solve_and_legacy_entry_point():
    model = _qp()
    model.optimize(device="CPU")
    assert model.Status == "OPTIMAL"
    np.testing.assert_allclose(model.X, [1.0, 2.0], atol=1e-5)
    assert model.ObjVal == pytest.approx(-5.0, abs=1e-5)
    from pdhcg._pdhcg_core import get_default_params as legacy_defaults

    assert legacy_defaults() == get_default_params()


@pytest.mark.skipif("cpu" not in built_devices(), reason="CPU was not compiled")
def test_cpu_use_does_not_import_cuda_even_if_cuda_is_default():
    _run_python("""
import importlib.abc
import sys
import numpy as np
class BlockOtherDevices(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('pdhcg._pdhcg_') and fullname != 'pdhcg._pdhcg_cpu':
            raise AssertionError('Unexpected backend import: ' + fullname)
sys.meta_path.insert(0, BlockOtherDevices())
from pdhcg import Model
from pdhcg import _core
_core.DEFAULT_DEVICE = 'cuda'
m = Model(objective_vector=[1.0], objective_matrix=np.eye(1), variable_lower_bound=[2.0])
m.setParams(Presolve=False, LogLevel=0, OptimalityTol=1e-7, FeasibilityTol=1e-7)
m.optimize(device='cpu')
assert m.Status == 'OPTIMAL'
assert abs(m.X[0] - 2.0) < 1e-4
assert 'pdhcg._pdhcg_cuda' not in sys.modules
""")


@pytest.mark.skipif("cpu" not in built_devices(), reason="CPU was not compiled")
def test_file_reading_does_not_bind_a_device(tmp_path):
    path = tmp_path / "one.mps"
    path.write_text("""NAME ONE
ROWS
 N OBJ
 G ROW1
COLUMNS
 X OBJ 1 ROW1 1
RHS
 RHS1 ROW1 2
ENDATA
""")
    model = Model.read_file(str(path))
    model.setParams(Presolve=False, LogLevel=0)
    model.optimize(device="cpu")
    assert model.Status == "OPTIMAL"
    np.testing.assert_allclose(model.X, [2.0], atol=1e-3)


@pytest.mark.skipif("cpu" not in built_devices(), reason="CPU was not compiled")
def test_cvxpy_device_option_reaches_backend():
    cp = pytest.importorskip("cvxpy")
    import pdhcg.cvxpy_backend  # noqa: F401

    x = cp.Variable()
    problem = cp.Problem(cp.Minimize(x), [x >= 2])
    problem.solve(solver="PDHCG", device="cpu", presolve=False)
    assert problem.status == cp.OPTIMAL
    assert x.value == pytest.approx(2.0, abs=1e-3)
    with pytest.raises(ValueError, match="not compiled"):
        problem.solve(solver="PDHCG", device="not-built", presolve=False)



@pytest.mark.skipif(
    not {"cpu", "cuda"}.issubset(built_devices())
    or os.environ.get("PDHCG_TEST_CUDA_LIBRARIES") != "1",
    reason="requires both backends and their runtime libraries",
)
@pytest.mark.parametrize("order", [("cpu", "cuda"), ("cuda", "cpu")])
def test_backend_identity_and_cpu_solve_after_global_imports(order):
    _run_python(f"""
import importlib, os, sys
import numpy as np
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
for device in {order!r}:
    module = importlib.import_module('pdhcg._pdhcg_' + device)
    assert module.device == device, (device, module.device)
from pdhcg import Model
model = Model(objective_vector=np.array([-2.0]), objective_matrix=np.diag([2.0]))
model.setParams(Presolve=False, LogLevel=0, OptimalityTol=1e-7)
model.optimize(device='cpu')
assert model.Status == 'OPTIMAL'
assert abs(model.X[0] - 1.0) < 1e-5
""")

cuda_runtime = pytest.mark.skipif(
    "cuda" not in built_devices() or os.environ.get("PDHCG_TEST_CUDA") != "1",
    reason="requires a CUDA allocation and PDHCG_TEST_CUDA=1",
)


@cuda_runtime
def test_cuda_low_level_selection():
    result = solve_once(
        Q=np.diag([2.0, 2.0]), R=None, A=None,
        objective_vector=np.array([-2.0, -4.0]),
        variable_lower_bound=np.zeros(2),
        params={"presolve": False, "verbose": 0, "eps_optimal_relative": 1e-7},
        device="cuda",
    )
    assert result["Status"] == "OPTIMAL"
    np.testing.assert_allclose(result["X"], [1.0, 2.0], atol=1e-5)


@cuda_runtime
@pytest.mark.skipif("cpu" not in built_devices(), reason="requires both backends")
def test_each_solve_selects_its_own_backend(monkeypatch):
    from pdhcg import _core

    calls = []
    for name in built_devices():
        module = _core._backend(name)
        original = module.solve_once

        def record(*args, _name=name, _solve=original, **kwargs):
            calls.append(_name)
            return _solve(*args, **kwargs)

        monkeypatch.setattr(module, "solve_once", record)
    model, other = _qp(), _qp()
    for selected in ("cpu", "cuda", "cpu"):
        model.optimize(device=selected)
        assert model.Status == "OPTIMAL"
        np.testing.assert_allclose(model.X, [1.0, 2.0], atol=1e-5)
    # An explicit selection must not change the default for the same or another model.
    model.optimize()
    other.optimize()
    assert calls == ["cpu", "cuda", "cpu", DEFAULT_DEVICE, DEFAULT_DEVICE]


@cuda_runtime
@pytest.mark.skipif("cpu" not in built_devices(), reason="requires both backends")
@pytest.mark.parametrize("order", [("cpu", "cuda"), ("cuda", "cpu")])
def test_native_symbols_are_isolated_with_global_imports(order):
    _run_python(f"""
import importlib, os, sys
import numpy as np
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
for device in {order!r}:
    module = importlib.import_module('pdhcg._pdhcg_' + device)
    assert module.device == device, (device, module.device)
    r = module.solve_once(Q=np.diag([2.0]), R=None, A=None,
        objective_vector=np.array([-2.0]), params={{'presolve': False, 'verbose': 0}})
    assert r['Status'] == 'OPTIMAL'
    assert abs(r['X'][0] - 1.0) < 1e-3
""")


@pytest.mark.skipif(
    not {"cpu", "cuda"}.issubset(built_devices())
    or os.environ.get("PDHCG_TEST_CUDA_LIBRARIES") != "1",
    reason="requires both backends and their runtime libraries",
)
def test_unavailable_cuda_raises_and_cpu_still_works():
    _run_python("""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from pdhcg import Model
gpu = Model(objective_vector=[1.0], objective_matrix=np.eye(1), variable_lower_bound=[2.0])
gpu.setParams(Presolve=False, LogLevel=0)
try:
    gpu.optimize(device='cuda')
except RuntimeError as exc:
    assert 'initialize' in str(exc) and 'cuda' in str(exc)
else:
    raise AssertionError('Expected CUDA initialization failure')
cpu = Model(objective_vector=[1.0], objective_matrix=np.eye(1), variable_lower_bound=[2.0])
cpu.setParams(Presolve=False, LogLevel=0)
cpu.optimize(device='cpu')
assert cpu.Status == 'OPTIMAL'
""")
