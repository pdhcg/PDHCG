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

"""Device selection through the installed unified native interface."""

import os
import subprocess
import sys

import numpy as np
import pytest

from pdhcg import Model, built_devices
from pdhcg import _core
from pdhcg._core import get_default_params, solve_once


DEFAULT_DEVICE = _core.default_device()
CPU_AVAILABLE = "cpu" in built_devices()
CUDA_ENABLED = "cuda" in built_devices() and os.environ.get("PDHCG_TEST_CUDA") == "1"


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


def test_import_is_lazy_and_metadata_uses_one_native_extension():
    _run_python("""
import contextlib
import importlib.machinery
import io
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import pdhcg
assert 'pdhcg._pdhcg_core' not in sys.modules
from pdhcg import _core
assert pdhcg.built_devices()
from pdhcg import _pdhcg_core as native
assert any(native.__file__.endswith(suffix)
           for suffix in importlib.machinery.EXTENSION_SUFFIXES)
assert pdhcg.built_devices() == native.built_devices()
assert _core.default_device() == native.default_device()
assert _core.default_device() in pdhcg.built_devices()
output = io.StringIO()
with contextlib.redirect_stdout(output):
    result = pdhcg.print_devices()
assert result is None
assert output.getvalue().splitlines() == [
    'Built devices: ' + ', '.join(native.built_devices()),
    'Default device: ' + native.default_device(),
]
assert [name for name in sys.modules if name.startswith('pdhcg._pdhcg_')] == ['pdhcg._pdhcg_core']
""")


def test_device_rejected_before_problem_conversion():
    from pdhcg import _pdhcg_core as native

    invalid = ["unknown", "", *sorted({"cpu", "cuda"} - set(built_devices()))]
    # Q is deliberately not a matrix: device validation must run first, both
    # through the Python wrapper and when calling the extension directly.
    for solve in (solve_once, native.solve_once):
        for device in invalid:
            with pytest.raises(ValueError, match="not compiled"):
                solve(Q=object(), R=None, A=None, objective_vector=None, device=device)
        with pytest.raises(TypeError, match="backend name"):
            solve(Q=object(), R=None, A=None, objective_vector=None, device=0)
        with pytest.raises(ValueError, match="NUL"):
            solve(Q=object(), R=None, A=None, objective_vector=None, device="cpu\0junk")
    with pytest.raises(ValueError, match="not compiled"):
        _qp().optimize(device="unknown")



@pytest.mark.skipif(not CPU_AVAILABLE, reason="CPU was not compiled")
@pytest.mark.parametrize("device", ["cpu", "CPU", " CPU "])
def test_cpu_solve_and_native_defaults(device):
    model = _qp()
    model.optimize(device=device)
    assert model.Status == "OPTIMAL"
    np.testing.assert_allclose(model.X, [1.0, 2.0], atol=1e-5)
    assert model.ObjVal == pytest.approx(-5.0, abs=1e-5)
    from pdhcg import _pdhcg_core as native

    assert native.get_default_params() == get_default_params()


@pytest.mark.skipif(not CPU_AVAILABLE, reason="CPU was not compiled")
def test_file_reading_and_cpu_solve_do_not_initialize_cuda(tmp_path):
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
    _run_python(f"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from pdhcg import Model, built_devices
from pdhcg._core import get_default_params
assert 'cpu' in built_devices()
assert get_default_params()
model = Model.read_file({str(path)!r})
model.setParams(Presolve=False, LogLevel=0)
model.optimize(device='cpu')
assert model.Status == 'OPTIMAL'
np.testing.assert_allclose(model.X, [2.0], atol=1e-3)
""")


@pytest.mark.skipif(not CPU_AVAILABLE, reason="CPU was not compiled")
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


@pytest.mark.skipif(not CPU_AVAILABLE, reason="CPU was not compiled")
@pytest.mark.parametrize("field", ["objective_vector", "primal_start"])
def test_failed_native_input_does_not_break_next_cpu_solve(field):
    from pdhcg import _pdhcg_core as native

    kwargs = dict(Q=np.eye(2), R=None, A=None, objective_vector=np.zeros(2), device="cpu")
    # A bad objective fails before problem creation; a bad warm start fails
    # after it, exercising both cleanup paths in the native binding.
    kwargs[field] = np.zeros(1)
    with pytest.raises(ValueError, match=field):
        native.solve_once(**kwargs)
    model = _qp()
    model.optimize(device="cpu")
    assert model.Status == "OPTIMAL"
    np.testing.assert_allclose(model.X, [1.0, 2.0], atol=1e-5)


@pytest.mark.skipif(not CPU_AVAILABLE, reason="CPU was not compiled")
def test_concurrent_cpu_solves_restore_signal_handler():
    _run_python("""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import signal
import sys
from threading import Barrier
import numpy as np
import scipy.sparse as sparse
from pdhcg import Model

def signal_handler():
    if sys.platform != 'linux':
        return None
    # sigaction reads the actual C handler; signal.getsignal only reports
    # Python's cached handler and misses a stale native signal handler.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.sigaction.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    action = ctypes.create_string_buffer(256)
    assert libc.sigaction(signal.SIGINT, None, ctypes.byref(action)) == 0
    return ctypes.cast(ctypes.byref(action), ctypes.POINTER(ctypes.c_void_p)).contents.value

before = signal_handler()
ready = Barrier(2)
def solve(target):
    model = Model(objective_matrix=sparse.eye(8192, format='csr'),
                  objective_vector=np.full(8192, -target))
    model.setParams(Presolve=False, LogLevel=0, Threads=1,
                    OptimalityTol=1e-7, FeasibilityTol=1e-7)
    ready.wait(timeout=10)
    model.optimize(device='cpu')
    assert model.Status == 'OPTIMAL'
    np.testing.assert_allclose(model.X, target, atol=1e-5)
    return model.X.copy()

with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(solve, (1.0, 3.0)))
assert not np.shares_memory(*results)
assert signal_handler() == before
""")


cuda_runtime = pytest.mark.skipif(
    not CUDA_ENABLED,
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


@pytest.mark.skipif(
    DEFAULT_DEVICE != "cpu" and not CUDA_ENABLED,
    reason="the compiled default requires a CUDA allocation",
)
def test_default_and_auto_selection_do_not_change_after_explicit_solve():
    model, other = _qp(), _qp()
    explicit = "cpu" if CPU_AVAILABLE else "cuda"
    model.optimize(device=explicit)
    model.optimize()
    model.optimize(device="auto")
    other.optimize()
    for solved in (model, other):
        assert solved.Status == "OPTIMAL"
        np.testing.assert_allclose(solved.X, [1.0, 2.0], atol=1e-5)
    assert _core.default_device() == DEFAULT_DEVICE


@cuda_runtime
@pytest.mark.skipif(not CPU_AVAILABLE, reason="requires both backends")
def test_same_model_can_alternate_devices():
    model = _qp()
    for selected in ("cpu", "cuda", "cpu", "cuda"):
        model.optimize(device=selected)
        assert model.Status == "OPTIMAL"
        np.testing.assert_allclose(model.X, [1.0, 2.0], atol=1e-5)


@pytest.mark.skipif(
    not {"cpu", "cuda"}.issubset(built_devices())
    or os.environ.get("PDHCG_TEST_CUDA_LIBRARIES") != "1",
    reason="requires both backends and their runtime libraries",
)
def test_unavailable_cuda_does_not_fall_back_and_cpu_still_works():
    _run_python("""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from pdhcg import Model
from pdhcg._core import default_device
model = Model(objective_vector=[1.0], objective_matrix=np.eye(1), variable_lower_bound=[2.0])
model.setParams(Presolve=False, LogLevel=0)
selections = ['cuda']
if default_device() == 'cuda':
    selections += [None, 'auto']
for device in selections:
    try:
        model.optimize(device=device)
    except RuntimeError as exc:
        assert 'cuda' in str(exc).lower()
    else:
        raise AssertionError('Unavailable CUDA must not fall back to CPU')
model.optimize(device='cpu')
assert model.Status == 'OPTIMAL'
assert abs(model.X[0] - 2.0) < 1e-3
""")
