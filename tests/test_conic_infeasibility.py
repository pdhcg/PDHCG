# SPDX-License-Identifier: Apache-2.0

"""Conic status regressions with explicit separating vectors and recession rays."""

import numpy as np
import pytest
import scipy.sparse as sp

from pdhcg import ConeSpec, Model


KINDS = ["soc", "rsoc", "psd", "exp", "power"]
ALPHA = 0.25


def cone(kind, fixed_mask=None):
    return ConeSpec(
        kind, np.array([0], dtype=np.int32), v_dims=2 if kind == "psd" else 1,
        power_alphas=ALPHA if kind == "power" else 0.0, fixed_mask=fixed_mask,
    )


def geometry(kind):
    # w lies in K*, d lies in K, and w'd > 0. EXP and POWER deliberately
    # use dual vectors outside K: these cones are not self-dual.
    vectors = {
        "soc": ([1, 0, 1], [0, 0, 1], [1, 0, -1]),
        "rsoc": ([1, 1, 1], [0, 1, 1], [1, -1, 0]),
        "psd": ([1, np.sqrt(2), 1], [1, 0, 1], [1, 0, -1]),
        "exp": ([-1, -1, 1], [0, 1, 2], [1, -1, 0]),
        "power": ([ALPHA, 1 - ALPHA, -1], [1, 1, 0], [3, -1, 0]),
    }
    w, d, tangent = (np.asarray(value, dtype=float) for value in vectors[kind])
    assert_member(kind, w, dual=True)
    assert_member(kind, d)
    assert w @ d > 0
    assert w @ tangent == pytest.approx(0, abs=1e-14)
    return w, d, tangent


def assert_member(kind, point, *, dual=False, tolerance=1e-10):
    """Independent cone inequalities in the public coordinate ordering."""
    x, y, z = np.asarray(point, dtype=float)
    assert np.all(np.isfinite(point))
    if kind == "soc":
        assert np.hypot(x, y) <= z + tolerance
    elif kind == "rsoc":
        assert y >= -tolerance and z >= -tolerance
        assert x * x <= 2 * y * z + tolerance
    elif kind == "psd":
        matrix = [[x, y / np.sqrt(2)], [y / np.sqrt(2), z]]
        assert np.linalg.eigvalsh(matrix).min() >= -tolerance
    elif kind == "exp":
        if dual:
            if x < -tolerance:
                assert -x * np.exp(y / x) <= np.e * z + tolerance
            else:
                assert abs(x) <= tolerance and y >= -tolerance and z >= -tolerance
        elif y > tolerance:
            assert y * np.exp(x / y) <= z + tolerance
        else:
            assert abs(y) <= tolerance and x <= tolerance and z >= -tolerance
    else:
        assert kind == "power"
        assert x >= -tolerance and y >= -tolerance
        if dual:
            x, y = x / ALPHA, y / (1 - ALPHA)
        assert max(x, 0) ** ALPHA * max(y, 0) ** (1 - ALPHA) + tolerance >= abs(z)


def configure(model):
    # Keep the default 1e-10 certificate tolerance, including ill-scaled cases.
    model.setParams(
        Presolve=False, LogLevel=0, TimeLimit=15.0, iteration_limit=20000,
        OptimalityTol=1e-7, FeasibilityTol=1e-7,
    )
    return model


def assert_status(model, device, expected):
    model.optimize(device=device)
    assert model.Status == expected


@pytest.mark.parametrize("kind", KINDS)
def test_variable_cone_strongly_infeasible(solver_device, kind):
    w, _, _ = geometry(kind)
    # x in K implies w'x >= 0, contradicting w'x <= -1. The linear
    # multiplier -1 and cone multiplier w have zero stationarity and gap 1.
    model = configure(Model(
        objective_vector=np.zeros(3), constraint_matrix=sp.csr_matrix(w[None, :]),
        constraint_upper_bound=[-1.0], variable_cones=cone(kind),
    ))
    assert_status(model, solver_device, "PRIMAL_INFEASIBLE")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("feasible", [False, True], ids=["negative-offset", "feasible-offset"])
def test_affine_cone_offset_sign(solver_device, kind, feasible):
    w, d, tangent = geometry(kind)
    offset = d if feasible else -w
    # F'w = 0. For g=-w, w'(Fx+g)=-||w||^2 < 0 separates every
    # affine image from K; for g=d, x=0 is an explicit feasible point.
    assert tangent @ w == pytest.approx(0, abs=1e-14)
    if not feasible:
        assert offset @ w < 0
    model = configure(Model(
        objective_vector=[0.0], affine_cone_matrix=sp.csr_matrix(tangent[:, None]),
        affine_cone_offset=offset, affine_cones=cone(kind),
    ))
    assert_status(model, solver_device, "OPTIMAL" if feasible else "PRIMAL_INFEASIBLE")
    if feasible:
        assert_member(kind, tangent * model.X[0] + offset, tolerance=2e-5)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("affine", [False, True], ids=["variable", "affine"])
def test_conic_recession_direction(solver_device, kind, affine):
    w, d, _ = geometry(kind)
    if affine:
        # x=0 is feasible, F*1=d belongs to K, and c*1=-1.
        model = Model(
            objective_vector=[-1.0], affine_cone_matrix=sp.csr_matrix(d[:, None]),
            affine_cone_offset=d, affine_cones=cone(kind),
        )
    else:
        # The origin is feasible and the ray d in K has c'd=-w'd < 0.
        assert -w @ d < 0
        model = Model(
            objective_vector=-w, constraint_matrix=sp.csr_matrix((0, 3)),
            variable_cones=cone(kind),
        )
    assert_status(configure(model), solver_device, "DUAL_INFEASIBLE")


@pytest.mark.parametrize("kind", KINDS)
def test_variable_cone_free_bound_sentinels(solver_device, kind):
    _, d, _ = geometry(kind)
    # The cone API accepts +/-1e30 as free-bound sentinels. Treating those
    # entries as an ordinary finite box would incorrectly erase every ray.
    model = configure(Model(
        objective_vector=-d, constraint_matrix=sp.csr_matrix((0, 3)),
        variable_lower_bound=np.full(3, -1e30), variable_upper_bound=np.full(3, 1e30),
        variable_cones=cone(kind),
    ))
    assert_status(model, solver_device, "DUAL_INFEASIBLE")


@pytest.mark.parametrize("kind", KINDS)
def test_quadratic_curvature_blocks_conic_unboundedness(solver_device, kind):
    _, d, _ = geometry(kind)
    # 1/2 ||x||^2 - d'x = 1/2 ||x-d||^2 - 1/2 ||d||^2.
    # Since d is in K, the exact optimum is x=d. A decreasing linear
    # recession direction alone must not override the nonzero H*d test.
    model = configure(Model(
        objective_vector=-d, objective_matrix=sp.eye(3, format="csr"),
        variable_cones=cone(kind),
    ))
    assert_status(model, solver_device, "OPTIMAL")
    np.testing.assert_allclose(model.X, d, atol=3e-5, rtol=0)
    assert model.ObjVal == pytest.approx(-0.5 * d @ d, abs=3e-5)
    assert_member(kind, model.X, tolerance=3e-5)


def section(kind):
    # Each fixed section is nonempty, but its support in the selected
    # coordinate is finite. The support values below follow directly from
    # ||(u,w)||<=1, v^2<=2, exp(x)<=1, and |z|<=1, respectively.
    sections = {
        "soc": ([0, 0, 1], [0, 0, 1], 0, 1.0, [1, 0, 1]),
        "rsoc": ([0, 1, 1], [0, 1, 1], 0, np.sqrt(2), [np.sqrt(2), 1, 1]),
        "exp": ([0, 1, 1], [0, 1, 1], 0, 0.0, [0, 1, 1]),
        "power": ([1, 1, 0], [1, 1, 0], 2, 1.0, [1, 1, 1]),
    }
    mask, start, coordinate, support, optimum = sections[kind]
    assert_member(kind, start)
    assert_member(kind, optimum)
    return np.asarray(mask, dtype=np.uint8), np.asarray(start, dtype=float), coordinate, support, optimum


@pytest.mark.parametrize("kind", ["soc", "rsoc", "exp", "power"])
@pytest.mark.parametrize("infeasible", [False, True], ids=["bounded-section", "infeasible-section"])
def test_fixed_cone_section_support(solver_device, kind, infeasible):
    mask, start, coordinate, support, optimum = section(kind)
    direction = np.eye(3)[coordinate]
    # For the infeasible case, the multiplier 1 yields certificate objective
    # (support + 1) - support = 1. Dropping the fixed-section support term
    # would be incorrect; treating the section as an entire cone also fails.
    model = configure(Model(
        objective_vector=np.zeros(3) if infeasible else -direction,
        constraint_matrix=sp.csr_matrix(direction[None, :]) if infeasible else sp.csr_matrix((0, 3)),
        constraint_lower_bound=[support + 1] if infeasible else None,
        variable_cones=cone(kind, fixed_mask=mask),
    ))
    model.setWarmStart(primal=start)
    assert_status(model, solver_device, "PRIMAL_INFEASIBLE" if infeasible else "OPTIMAL")
    if not infeasible:
        np.testing.assert_allclose(model.X, optimum, atol=5e-5, rtol=0)
        assert model.ObjVal == pytest.approx(-support, abs=5e-5)
        np.testing.assert_allclose(model.X[mask != 0], start[mask != 0], atol=1e-10, rtol=0)


@pytest.mark.parametrize("feasible", [False, True], ids=["infeasible", "feasible"])
def test_mixed_variable_and_affine_cones(solver_device, feasible):
    axis = np.array([0.0, 0.0, 1.0])
    offset = axis if feasible else -axis
    # x in SOC gives t>=0; -x-e_t in SOC would require t<=-1.
    # The two cone multipliers e_t cancel through F=-I, with gap 1.
    # With +e_t, 0<=t<=1 and min -t has the exact solution e_t.
    model = configure(Model(
        objective_vector=-axis, variable_cones=cone("soc"),
        affine_cone_matrix=-sp.eye(3, format="csr"), affine_cone_offset=offset,
        affine_cones=cone("soc"),
    ))
    assert_status(model, solver_device, "OPTIMAL" if feasible else "PRIMAL_INFEASIBLE")
    if feasible:
        np.testing.assert_allclose(model.X, axis, atol=5e-5, rtol=0)
        assert model.ObjVal == pytest.approx(-1.0, abs=5e-5)


@pytest.mark.parametrize("kind", ["soc", "exp", "power"])
def test_nonuniform_affine_scaling_certificate(solver_device, kind):
    w, _, _ = geometry(kind)
    diagonal = np.array([1e-4, 1.0, 1e4])
    # F x in K and w'F x <= -1 are strongly inconsistent. The multipliers
    # y_linear=-1 and y_cone=w cancel exactly even with a 1e8 column ratio.
    model = configure(Model(
        objective_vector=np.zeros(3), constraint_matrix=sp.csr_matrix((w * diagonal)[None, :]),
        constraint_upper_bound=[-1.0], affine_cone_matrix=sp.diags(diagonal, format="csr"),
        affine_cones=cone(kind),
    ))
    model.setParam("use_cone_preserving_scaling", False)
    assert_status(model, solver_device, "PRIMAL_INFEASIBLE")
