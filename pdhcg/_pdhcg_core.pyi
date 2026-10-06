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

"""
pdhcg core bindings (auto-detect dense/CSR/CSC/COO; initialize default params here)
"""
from __future__ import annotations
import typing
__all__: list[str] = ['get_default_params', 'read_problem_file', 'solve_once', 'validate_params']
def get_default_params() -> dict:
    """
    Return default PDHG parameters as a dict
    """
def read_problem_file(path: str) -> dict:
    """
    Read an MPS or CBF file (.mps/.mps.gz/.cbf/.cbf.gz) and return a dict with c, obj_const, Q, A, constr_lb, constr_ub, var_lb, var_ub, cones, affine_F, affine_g, affine_cones, and primal_start.
    """
def solve_once(Q: typing.Any, R: typing.Any, A: typing.Any, objective_vector: typing.Any, objective_constant: typing.Any = None, variable_lower_bound: typing.Any = None, variable_upper_bound: typing.Any = None, constraint_lower_bound: typing.Any = None, constraint_upper_bound: typing.Any = None, zero_tolerance: typing.SupportsFloat | typing.SupportsIndex = 0.0, params: typing.Any = None, primal_start: typing.Any = None, dual_start: typing.Any = None, D: typing.Any = None, cones: typing.Any = None, affine_F: typing.Any = None, affine_g: typing.Any = None, affine_cones: typing.Any = None) -> dict:
    ...
def validate_params(params: typing.Any) -> None:
    """
    Validate a PDHG parameter dict
    """
