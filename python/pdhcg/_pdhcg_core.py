# SPDX-License-Identifier: Apache-2.0

"""Compatibility entry point for the former single-backend extension."""

from ._core import get_default_params, read_problem_file, solve_once, validate_params

__all__ = ["get_default_params", "read_problem_file", "solve_once", "validate_params"]
