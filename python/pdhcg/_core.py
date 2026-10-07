# Copyright 2025 Haihao Lu
# Copyright 2026 Hongpei Li
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python access to the unified native solver."""

from importlib import import_module

__all__ = ["built_devices", "default_device", "print_devices", "get_default_params", "read_problem_file", "solve_once", "validate_params"]


def built_devices():
    """Return compiled backend names without initializing a device."""
    return _native().built_devices()


def default_device():
    """Return the native library's default backend."""
    return _native().default_device()


def print_devices() -> None:
    """Print compiled backends and the default without initializing a device."""
    print("Built devices: " + ", ".join(built_devices()))
    print("Default device: " + default_device())


def _native():
    try:
        return import_module("._pdhcg_core", __package__)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"Could not load the PDHCG native library: {exc}. "
            "Check that its runtime libraries are installed."
        ) from exc


def get_default_params():
    return _native().get_default_params()


def validate_params(params):
    return _native().validate_params(params)


def read_problem_file(path):
    return _native().read_problem_file(path)


def solve_once(*args, device=None, **kwargs):
    """Solve using the requested backend, or the default chosen at build time."""
    return _native().solve_once(*args, device=device, **kwargs)
