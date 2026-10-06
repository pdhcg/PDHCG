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

"""Select an installed native backend without importing the other devices."""

from importlib import import_module

from ._build_config import BUILT_DEVICES, DEFAULT_DEVICE

__all__ = ["built_devices", "print_devices", "get_default_params", "read_problem_file", "solve_once", "validate_params"]


def built_devices():
    """Return compiled backend names; this does not probe hardware or load drivers."""
    return BUILT_DEVICES


def print_devices() -> None:
    """Print compiled backends and the default, without loading solver backends."""
    print("Built devices: " + ", ".join(BUILT_DEVICES))
    print("Default device: " + DEFAULT_DEVICE)


def resolve_device(device=None):
    if device is None:
        return DEFAULT_DEVICE
    if not isinstance(device, str):
        raise TypeError("device must be a backend name such as 'cpu' or 'cuda'")
    name = device.strip().lower()
    if name not in BUILT_DEVICES:
        raise ValueError(
            f"PDHCG device {device!r} is not compiled into this installation. "
            f"Built devices: {', '.join(BUILT_DEVICES)}. "
            "Select backends with PDHCG_DEVICES when building from source."
        )
    return name


def _backend(device=None):
    name = resolve_device(device)
    try:
        return import_module(f"._pdhcg_{name}", __package__)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"Could not load the compiled PDHCG {name!r} backend: {exc}. "
            "Check that its runtime libraries are installed."
        ) from exc


def _metadata_backend():
    # Model construction and file I/O are host operations. Prefer the CPU
    # extension, if built, so they never require CUDA libraries in a mixed wheel.
    return _backend("cpu" if "cpu" in BUILT_DEVICES else DEFAULT_DEVICE)


def get_default_params():
    return _metadata_backend().get_default_params()


def validate_params(params):
    return _metadata_backend().validate_params(params)


def read_problem_file(path):
    return _metadata_backend().read_problem_file(path)


def solve_once(*args, device=None, **kwargs):
    """Solve using the requested backend, or the default chosen at build time."""
    return _backend(device).solve_once(*args, **kwargs)
