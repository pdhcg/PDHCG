# SPDX-License-Identifier: Apache-2.0

"""Run numerical regressions on CPU, and on CUDA when explicitly enabled."""

import os

import pytest

from pdhcg import built_devices


@pytest.fixture(
    params=[
        pytest.param(
            device,
            id=device,
            marks=(
                [
                    pytest.mark.gpu,
                    pytest.mark.skipif(
                        os.environ.get("PDHCG_TEST_CUDA") != "1",
                        reason="requires a CUDA allocation and PDHCG_TEST_CUDA=1",
                    ),
                ]
                if device == "cuda"
                else []
            ),
        )
        for device in built_devices()
    ]
)
def solver_device(request):
    return request.param
