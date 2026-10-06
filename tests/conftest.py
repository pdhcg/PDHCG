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
