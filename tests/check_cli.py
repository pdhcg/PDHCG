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

"""Exercise an explicit CPU CLI build using only the Python standard library."""

import argparse
import gzip
import math
from pathlib import Path
import subprocess
import tempfile


def run(binary, *args, expect_failure=False):
    result = subprocess.run(
        [str(binary), *map(str, args)], capture_output=True, text=True, timeout=45
    )
    log = result.stdout + result.stderr
    if bool(result.returncode) != expect_failure:
        raise AssertionError(f"Unexpected exit code {result.returncode}:\n{log}")
    return log


def check_solution(
    binary, problem, output, expected_x, expected_objective,
    *, infeasibility_tolerance=None, device=None,
):
    output.mkdir()
    options = [] if infeasibility_tolerance is None else ["--eps_infeasible", infeasibility_tolerance]
    if device is not None:
        options.extend(["--device", device])
    log = run(
        binary, "--verbose", "2", "--threads", "2", "--time_limit", "30",
        "--eps_opt", "1e-7", "--eps_feas", "1e-7", *options, problem, output,
    )
    tolerance_log = [
        line.partition(":")[2] for line in log.splitlines()
        if line.partition(":")[0].strip() == "eps_infeasible"
    ]
    assert len(tolerance_log) == 1, log
    expected_tolerance = 1e-10 if infeasibility_tolerance is None else infeasibility_tolerance
    assert float(tolerance_log[0]) == expected_tolerance, log
    summaries = list(output.glob("*_summary.txt"))
    primals = list(output.glob("*_primal_solution.txt"))
    assert len(summaries) == len(primals) == 1, log
    summary = dict(line.split(": ", 1) for line in summaries[0].read_text().splitlines())
    assert summary["Termination Reason"] == "OPTIMAL", log
    actual_x = [float(value) for value in primals[0].read_text().split()]
    assert len(actual_x) == len(expected_x), actual_x
    for actual, expected in zip(actual_x, expected_x):
        assert math.isclose(actual, expected, rel_tol=0, abs_tol=2e-4), (actual_x, log)
    objective = float(summary["Primal Objective Value"])
    assert math.isclose(objective, expected_objective, rel_tol=0, abs_tol=2e-4), log
    for field in ["Relative Primal Residual", "Relative Dual Residual", "Relative Objective Gap"]:
        assert abs(float(summary[field])) <= 2e-7, (field, summary[field], log)
    print("PASS CLI", problem.name, flush=True)


def check_infeasibility_options(binary):
    # --help follows the option so parsing is exercised without another solve.
    # A finite value that rounds down to zero is accepted, as in C/Python.
    run(binary, "--eps_infeasible", "1e-400", "--help")
    # Parameter validation runs before attempting to read the input file.
    for value in ["nan", "inf", "-1"]:
        log = run(binary, "--eps_infeasible", value, "unused.qps", "unused-output", expect_failure=True)
        assert "eps_infeasible must be finite and nonnegative" in log, log
    log = run(binary, "--threads", "-1", "unused.qps", "unused-output", expect_failure=True)
    assert "num_threads must be nonnegative" in log, log
    print("PASS CLI infeasibility options", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    binary = parser.parse_args().binary.resolve()
    assert run(binary, "--list-devices").splitlines() == [
        "Built devices: cpu", "Default device: cpu",
    ]
    print("PASS CLI --list-devices", flush=True)
    assert "--device <name>" in run(binary, "--help")
    for device in ["unknown", "cuda"]:
        log = run(binary, "--device", device, "unused.qps", "unused-output", expect_failure=True)
        assert "Error:" in log and "device" in log and device in log, log
        assert "Failed to read" not in log, log
    print("PASS CLI device validation", flush=True)
    check_infeasibility_options(binary)
    with tempfile.TemporaryDirectory(prefix="pdhcg-cli-") as directory:
        root = Path(directory)
        # min x^2 + y^2 - 2x - 4y, x+y >= 4, x,y >= 0.
        # The unique optimum is (1.5, 2.5), with objective -4.5.
        qp = root / "tiny.qps"
        qp.write_text("""NAME TINY_QP
ROWS
 N OBJ
 G SUM
COLUMNS
 X OBJ -2 SUM 1
 Y OBJ -4 SUM 1
RHS
 RHS1 SUM 4
BOUNDS
 LO BND X 0
 LO BND Y 0
QUADOBJ
 X X 2
 Y Y 2
ENDATA
""")
        check_solution(
            binary, qp, root / "qp", [1.5, 2.5], -4.5,
            infeasibility_tolerance=1e-12,
        )
        compressed = root / "tiny.qps.gz"
        with gzip.open(compressed, "wb") as stream:
            stream.write(qp.read_bytes())
        check_solution(
            binary, compressed, root / "gzip", [1.5, 2.5], -4.5,
            infeasibility_tolerance=0.0, device="cpu",
        )
        # The parser converts CBF's scalar-first SOC to PDHCG's (v, w, z) order.
        cbf = Path(__file__).parent / "data" / "cbf_q3_smoke.cbf"
        check_solution(binary, cbf, root / "soc", [3.0, 4.0, 5.0], 5.0, device="auto")


if __name__ == "__main__":
    main()
