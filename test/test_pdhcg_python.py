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

import unittest
import numpy as np
import scipy.sparse as sp
import sys
import os

# Attempt to import pdhcg.
try:
    import pdhcg
except ImportError:
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../python"))
    )
    try:
        import pdhcg
    except ImportError:
        print("Error: Could not import pdhcg module.")
        sys.exit(1)

from pdhcg import Model


class TestPDHCGInterface(unittest.TestCase):
    def setUp(self):
        self.default_params = {
            "Presolve": False,
            "LogLevel": 2,
            "OptimalityTol": 1e-6,
            "FeasibilityTol": 1e-6,
        }

    def _get_debug_msg(self, m):
        """Helper function to format the solver's current state for error messages."""
        return f"\n--> [Solver State] Status: {m.Status}, ObjVal: {m.ObjVal}, X: {m.X}"

    def test_lowrank_qp(self):
        """
        Test Low-Rank QP: Minimize 0.5 * ||Rx||^2 + c^T x
        Subject to: x1 + x2 = 1, x >= 0
        """
        print("\n[Test] Low-Rank QP Minimization (R only)")

        R = sp.eye(2, format="csc")
        c = np.array([0.0, 0.0])
        A = np.array([[1.0, 1.0]])
        l = np.array([1.0])
        u = np.array([1.0])
        lb = np.array([0.0, 0.0])

        m = Model(
            objective_vector=c,
            constraint_matrix=A,
            constraint_lower_bound=l,
            constraint_upper_bound=u,
            objective_matrix=None,
            objective_matrix_low_rank=R,
            variable_lower_bound=lb,
        )

        m.setParams(**self.default_params)
        m.optimize()

        debug_msg = self._get_debug_msg(m)
        self.assertEqual(m.Status, "OPTIMAL", msg=f"Status mismatch! {debug_msg}")
        self.assertAlmostEqual(m.ObjVal, 0.25, places=4, msg=f"ObjVal mismatch! {debug_msg}")

        self.assertIsNotNone(m.X, msg=f"Solution X is None! {debug_msg}")
        self.assertAlmostEqual(m.X[0], 0.5, places=4, msg=f"X[0] mismatch! {debug_msg}")
        self.assertAlmostEqual(m.X[1], 0.5, places=4, msg=f"X[1] mismatch! {debug_msg}")

    def test_mixed_qp_sparse_and_lowrank(self):
        """
        Test Mixed QP: Minimize 0.5 * x^T Q x + 0.5 * ||Rx||^2 + c^T x
        Subject to: x1 + x2 = 1, x >= 0
        """
        print("\n[Test] Mixed QP (Sparse Q + Low-Rank R)")

        Q = sp.diags([2.0, 2.0], format="csc")
        R = sp.eye(2, format="csc")
        c = np.array([0.0, 0.0])
        A = np.array([[1.0, 1.0]])
        l = np.array([1.0])
        u = np.array([1.0])
        lb = np.array([0.0, 0.0])

        m = Model(
            objective_vector=c,
            constraint_matrix=A,
            constraint_lower_bound=l,
            constraint_upper_bound=u,
            objective_matrix=Q,
            objective_matrix_low_rank=R,
            variable_lower_bound=lb,
        )

        m.setParams(**self.default_params)
        m.optimize()

        debug_msg = self._get_debug_msg(m)
        self.assertEqual(m.Status, "OPTIMAL", msg=f"Status mismatch! {debug_msg}")
        self.assertAlmostEqual(m.ObjVal, 0.75, places=4, msg=f"ObjVal mismatch! {debug_msg}")

        self.assertIsNotNone(m.X, msg=f"Solution X is None! {debug_msg}")
        self.assertAlmostEqual(m.X[0], 0.5, places=4, msg=f"X[0] mismatch! {debug_msg}")
        self.assertAlmostEqual(m.X[1], 0.5, places=4, msg=f"X[1] mismatch! {debug_msg}")

    def test_qp_no_constraints_A(self):
        """
        Test QP without matrix A (Box constrained or Unconstrained).
        """
        print("\n[Test] QP without Constraint Matrix A")

        Q = sp.diags([2.0, 2.0], format="csc")
        c = np.array([-2.0, -2.0])

        m = Model(
            objective_vector=c,
            objective_matrix=Q,
            constraint_matrix=None,
            constraint_lower_bound=None,
            constraint_upper_bound=None,
            variable_lower_bound=None,
            variable_upper_bound=None,
        )

        m.setParams(**self.default_params)
        m.optimize()

        debug_msg = self._get_debug_msg(m)
        self.assertEqual(m.Status, "OPTIMAL", msg=f"Status mismatch! {debug_msg}")
        self.assertAlmostEqual(m.ObjVal, -2.0, places=4, msg=f"ObjVal mismatch! {debug_msg}")

        self.assertIsNotNone(m.X, msg=f"Solution X is None! {debug_msg}")
        self.assertAlmostEqual(m.X[0], 1.0, places=4, msg=f"X[0] mismatch! {debug_msg}")
        self.assertAlmostEqual(m.X[1], 1.0, places=4, msg=f"X[1] mismatch! {debug_msg}")

    def test_qp_a_simple_minimization(self):
        """
        Existing test: Minimize 0.5 * x^T Q x + c^T x s.t. x1+x2=1, x>=0
        Q = diag(4, 2), c = [1, 1]
        """
        print("\n[Test] Simple QP Minimization (Sparse Q)")
        Q = sp.diags([4.0, 2.0], format="csc")
        c = np.array([1.0, 1.0])
        A = np.array([[1.0, 1.0]])
        l = np.array([1.0])
        u = np.array([1.0])
        lb = np.array([0.0, 0.0])

        m = Model(
            objective_matrix=Q,
            objective_vector=c,
            constraint_matrix=A,
            constraint_lower_bound=l,
            constraint_upper_bound=u,
            variable_lower_bound=lb,
        )

        m.setParams(**self.default_params)
        m.optimize()

        debug_msg = self._get_debug_msg(m)
        self.assertEqual(m.Status, "OPTIMAL", msg=f"Status mismatch! {debug_msg}")
        # Analytical result: 5/3 ~ 1.6667
        self.assertAlmostEqual(m.ObjVal, 5.0 / 3.0, places=4, msg=f"ObjVal mismatch! {debug_msg}")

        self.assertIsNotNone(m.X, msg=f"Solution X is None! {debug_msg}")
        self.assertAlmostEqual(m.X[0], 1.0 / 3.0, places=4, msg=f"X[0] mismatch! {debug_msg}")
        self.assertAlmostEqual(m.X[1], 2.0 / 3.0, places=4, msg=f"X[1] mismatch! {debug_msg}")


if __name__ == "__main__":
    # 使用 verbosity=2 可以打印出每一个正在跑的测试用例名字和结果(OK / FAIL)
    unittest.main(verbosity=2)
