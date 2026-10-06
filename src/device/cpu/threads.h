/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

/* CPU-private BLAS scope. Pair on the same host thread, including on failure.
 * Nested/overlapping scopes keep BLAS single-threaded until their last exit.
 * The token preserves MKL's thread-local override (0 means inherit global).
 * OpenBLAS has library-wide settings: applications must synchronize unrelated
 * calls/settings that share the same library instance with these scopes. */
int pdhcg_cpu_blas_enter(void);
void pdhcg_cpu_blas_leave(int previous);
