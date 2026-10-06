/*
Copyright 2026 Hongpei Li

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

/* CPU-private BLAS scope. Pair on the same host thread, including on failure.
 * Nested/overlapping scopes keep BLAS single-threaded until their last exit.
 * The token preserves MKL's thread-local override (0 means inherit global).
 * OpenBLAS has library-wide settings: applications must synchronize unrelated
 * calls/settings that share the same library instance with these scopes. */
int pdhcg_cpu_blas_enter(void);
void pdhcg_cpu_blas_leave(int previous);
