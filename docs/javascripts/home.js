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

(() => {
  const toggle = document.querySelector("[data-formulation-toggle]");
  if (!toggle) return;

  const compact = document.getElementById("compact-formulation");
  const full = document.getElementById("full-formulation");
  const label = toggle.querySelector("[data-formulation-label]");

  toggle.addEventListener("click", () => {
    const expanded = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", String(expanded));
    compact.hidden = expanded;
    full.hidden = !expanded;
    label.textContent = expanded
      ? "Show compact formulation"
      : "Show full formulation";
  });
})();
