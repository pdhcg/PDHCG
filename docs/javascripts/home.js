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
