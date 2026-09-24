// Theme toggle (persisted). Default: light (LoanLens cream canvas).
(function () {
  const root = document.documentElement;
  const saved = localStorage.getItem("loanlens-theme") || localStorage.getItem("loanml-theme") || "light";
  root.setAttribute("data-theme", saved);

  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("themeBtn");
    if (!btn) return;
    updateIcon(saved);
    btn.addEventListener("click", () => {
      const cur = root.getAttribute("data-theme");
      const next = cur === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      localStorage.setItem("loanlens-theme", next);
      updateIcon(next);
    });
    function updateIcon(mode) {
      btn.setAttribute("aria-pressed", mode === "dark" ? "true" : "false");
      btn.innerHTML = mode === "dark"
        ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M21 12.8A9 9 0 1111.2 3 7 7 0 0021 12.8z"/></svg>'
        : '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
    }
  });
})();

// Mobile sidebar: pointer, keyboard, overlay
document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("sbToggle");
  const sb = document.getElementById("sidebar");
  const backdrop = document.getElementById("sbBackdrop");
  if (!btn || !sb) return;

  function setOpen(open) {
    sb.classList.toggle("open", open);
    btn.setAttribute("aria-expanded", open ? "true" : "false");
    btn.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
    if (backdrop) {
      backdrop.classList.toggle("open", open);
      backdrop.hidden = !open;
    }
  }

  btn.addEventListener("click", () => setOpen(!sb.classList.contains("open")));
  if (backdrop) backdrop.addEventListener("click", () => setOpen(false));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && sb.classList.contains("open")) {
      setOpen(false);
      btn.focus();
    }
  });
});

// Number counter animation for stat cards
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-count]").forEach((el) => {
    const target = parseFloat(el.dataset.count);
    const suffix = el.dataset.suffix || "";
    const decimals = el.dataset.count.includes(".") ? 1 : 0;
    let cur = 0;
    const steps = 30;
    const inc = target / steps;
    const iv = setInterval(() => {
      cur += inc;
      if (cur >= target) { cur = target; clearInterval(iv); }
      el.textContent = cur.toFixed(decimals) + suffix;
    }, 20);
  });
});
