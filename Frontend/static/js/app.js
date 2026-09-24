// Theme toggle (persisted)
(function () {
  const root = document.documentElement;
  const saved = localStorage.getItem("loanml-theme") || "dark";
  root.setAttribute("data-theme", saved);

  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("themeBtn");
    if (!btn) return;
    updateIcon(saved);
    btn.addEventListener("click", () => {
      const cur = root.getAttribute("data-theme");
      const next = cur === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      localStorage.setItem("loanml-theme", next);
      updateIcon(next);
    });
    function updateIcon(mode) {
      btn.innerHTML = mode === "dark"
        ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.8A9 9 0 1111.2 3 7 7 0 0021 12.8z"/></svg>'
        : '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
    }
  });
})();

// Mobile sidebar toggle
document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("sbToggle");
  const sb = document.getElementById("sidebar");
  if (btn && sb) {
    btn.addEventListener("click", () => sb.classList.toggle("open"));
    document.addEventListener("click", (e) => {
      if (sb.classList.contains("open") && !sb.contains(e.target) && e.target !== btn && !btn.contains(e.target)) {
        sb.classList.remove("open");
      }
    });
  }
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

// Hero mini-visual: animate ring + bars on load
document.addEventListener("DOMContentLoaded", () => {
  const ring = document.getElementById("heroRing");
  if (ring) {
    const circ = 2 * Math.PI * 34;
    const pct = 0.187; // demo figure shown on the hero
    ring.style.strokeDasharray = circ;
    ring.style.strokeDashoffset = circ;
    requestAnimationFrame(() => {
      ring.style.transition = "stroke-dashoffset 1.4s cubic-bezier(.2,.8,.2,1)";
      ring.style.strokeDashoffset = circ * (1 - pct);
    });
  }
  document.querySelectorAll(".mini-fill").forEach((el) => {
    const w = el.dataset.width;
    setTimeout(() => { el.style.width = w + "%"; }, 200);
  });
});
