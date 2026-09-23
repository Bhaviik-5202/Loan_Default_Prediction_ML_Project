(function () {
  const steps = ["step-1", "step-2", "step-3", "step-4", "step-5"];
  let current = 0;

  const dots = document.querySelectorAll(".step-dot");
  const labels = document.querySelectorAll(".steps-labels span");
  const prevBtn = document.getElementById("prevBtn");
  const nextBtn = document.getElementById("nextBtn");
  const form = document.getElementById("wizardForm");

  function show(i) {
    steps.forEach((id, idx) => {
      document.getElementById(id).classList.toggle("active", idx === i);
      dots[idx].classList.toggle("active", idx === i);
      dots[idx].classList.toggle("done", idx < i);
      labels[idx].classList.toggle("on", idx <= i);
    });
    prevBtn.style.visibility = i === 0 ? "hidden" : "visible";
    nextBtn.textContent = i === steps.length - 1 ? "Predict Default Risk" : "Continue";
  }

  function validateStep(i) {
    const view = document.getElementById(steps[i]);
    let ok = true;
    view.querySelectorAll("input[required], select[required]").forEach((el) => {
      const field = el.closest(".field");
      if (!el.value) { field.classList.add("err"); ok = false; }
      else field.classList.remove("err");
    });
    return ok;
  }

  nextBtn.addEventListener("click", () => {
    if (!validateStep(current)) return;
    if (current < steps.length - 1) {
      current++;
      show(current);
    } else {
      runPrediction();
    }
  });

  prevBtn.addEventListener("click", () => {
    if (current > 0) { current--; show(current); }
  });

  show(0);

  // ---------------- analysis animation + prediction call ----------------

  const analysisLines = [
    "Collecting applicant data",
    "Validating financial profile",
    "Analyzing credit behavior",
    "Running ML prediction",
    "Calculating risk factors",
    "Generating risk explanation",
  ];

  function runPrediction() {
    document.getElementById("wizardPanel").style.display = "none";
    const av = document.getElementById("analysisView");
    av.classList.add("active");
    av.innerHTML =
      '<div class="analysis-ring"><svg width="76" height="76" viewBox="0 0 76 76"><circle cx="38" cy="38" r="32" fill="none" stroke="var(--border)" stroke-width="5"/><circle cx="38" cy="38" r="32" fill="none" stroke="var(--brand)" stroke-width="5" stroke-linecap="round" stroke-dasharray="140 200"/></svg></div>' +
      '<div class="analysis-lines">' +
      analysisLines.map((t, i) => `<div class="analysis-line" id="al-${i}"><span class="chk"></span>${t}...</div>`).join("") +
      "</div>";

    const data = Object.fromEntries(new FormData(form).entries());

    let i = 0;
    const iv = setInterval(() => {
      if (i > 0) document.getElementById(`al-${i - 1}`).classList.add("done");
      if (i >= analysisLines.length) {
        clearInterval(iv);
        fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        })
          .then((r) => r.json())
          .then(renderResult)
          .catch(() => renderResult(null));
        return;
      }
      i++;
    }, 420);
  }

  function riskClass(level) {
    return level === "High" ? "high" : level === "Medium" ? "med" : "low";
  }

  function renderResult(res) {
    document.getElementById("analysisView").classList.remove("active");
    const rv = document.getElementById("resultView");
    rv.classList.add("active");

    if (!res) {
      rv.innerHTML = '<p style="color:var(--high)">Could not reach the prediction service. Please try again.</p>';
      return;
    }

    const cls = riskClass(res.risk_level);
    const circ = 2 * Math.PI * 52;
    const offset = circ * (1 - res.probability / 100);

    rv.innerHTML = `
      <div class="result-top">
        <div class="gauge-big">
          <svg width="150" height="150" viewBox="0 0 150 150">
            <circle cx="75" cy="75" r="52" fill="none" stroke="var(--border)" stroke-width="12"/>
            <circle cx="75" cy="75" r="52" fill="none" stroke="var(--${cls === 'high' ? 'high' : cls === 'med' ? 'med' : 'low'})"
              stroke-width="12" stroke-linecap="round" stroke-dasharray="${circ}" stroke-dashoffset="${circ}"
              transform="rotate(-90 75 75)" id="gaugeCircle"/>
          </svg>
          <div class="center"><div class="pct">${res.probability}%</div><div class="pct-lbl">Default Probability</div></div>
        </div>
        <div>
          <span class="risk-tag ${cls}">${res.risk_level} Risk</span>
          <div class="result-stats">
            <div class="rs"><span class="v">${(100 - res.probability).toFixed(1)}%</span><span class="k">Repayment Probability</span></div>
            <div class="rs"><span class="v">${res.risk_score}/100</span><span class="k">Risk Score</span></div>
            <div class="rs"><span class="v">${res.prediction}</span><span class="k">Prediction</span></div>
          </div>
        </div>
      </div>

      <div class="two-col">
        <div class="card-sm">
          <h3>Why this prediction?</h3>
          ${res.factors.map(f => `
            <div class="factor">
              <div class="factor-top"><span>${f.label}</span><span class="v">${f.impact}%</span></div>
              <div class="factor-track"><div class="factor-fill ${f.direction}" data-w="${f.impact}"></div></div>
            </div>`).join("")}
        </div>
        <div class="card-sm">
          <h3>Applicant Risk Profile</h3>
          <div class="radar-wrap">${radarSVG(res.profile)}</div>
        </div>
      </div>

      <div class="reco-box">
        <div class="head">Recommended Action: ${res.recommendation.action}</div>
        <ul>${res.recommendation.points.map(p => `<li>${p}</li>`).join("")}</ul>
        <div class="note">System-generated decision-support information — not guaranteed financial advice.</div>
      </div>

      <div class="result-actions">
        <button class="btn-primary" onclick="location.reload()">Run Another Assessment</button>
        <a class="btn-ghost" href="/">Back to Home</a>
      </div>
    `;

    requestAnimationFrame(() => {
      document.getElementById("gaugeCircle").style.transition = "stroke-dashoffset 1.1s cubic-bezier(.2,.8,.2,1)";
      document.getElementById("gaugeCircle").style.strokeDashoffset = offset;
      document.querySelectorAll(".factor-fill").forEach(el => {
        setTimeout(() => { el.style.width = el.dataset.w + "%"; }, 150);
      });
    });
  }

  function radarSVG(profile) {
    const keys = Object.keys(profile);
    const n = keys.length;
    const cx = 100, cy = 100, r = 78;
    const pts = keys.map((k, i) => {
      const ang = (Math.PI * 2 * i) / n - Math.PI / 2;
      const val = profile[k] / 100;
      return [cx + r * val * Math.cos(ang), cy + r * val * Math.sin(ang)];
    });
    const poly = pts.map(p => p.join(",")).join(" ");
    const rings = [0.33, 0.66, 1].map(f => {
      const rp = keys.map((k, i) => {
        const ang = (Math.PI * 2 * i) / n - Math.PI / 2;
        return [cx + r * f * Math.cos(ang), cy + r * f * Math.sin(ang)].join(",");
      }).join(" ");
      return `<polygon points="${rp}" fill="none" stroke="var(--border)" stroke-width="1"/>`;
    }).join("");
    const labelsHTML = keys.map((k, i) => {
      const ang = (Math.PI * 2 * i) / n - Math.PI / 2;
      const lx = cx + (r + 22) * Math.cos(ang);
      const ly = cy + (r + 22) * Math.sin(ang);
      return `<text x="${lx}" y="${ly}" font-size="9" fill="var(--muted)" text-anchor="middle">${k}</text>`;
    }).join("");
    return `<svg width="220" height="220" viewBox="0 0 200 200">
      ${rings}
      <polygon points="${poly}" fill="var(--brand)" fill-opacity="0.22" stroke="var(--brand)" stroke-width="2"/>
      ${labelsHTML}
    </svg>`;
  }
})();
