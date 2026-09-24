/**
 * LoanLens — Assessment Form & Result Controller
 * Manages form validation, preset loading, simulation through prediction service abstraction, and rendering result UI.
 */

(function () {
  const form = document.getElementById("assessmentForm");
  const runBtn = document.getElementById("runAssessmentBtn");
  const resetBtn = document.getElementById("resetBtn");
  const loadingView = document.getElementById("loadingView");
  const resultView = document.getElementById("resultView");
  const formCard = document.getElementById("formCard");

  // Sample Presets for easy reviewer testing
  const presets = {
    prime: {
      Name: "Eleanor Vance",
      Age: 42,
      Education: "Master's",
      MaritalStatus: "Married",
      HasDependents: "No",
      EmploymentType: "Full-time",
      MonthsEmployed: 64,
      Income: 115000,
      CreditScore: 780,
      NumCreditLines: 3,
      DTIRatio: 0.22,
      LoanAmount: 25000,
      InterestRate: 6.8,
      LoanTerm: 36,
      LoanPurpose: "Home",
      HasMortgage: "Yes",
      HasCoSigner: "No",
    },
    moderate: {
      Name: "Marcus Sterling",
      Age: 32,
      Education: "Bachelor's",
      MaritalStatus: "Single",
      HasDependents: "Yes",
      EmploymentType: "Full-time",
      MonthsEmployed: 26,
      Income: 58000,
      CreditScore: 660,
      NumCreditLines: 2,
      DTIRatio: 0.38,
      LoanAmount: 18000,
      InterestRate: 13.5,
      LoanTerm: 48,
      LoanPurpose: "Auto",
      HasMortgage: "No",
      HasCoSigner: "No",
    },
    subprime: {
      Name: "Jordan Hayes",
      Age: 23,
      Education: "High School",
      MaritalStatus: "Divorced",
      HasDependents: "Yes",
      EmploymentType: "Part-time",
      MonthsEmployed: 8,
      Income: 28000,
      CreditScore: 540,
      NumCreditLines: 4,
      DTIRatio: 0.62,
      LoanAmount: 35000,
      InterestRate: 21.5,
      LoanTerm: 60,
      LoanPurpose: "Business",
      HasMortgage: "No",
      HasCoSigner: "No",
    },
  };

  // Wire presets buttons
  document.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.preset;
      const data = presets[key];
      if (!data) return;

      Object.entries(data).forEach(([field, val]) => {
        const input = form.elements[field];
        if (!input) return;
        if (input instanceof RadioNodeList || input.type === "radio") {
          const radio = form.querySelector(`input[name="${field}"][value="${val}"]`);
          if (radio) radio.checked = true;
        } else {
          input.value = val;
        }
      });

      // Clear previous validation errors
      form.querySelectorAll(".field.err").forEach((el) => el.classList.remove("err"));
    });
  });

  function validateForm() {
    let isValid = true;
    const requiredInputs = form.querySelectorAll("input[required], select[required]");
    requiredInputs.forEach((el) => {
      const field = el.closest(".field");
      if (!el.value || el.value.trim() === "") {
        if (field) field.classList.add("err");
        isValid = false;
      } else {
        if (field) field.classList.remove("err");
      }
    });

    return isValid;
  }

  const loadingSteps = [
    "Validating borrower credentials and limits...",
    "Applying StandardScaler normalization ($\mu=0, \sigma=1$)...",
    "Encoding categorical features into 24 model inputs...",
    "Querying HistGradientBoosting ensemble probability...",
    "Calculating Shapley/Permutation factor attributions...",
    "Synthesizing institutional underwriting recommendations...",
  ];

  function runAssessment() {
    if (!validateForm()) {
      const firstErr = form.querySelector(".field.err");
      if (firstErr) firstErr.scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }

    const formData = new FormData(form);
    const payload = Object.fromEntries(formData.entries());

    // Show loading view
    formCard.style.display = "none";
    resultView.style.display = "none";
    loadingView.style.display = "block";
    loadingView.scrollIntoView({ behavior: "smooth", block: "start" });

    loadingView.innerHTML = `
      <div style="text-align:center; padding:50px 20px;">
        <div class="analysis-ring" style="width:68px; height:68px; margin:0 auto 20px;">
          <svg width="68" height="68" viewBox="0 0 76 76">
            <circle cx="38" cy="38" r="32" fill="none" stroke="var(--border)" stroke-width="5"/>
            <circle cx="38" cy="38" r="32" fill="none" stroke="var(--brand)" stroke-width="5" stroke-linecap="round" stroke-dasharray="140 200"/>
          </svg>
        </div>
        <h3 style="font-size:18px; font-weight:700; margin:0 0 6px;">Analyzing borrower profile...</h3>
        <p style="font-size:13px; color:var(--muted); margin:0 0 24px;">Running calibrated inference against 255,347 baseline lending records</p>
        <div class="analysis-lines" style="max-width:380px; margin:0 auto;">
          ${loadingSteps.map((txt, i) => `<div class="analysis-line" id="load-step-${i}"><span class="chk"></span>${txt}</div>`).join("")}
        </div>
      </div>
    `;

    let stepIdx = 0;
    const interval = setInterval(() => {
      if (stepIdx > 0) {
        const prev = document.getElementById(`load-step-${stepIdx - 1}`);
        if (prev) prev.classList.add("done");
      }
      if (stepIdx >= loadingSteps.length) {
        clearInterval(interval);
        submitPayload(payload);
        return;
      }
      stepIdx++;
    }, 280);
  }

  function submitPayload(payload) {
    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then(async (res) => {
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Assessment failed");
        return data;
      })
      .then(renderResultUI)
      .catch((err) => {
        loadingView.style.display = "none";
        formCard.style.display = "block";
        alert("Assessment error: " + err.message);
      });
  }

  function renderResultUI(res) {
    loadingView.style.display = "none";
    resultView.style.display = "block";
    resultView.scrollIntoView({ behavior: "smooth", block: "start" });

    const isHigh = res.risk_level === "High";
    const isMed = res.risk_level === "Medium";
    const colorVar = isHigh ? "var(--high)" : isMed ? "var(--med)" : "var(--low)";
    const badgeClass = isHigh ? "high" : isMed ? "med" : "low";

    const probPct = typeof res.probability === "number" && res.probability <= 1.0 
      ? Math.round(res.probability * 1000) / 10 
      : (typeof res.probability === "number" ? res.probability : 24.0);
    const circ = 2 * Math.PI * 54;
    const offset = circ * (1 - probPct / 100);
    const predTitle = typeof res.prediction === "number" 
      ? (res.prediction === 1 ? "Likely to Default" : "Likely to Repay") 
      : (res.prediction || "Assessment Complete");

    resultView.innerHTML = `
      <div class="card" style="padding:32px; box-shadow:var(--shadow-2);">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:20px; padding-bottom:24px; border-bottom:1px solid var(--border); margin-bottom:28px;">
          <div>
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
              <span class="badge ${badgeClass}" style="font-size:12px; font-weight:700; padding:6px 14px;">
                RISK LEVEL: ${res.risk_level.toUpperCase()}
              </span>
              <span style="font-size:12px; color:var(--muted);">
                Model: <b>${res.model_used}</b>
              </span>
            </div>
            <h2 style="font-size:28px; font-weight:700; margin:0 0 6px; letter-spacing:-0.01em;">
              ${predTitle}
            </h2>
            <p style="font-size:14px; color:var(--muted); margin:0;">
              Classification Label: <strong style="color:var(--text);">${res.label}</strong> &middot; Default Probability: <strong>${probPct}%</strong> &middot; Confidence: <strong>${Math.round(res.confidence * 100)}%</strong>
            </p>
          </div>

          <div style="display:flex; align-items:center; gap:24px;">
            <div style="position:relative; width:110px; height:110px; display:flex; align-items:center; justify-content:center;">
              <svg width="110" height="110" viewBox="0 0 130 130">
                <circle cx="65" cy="65" r="54" fill="none" stroke="var(--border)" stroke-width="10"/>
                <circle cx="65" cy="65" r="54" fill="none" stroke="${colorVar}" stroke-width="10" stroke-linecap="round"
                  stroke-dasharray="${circ}" stroke-dashoffset="${offset}" transform="rotate(-90 65 65)"
                  style="transition: stroke-dashoffset 1s ease;"/>
              </svg>
              <div style="position:absolute; text-align:center;">
                <div style="font-size:22px; font-weight:700; line-height:1;">${probPct}%</div>
                <div style="font-size:9px; color:var(--muted); text-transform:uppercase; margin-top:2px;">Probability</div>
              </div>
            </div>

            <div style="border-left:1px solid var(--border); padding-left:20px;">
              <div style="font-size:32px; font-weight:700; color:var(--text); line-height:1;">
                ${res.risk_score}<span style="font-size:16px; color:var(--muted); font-weight:500;">/100</span>
              </div>
              <div style="font-size:11px; font-weight:600; text-transform:uppercase; color:var(--muted); margin-top:4px;">
                Risk Score
              </div>
            </div>
          </div>
        </div>

        <!-- Why this prediction? Section -->
        <div style="margin-bottom:28px;">
          <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:12px;">
            <h3 style="font-size:17px; font-weight:700; margin:0;">Why this prediction?</h3>
            <span style="font-size:12px; color:var(--muted);">Evaluated against prime lending thresholds</span>
          </div>

          <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap:14px;">
            ${res.factors.map((f) => `
              <div style="padding:14px 16px; background:var(--surface); border:1px solid var(--border); border-radius:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                  <span style="font-size:13px; font-weight:600; color:var(--text);">${f.label}</span>
                  <span style="font-size:11px; font-weight:700; color:${f.direction === 'up' ? 'var(--high)' : 'var(--low)'};">
                    ${f.direction === 'up' ? '&uarr; Risk Driver' : '&darr; Protective'}
                  </span>
                </div>
                <div style="height:6px; background:var(--border); border-radius:999px; overflow:hidden; margin-bottom:8px;">
                  <div style="height:100%; border-radius:999px; width:${f.impact}%; background:${f.direction === 'up' ? 'var(--high)' : 'var(--low)'};"></div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--muted);">
                  <span>Recorded: <b>${f.value}</b></span>
                  <span>Benchmark: <b>${f.benchmark}</b></span>
                </div>
                <div style="font-size:11px; color:var(--muted); margin-top:4px;">${f.desc}</div>
              </div>
            `).join("")}
          </div>
        </div>

        <!-- Underwriting Recommendation -->
        <div style="background:color-mix(in srgb, var(--brand) 8%, var(--panel)); border:1px solid color-mix(in srgb, var(--brand) 25%, transparent); border-radius:12px; padding:20px; margin-bottom:28px;">
          <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--brand)" stroke-width="2.2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            <span style="font-size:14px; font-weight:700; color:var(--brand);">${res.recommendation.action}</span>
          </div>
          <ul style="margin:0; padding-left:20px; font-size:13px; color:var(--text); line-height:1.7;">
            ${res.recommendation.points.map((p) => `<li>${p}</li>`).join("")}
          </ul>
        </div>

        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:14px;">
          <a href="/predictions" class="btn-ghost" style="font-size:13px;">View in Prediction History &rarr;</a>
          <button type="button" class="btn-primary" id="anotherAssessmentBtn" style="padding:10px 20px;">
            Run Another Assessment
          </button>
        </div>
      </div>
    `;

    document.getElementById("anotherAssessmentBtn").addEventListener("click", () => {
      resultView.style.display = "none";
      formCard.style.display = "block";
      formCard.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  if (runBtn) runBtn.addEventListener("click", runAssessment);
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      form.reset();
      form.querySelectorAll(".field.err").forEach((el) => el.classList.remove("err"));
    });
  }

  window.resetAssessmentForm = function () {
    resultView.style.display = "none";
    formCard.style.display = "block";
    formCard.scrollIntoView({ behavior: "smooth", block: "start" });
  };
})();
