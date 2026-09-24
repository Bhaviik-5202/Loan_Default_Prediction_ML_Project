(function () {
  const ids = ["CreditScore", "Income", "LoanAmount", "InterestRate", "DTIRatio", "MonthsEmployed"];
  const fmt = { Income: (v) => Number(v).toLocaleString(), LoanAmount: (v) => Number(v).toLocaleString(),
    InterestRate: (v) => v + "%" };
  let lastPct = null;
  let timer = null;

  ids.forEach((id) => {
    const input = document.getElementById(id);
    input.addEventListener("input", () => {
      document.getElementById("v-" + id).textContent = fmt[id] ? fmt[id](input.value) : input.value;
      clearTimeout(timer);
      timer = setTimeout(run, 160);
    });
  });

  function run() {
    const data = {};
    ids.forEach((id) => { data[id] = document.getElementById(id).value; });

    fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    })
      .then(async (r) => {
        const body = await r.json();
        if (!r.ok) throw new Error(body.error || "Simulation failed.");
        return body;
      })
      .then((res) => {
        const circ = 326.7;
        const offset = circ * (1 - res.probability / 100);
        const color = res.risk_level === "High" ? "var(--high)" : res.risk_level === "Medium" ? "var(--med)" : "var(--low)";
        const gauge = document.getElementById("simGauge");
        gauge.style.stroke = color;
        gauge.style.strokeDashoffset = offset;
        document.getElementById("simPct").textContent = res.probability + "%";

        const tag = document.getElementById("simTag");
        tag.textContent = res.risk_level + " Risk";
        tag.className = "risk-tag " + (res.risk_level === "High" ? "high" : res.risk_level === "Medium" ? "med" : "low");

        const deltaEl = document.getElementById("simDelta");
        if (lastPct !== null && lastPct !== res.probability) {
          const dir = res.probability > lastPct ? "up" : "down";
          const diff = Math.abs(res.probability - lastPct).toFixed(1);
          deltaEl.innerHTML = `${lastPct}% <span style="color:var(--${dir === 'up' ? 'high' : 'low'})">→ ${res.probability}% (${dir === 'up' ? '+' : '-'}${diff})</span>`;
        }
        lastPct = res.probability;
      })
      .catch((error) => {
        document.getElementById("simDelta").textContent = error.message;
      });
  }

  run();
})();
