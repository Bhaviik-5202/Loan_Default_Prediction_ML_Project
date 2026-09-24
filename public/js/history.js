(function () {
  const PAGE_SIZE = 10;
  let page = 1;
  const search = document.getElementById("searchInput");
  const riskFilter = document.getElementById("riskFilter");
  const sortSelect = document.getElementById("sortSelect");
  const body = document.getElementById("tableBody");
  const pag = document.getElementById("pagination");

  function badgeClass(level) {
    return level === "Low" ? "low" : level === "Medium" ? "med" : "high";
  }

  function getFiltered() {
    let rows = window.APPLICATIONS.slice();
    const q = search.value.trim().toLowerCase();
    if (q) rows = rows.filter(a => a.name.toLowerCase().includes(q) || a.id.toLowerCase().includes(q));
    if (riskFilter.value) rows = rows.filter(a => a.risk_level === riskFilter.value);

    const sort = sortSelect.value;
    rows.sort((a, b) => {
      if (sort === "date-desc") return a.date < b.date ? 1 : -1;
      if (sort === "date-asc") return a.date > b.date ? 1 : -1;
      if (sort === "prob-desc") return b.probability - a.probability;
      if (sort === "prob-asc") return a.probability - b.probability;
      if (sort === "credit-desc") return b.credit_score - a.credit_score;
      return 0;
    });
    return rows;
  }

  function render() {
    const rows = getFiltered();
    const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
    page = Math.min(page, totalPages);
    const slice = rows.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

    if (slice.length === 0) {
      body.innerHTML = `<tr><td colspan="9" style="text-align:center;color:var(--muted);padding:32px">No applications match your filters.</td></tr>`;
    } else {
      body.innerHTML = slice.map(a => `
        <tr>
          <td style="font-family:var(--mono);font-size:.78rem;color:var(--muted)">${a.id}</td>
          <td>${a.name}</td>
          <td>${a.credit_score}</td>
          <td>$${a.loan_amount.toLocaleString()}</td>
          <td>${a.dti}</td>
          <td><span class="badge ${badgeClass(a.risk_level)}">${a.risk_level}</span></td>
          <td>${a.probability}%</td>
          <td>${a.prediction}</td>
          <td>${a.date}</td>
        </tr>`).join("");
    }

    let pagHTML = "";
    for (let i = 1; i <= totalPages; i++) {
      pagHTML += `<button data-p="${i}" class="${i === page ? 'on' : ''}">${i}</button>`;
    }
    pag.innerHTML = pagHTML;
    pag.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { page = +b.dataset.p; render(); }));
  }

  [search, riskFilter, sortSelect].forEach(el => el.addEventListener("input", () => { page = 1; render(); }));
  render();
})();
