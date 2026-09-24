function donutChart(el, segments, size = 160, thickness = 22) {
  const total = segments.reduce((s, x) => s + x.value, 0);
  const divisor = total || 1;
  const r = (size - thickness) / 2;
  const cx = size / 2, cy = size / 2;
  const circ = 2 * Math.PI * r;
  let offset = 0;
  let svg = `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
    <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="var(--border)" stroke-width="${thickness}"/>`;
  segments.forEach((seg) => {
    const frac = seg.value / divisor;
    const len = circ * frac;
    svg += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${seg.color}" stroke-width="${thickness}"
      stroke-dasharray="${len} ${circ - len}" stroke-dashoffset="${-offset}" transform="rotate(-90 ${cx} ${cy})"
      stroke-linecap="butt"/>`;
    offset += len;
  });
  svg += `<text x="${cx}" y="${cy - 2}" text-anchor="middle" font-size="20" font-weight="700" fill="var(--text)" font-family="var(--mono)">${total}</text>
    <text x="${cx}" y="${cy + 16}" text-anchor="middle" font-size="10" fill="var(--muted)">applications</text></svg>`;
  el.innerHTML = svg;
}

function lineChart(el, points, width = 520, height = 180) {
  const max = Math.max(...points.map((p) => p.count), 1);
  const step = width / (points.length - 1);
  const toXY = (p, i) => [i * step, height - 24 - (p.count / max) * (height - 44)];
  const linePts = points.map(toXY);
  const path = linePts.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(" ");
  const area = `${path} L${width},${height - 24} L0,${height - 24} Z`;

  const defPts = points.map((p, i) => toXY(p, i));
  const dPath = defPts.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(" ");

  let svg = `<svg width="100%" height="${height}" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
    <defs><linearGradient id="lg1" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="var(--brand)" stop-opacity="0.25"/>
      <stop offset="100%" stop-color="var(--brand)" stop-opacity="0"/>
    </linearGradient></defs>
    <path d="${area}" fill="url(#lg1)"/>
    <path d="${path}" fill="none" stroke="var(--brand)" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>`;

  points.forEach((p, i) => {
    const [x, y] = toXY(p, i);
    if (i % 2 === 0) svg += `<text x="${x}" y="${height - 6}" font-size="9" fill="var(--muted-2)" text-anchor="middle">${p.date}</text>`;
  });
  svg += "</svg>";
  el.innerHTML = svg;
}
