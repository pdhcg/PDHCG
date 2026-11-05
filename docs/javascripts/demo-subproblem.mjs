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

// Conic examples use the actual two-dimensional primal objective and an
// exact planar section of the three-dimensional dual proximal objective.
// Legacy scalar examples use F(u,v) = f(u) + .45*h*v²/2 on C × R;
// their minimizer remains (the original scalar update, 0).
export function subproblemGeometry(p, r, dual) {
  if (p.kind === 'conic') {
    if (!dual) {
      const center = r.primalCenter, optimum = r.vectorXp, curvature = p.q + 1 / p.tau;
      return {center, optimum, curvature, shape: 'disk', radius: p.radius,
        value: (u, v) => .5 * curvature * ((u - center[0]) ** 2 + (v - center[1]) ** 2)};
    }
    // SOC projection preserves the direction of the spatial component. This
    // is its exact 2D section (radial coordinate, cone height), not a new solve.
    const norm = Math.hypot(...r.dualCenter.slice(1));
    const direction = norm > 1e-15 ? r.dualCenter.slice(1).map(v => v / norm) : [1, 0];
    const center = [norm, r.dualCenter[0]];
    const optimum = [r.vectorYp.slice(1).reduce((s, v, i) => s + v * direction[i], 0), r.vectorYp[0]];
    const curvature = 1 / p.sigma;
    return {center, optimum, curvature, direction, shape: 'cone',
      value: (u, v) => .5 * curvature * ((u - center[0]) ** 2 + (v - center[1]) ** 2)};
  }
  const curvature = dual ? 1 / p.sigma : p.q + 1 / p.tau;
  const center = dual ? r.trialY : (p.q * p.a - r.y + r.x / p.tau) / curvature;
  const optimum = dual ? r.yp : r.xp;
  const lower = dual ? (p.constraint === 'equality' ? -Infinity : 0) : p.lower;
  const upper = dual ? Infinity : p.upper;
  return {center, optimum, lower, upper, curvature,
    value: (u, v) => .5 * curvature * ((u - center) ** 2 + .45 * v ** 2)};
}

export function drawSubproblem(ctx, W, H, p, model, colors, compact = false, {hideLabels = false} = {}) {
  const dual = model.scope === 'dual', g = subproblemGeometry(p, model.r, dual);
  const conic = p.kind === 'conic', accent = dual ? colors.teal : colors.orange;
  const name = dual ? 'y' : 'x';
  const center = conic ? g.center : [g.center, 0];
  const optimum = conic ? g.optimum : [g.optimum, 0];
  const active = Math.hypot(...center.map((v, i) => v - optimum[i])) > 1e-8;
  const progress = Math.max(0, Math.min(1, model.solveProgress ?? 1));
  const reveal = Math.max(0, Math.min(1, (progress - .82) / .18));
  let lo, hi;
  if (g.shape === 'disk') {
    lo = [0, 1].map(i => Math.min(-g.radius, center[i]) - .35 * g.radius);
    hi = [0, 1].map(i => Math.max(g.radius, center[i]) + .35 * g.radius);
  } else if (g.shape === 'cone') {
    const extent = Math.max(.1, ...center.map(Math.abs), ...optimum.map(Math.abs));
    lo = [-.65 * extent, Math.min(-.25 * extent, center[1] - .3 * extent)];
    hi = [Math.max(center[0], optimum[0]) + .6 * extent, Math.max(center[1], optimum[1], .5 * extent) + .6 * extent];
  } else {
    const start = dual ? model.r.y : model.r.x;
    const scale = Math.max(.15, Math.abs(start - g.center), Math.abs(g.optimum - g.center) * 3);
    const nearest = [g.lower, g.upper].filter(Number.isFinite)
      .sort((a, b) => Math.abs(a - g.center) - Math.abs(b - g.center))[0];
    const included = [start, g.center, g.optimum];
    if (nearest !== undefined) included.push(nearest);
    const low = Math.min(...included) - scale * 1.25, high = Math.max(...included) + scale * 1.25;
    lo = [low, -(high - low) * .45]; hi = [high, (high - low) * .45];
  }
  const padding = W < 500 ? 24 : 52, top = compact ? 58 : 60, bottom = compact ? 68 : 96;
  const width = W - 2 * padding, height = H - top - bottom;
  const origin = lo.map((v, i) => (v + hi[i]) / 2);
  const span = Math.max(hi[0] - lo[0], hi[1] - lo[1]);
  const unit = Math.min(width / (hi[0] - lo[0]), height / (hi[1] - lo[1]));
  const P = (u, v) => [W / 2 + unit * (u - origin[0]), top + height / 2 - unit * (v - origin[1])];
  const trace = points => { ctx.beginPath(); points.forEach((q, i) => i ? ctx.lineTo(...q) : ctx.moveTo(...q)); };
  const stroke = (points, color, alpha = 1, weight = 1.5, dash = []) => {
    trace(points); ctx.strokeStyle = color; ctx.globalAlpha = alpha; ctx.lineWidth = weight;
    ctx.setLineDash(dash); ctx.stroke(); ctx.globalAlpha = 1; ctx.setLineDash([]);
  };
  const label = (text, x, y, color = colors.ink, align = 'left', size = 13) => {
    ctx.font = `500 ${size}px system-ui, sans-serif`; ctx.textAlign = align;
    const measured = ctx.measureText(text).width;
    const left = align === 'center' ? measured / 2 : align === 'right' ? measured : 0;
    x = Math.max(12 + left, Math.min(W - 12 - measured + left, x));
    ctx.lineWidth = 5; ctx.strokeStyle = colors.paper; ctx.strokeText(text, x, y);
    ctx.fillStyle = color; ctx.fillText(text, x, y);
  };
  const ellipse = radius => Array.from({length: 121}, (_, i) => {
    const t = i * Math.PI / 60;
    return P(center[0] + radius * Math.cos(t), center[1] + radius / Math.sqrt(conic ? 1 : .45) * Math.sin(t));
  });
  const disk = g.shape === 'disk' ? Array.from({length: 121}, (_, i) => P(g.radius * Math.cos(i * Math.PI / 60), g.radius * Math.sin(i * Math.PI / 60))) : [];
  const extent = span * 5;
  let domain;
  if (g.shape === 'disk') domain = disk;
  else if (g.shape === 'cone') domain = [P(0, 0), P(extent, extent), P(-extent, extent)];
  else domain = [P(Math.max(lo[0] - extent, g.lower), -extent), P(Math.min(hi[0] + extent, g.upper), -extent),
    P(Math.min(hi[0] + extent, g.upper), extent), P(Math.max(lo[0] - extent, g.lower), extent)];

  ctx.save();
  ctx.beginPath(); ctx.rect(padding - 8, top, width + 16, height); ctx.clip();
  trace(domain); ctx.closePath(); ctx.fillStyle = accent; ctx.globalAlpha = .12; ctx.fill(); ctx.globalAlpha = 1;
  // Exact level sets of the displayed proximal objective. The colored level
  // touches the feasible set at the computed minimizer.
  const contactRadius = Math.sqrt(2 * g.value(...optimum) / g.curvature);
  // This is a changing objective level, never an inner-solver iterate.
  const levelRadius = contactRadius + .3 * span * (1 - progress);
  for (const radius of [.1, .19, .29, .4, .53, .68].map(v => v * span)) {
    trace(ellipse(radius)); ctx.closePath(); ctx.fillStyle = colors.ink;
    ctx.globalAlpha = .014; ctx.fill(); ctx.globalAlpha = 1;
    stroke(ellipse(radius), colors.ink, .18, 1.2);
  }
  if (g.shape === 'disk') stroke(disk, accent, .9, 2.3);
  else if (g.shape === 'cone') stroke([P(-extent, extent), P(0, 0), P(extent, extent)], accent, .9, 2.3);
  else for (const bound of [g.lower, g.upper].filter(Number.isFinite))
    stroke([P(bound, -extent), P(bound, extent)], accent, .9, 2.3);
  const centerPixel = P(...center), optimumPixel = P(...optimum);
  if (levelRadius > 1e-12) stroke(ellipse(levelRadius), accent, .9, 2.6);
  if (active) {
    stroke([centerPixel, optimumPixel], accent, .65 * progress, 1.5, [4, 5]);
    stroke([[centerPixel[0]-5,centerPixel[1]-5],[centerPixel[0]+5,centerPixel[1]+5]], colors.ink, .75, 1.6);
    stroke([[centerPixel[0]-5,centerPixel[1]+5],[centerPixel[0]+5,centerPixel[1]-5]], colors.ink, .75, 1.6);
  }
  if (reveal > 0) {
    const halo = ctx.createRadialGradient(...optimumPixel, 0, ...optimumPixel, 22);
    halo.addColorStop(0, accent + '40'); halo.addColorStop(1, accent + '00');
    ctx.globalAlpha = reveal;
    ctx.fillStyle = halo; ctx.beginPath(); ctx.arc(...optimumPixel, 22, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = accent; ctx.strokeStyle = colors.paper; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(...optimumPixel, 7, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    ctx.globalAlpha = 1;
  }
  ctx.restore();

  if (!hideLabels) {
    const small = W < 500;
    label('Proximal objective · level sets', padding, top + 17, colors.ink, 'left', small ? 11 : 13);
    const setPoint = g.shape === 'disk' ? P(0, -.45 * g.radius) : g.shape === 'cone' ? P(0, hi[1] * .65) :
      P(Math.min(Math.max(optimum[0] + span * .24, g.lower + span * .12), g.upper - span * .12), span * .24);
    const setLabel = g.shape === 'cone' ? 'Dual cone 𝒬₃*' : dual ? 'Dual domain D' : 'Primal domain C';
    label(setLabel, setPoint[0], Math.max(top + 50, Math.min(H - bottom - 12, setPoint[1])), accent, 'center', small ? 12 : 15);
    const outward = centerPixel[0] >= optimumPixel[0] ? 1 : -1;
    if (active) label('Unconstrained minimizer', centerPixel[0] + outward * 12, centerPixel[1] - 18,
      colors.ink, outward > 0 ? 'left' : 'right', small ? 10 : 12);
    ctx.globalAlpha = reveal;
    label(`${name}⁺`, optimumPixel[0] - outward * 14, optimumPixel[1] + 5, accent, outward > 0 ? 'right' : 'left', 20);
    ctx.globalAlpha = 1;
    label(`${name}⁺ · subproblem minimizer`, W / 2, H - bottom + 27, accent, 'center', small ? 11 : 14);
    const note = conic ? (dual ? 'Second-order cone · 2D cross-section' : '') : 'Scalar subproblem · 2D illustration';
    if (note) label(note, W / 2, H - bottom + 49, colors.ink, 'center', small ? 9 : 11);
  }
  return {geometry: g, screenPoint: optimumPixel, active, solveProgress: progress, levelRadius, contactRadius};
}
