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

// Genuine two-dimensional conic QP:
//   min q/2 ||x-target||², ||x||<=radius,
//   [coneRadius, x-offset] in the three-dimensional second-order cone.
// The browser uses exact disk/SOC proximal steps and fixed step sizes.
const norm = v => Math.hypot(...v);
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const add = (a, b) => a.map((value, i) => value + b[i]);
const subtract = (a, b) => a.map((value, i) => value - b[i]);
const scale = (v, factor) => v.map(value => value * factor);
const display = (x, y) => [x[0], -y[1]];

export function projectDisk(v, radius) {
  return scale(v, Math.min(1, radius / (norm(v) || 1)));
}

// Euclidean projection onto Q3 = {(t,z): ||z|| <= t}.
// Parikh & Boyd, Proximal Algorithms, section 6.3.2:
// https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
export function projectSOC(v) {
  const length = norm(v.slice(1)), t = v[0];
  if (length <= t) return [...v];
  if (length <= -t) return [0, 0, 0];
  const height = (length + t) / 2;
  return [height, ...scale(v.slice(1), height / length)];
}

export function makeConicProblem(options = {}) {
  const {q = .05, radius = 6, coneRadius = 3, tau = .22, sigma = .22,
    halpern = false} = options;
  const target = [...(options.target ?? [60, 3])];
  const offset = [...(options.offset ?? [0, -.4])];
  const initialX = [...(options.initialX ?? [Math.sqrt(35), 1])];
  const initialY = [...(options.initialY ?? [Math.sqrt(.05), -.2, -.1])];
  const validVector = (v, length) => v.length === length && v.every(Number.isFinite);
  if (![q, radius, coneRadius, tau, sigma].every(Number.isFinite) || q <= 0 ||
      radius <= 0 || coneRadius <= 0 || tau <= 0 || sigma <= 0 || tau * sigma >= 1 ||
      typeof halpern !== 'boolean' || !validVector(target, 2) || !validVector(offset, 2) ||
      !validVector(initialX, 2) || !validVector(initialY, 3))
    throw new Error('Invalid conic demo problem');
  const displacement = subtract(target, offset), length = norm(displacement);
  const direction = length ? scale(displacement, 1 / length) : [1, 0];
  const saddleX = length > coneRadius ? add(offset, scale(direction, coneRadius)) : [...target];
  const multiplier = q * Math.max(0, length - coneRadius);
  const saddleY = [multiplier, ...scale(direction, -multiplier)];
  // This example family keeps the known solution inside the primal disk;
  // a second active disk would require a different analytical KKT reference.
  if (norm(saddleX) >= radius || Math.abs(direction[0]) < 1e-8)
    throw new Error('The conic demo reference requires an interior disk solution and a nonvertical display direction');
  return {kind: 'conic', constraint: 'conic', q, radius, coneRadius, tau, sigma, target, offset,
    initialX, initialY, halpern, restarts: options.restarts !== false,
    eta: Math.sqrt(tau * sigma), weight: Math.sqrt(sigma / tau),
    saddleX, saddleY, direction, initial: display(initialX, initialY), saddle: display(saddleX, saddleY),
    displayLabels: ['primal x₁', 'dual −y₁']};
}

export const conicObjective = (p, x) => .5 * p.q * dot(subtract(x, p.target), subtract(x, p.target));
const residual = (p, x) => [p.coneRadius, ...subtract(x, p.offset)];
export const conicLagrangian = (p, x, y) => conicObjective(p, x) - dot(y, residual(p, x));

// A fixed, exact Lagrangian slice through the true saddle. The outer path is
// the projection (x1,-y1) of actual vector iterates; its height is a reference
// slice value, not the full five-dimensional iterate's Lagrangian value.
export function conicDisplayLagrangian(p, x, y) {
  const ratio = y / p.direction[0];
  return conicLagrangian(p, [x, p.saddleX[1]],
    [ratio, -y, -ratio * p.direction[1]]);
}

export function conicMetrics(p, x, y) {
  const coneResidual = residual(p, x), primalObjective = conicObjective(p, x);
  const dualMinimizer = projectDisk(add(p.target, scale(y.slice(1), 1 / p.q)), p.radius);
  const dualObjective = conicLagrangian(p, dualMinimizer, y);
  const primal = Math.max(0, norm(x) - p.radius, norm(subtract(x, p.offset)) - p.coneRadius) /
    (1 + norm(p.offset) + p.coneRadius);
  const dualCone = norm(subtract(y, projectSOC(y))) / (1 + norm(y));
  const gradient = subtract(scale(subtract(x, p.target), p.q), y.slice(1));
  const stationarity = norm(subtract(x, projectDisk(subtract(x, gradient), p.radius))) /
    (1 + p.q * norm(p.target));
  const complementarity = Math.abs(dot(y, coneResidual)) /
    (1 + norm(y) * (1 + norm(coneResidual)));
  const gap = Math.abs(primalObjective - dualObjective) /
    (1 + Math.abs(primalObjective) + Math.abs(dualObjective));
  return {objective: primalObjective, dualObjective, primal, dualCone, stationarity, complementarity, gap,
    error: Math.max(primal, dualCone, stationarity, complementarity, gap)};
}

export function solveConic(p, {limit = 1500, tolerance = 1e-6} = {}) {
  if (!Number.isInteger(limit) || limit < 1 || !Number.isFinite(tolerance) || tolerance <= 0)
    throw new Error('Invalid conic solve limits');
  let vectorX = [...p.initialX], vectorY = [...p.initialY];
  let vectorAnchor = [...vectorX, ...vectorY], cycle = 0;
  let fixedError = Infinity, initialError = Infinity, previousTrial = Infinity;
  const records = [];
  for (let k = 0; k < limit; k++) {
    let restart = false;
    if (p.halpern && p.restarts && k > 0 && k % 10 === 0) {
      restart = k === 10 || fixedError <= .2 * initialError ||
        (fixedError <= .8 * initialError && fixedError > previousTrial) || cycle >= .36 * k;
      previousTrial = fixedError;
      if (restart) {
        vectorX = [...records[k - 1].vectorXp]; vectorY = [...records[k - 1].vectorYp];
        vectorAnchor = [...vectorX, ...vectorY]; cycle = 0; previousTrial = Infinity;
      }
    }
    const primalCenter = scale(add(vectorX, scale(add(scale(p.target, p.q), vectorY.slice(1)), p.tau)),
      1 / (1 + p.tau * p.q));
    const vectorXp = projectDisk(primalCenter, p.radius);
    const vectorXbar = subtract(scale(vectorXp, 2), vectorX);
    const dualCenter = subtract(vectorY, scale(residual(p, vectorXbar), p.sigma));
    const vectorYp = projectSOC(dualCenter);
    const current = [...vectorX, ...vectorY], candidate = [...vectorXp, ...vectorYp];
    const vectorReflection = subtract(scale(candidate, 2), current);
    const theta = p.halpern ? (cycle + 1) / (cycle + 2) : 1;
    const vectorNext = p.halpern ? add(scale(vectorAnchor, 1 - theta), scale(vectorReflection, theta)) : candidate;
    const movement = subtract(vectorReflection, current), dx = movement.slice(0, 2), dy = movement.slice(2);
    // The Lagrangian contains -<y_vec,x>; hence the positive cross term.
    fixedError = Math.sqrt(Math.max(0, p.weight * dot(dx, dx) + dot(dy, dy) / p.weight +
      2 * p.eta * dot(dx, dy.slice(1))));
    if (k === 0 || restart) initialError = fixedError;
    const metrics = conicMetrics(p, vectorXp, vectorYp);
    const [x, y] = display(vectorX, vectorY), [xp, yp] = display(vectorXp, vectorYp);
    records.push({iteration: k + 1, cycle, vectorX: [...vectorX], vectorY: [...vectorY],
      vectorXp, vectorYp, vectorXbar, primalCenter, dualCenter,
      vectorAnchor: [...vectorAnchor], vectorReflection, vectorNext,
      x, y, xp, yp, xbar: vectorXbar[0], trialY: -dualCenter[1],
      anchor: display(vectorAnchor.slice(0, 2), vectorAnchor.slice(2)),
      reflection: display(vectorReflection.slice(0, 2), vectorReflection.slice(2)),
      next: display(vectorNext.slice(0, 2), vectorNext.slice(2)),
      primalPath: [x, xp], innerSteps: [], theta, restart, fixedError, metrics,
      primalProjected: norm(subtract(primalCenter, vectorXp)) > 1e-12,
      dualProjected: norm(subtract(dualCenter, vectorYp)) > 1e-12});
    if (![...vectorNext, ...candidate, fixedError, metrics.error].every(Number.isFinite))
      return {records, status: 'numerical-error', tolerance};
    if (metrics.error <= tolerance) return {records, status: 'converged', tolerance};
    vectorX = vectorNext.slice(0, 2); vectorY = vectorNext.slice(2); cycle++;
  }
  return {records, status: 'iteration-limit', tolerance};
}
