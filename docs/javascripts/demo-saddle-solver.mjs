// A scalar conic QP, represented by the same low-rank quadratic and row-bound
// operations as native PDHCG. This fixed-step CPU reference supports plain
// PDHG and reflected Halpern with a ten-iteration restart check. Its BB initial
// step is configurable; native scaling, defaults and stopping can change a run.

export function makeSaddleProblem(options = {}) {
  const {q = 0.05, a = 3, b = 3, tau = 0.2, sigma = 0.2,
    lower = 0, upper = 6, constraint = 'equality', halpern = false,
    innerInitialScale = 0.5} = options;
  const initial = [...(options.initial ?? [2, 2])];
  if (![q, a, b, tau, sigma, lower, upper, innerInitialScale].every(Number.isFinite) || q <= 0 ||
      tau <= 0 || sigma <= 0 || tau * sigma >= 1 || initial.length !== 2 ||
      lower >= upper || b < lower || (constraint === 'equality' && b > upper) ||
      !['equality', 'inequality'].includes(constraint) || typeof halpern !== 'boolean' ||
      innerInitialScale <= 0 || !initial.every(Number.isFinite))
    throw new Error('Invalid saddle demo problem');
  return {q, a, b, tau, sigma, lower, upper, initial, constraint, halpern, innerInitialScale,
    restarts: options.restarts !== false,
    eta: Math.sqrt(tau * sigma), weight: Math.sqrt(sigma / tau),
    saddle: constraint === 'equality' ? [b, q * (a - b)] :
      [Math.max(lower, Math.min(a, b, upper)), b <= upper ? Math.max(0, q * (a - b)) : 0]};
}

export const projectBox = (p, x) => Math.max(p.lower, Math.min(p.upper, x));

export const objective = (p, x) => 0.5 * p.q * (x - p.a) ** 2;
export const lagrangian = (p, x, y) => objective(p, x) + y * (x - p.b);

// The proximal curve meets the original Lagrangian slice at the working point.
export const localPrimal = (p, r, u) =>
  lagrangian(p, u, r.y) + (u - r.x) ** 2 / (2 * p.tau);

// The dual proximal objective is maximized at trialY, then restricted to the
// dual domain. Its linear term uses extrapolated xbar, not the displayed xp.
export const localDual = (p, r, v) =>
  lagrangian(p, r.xbar, v) - (v - r.y) ** 2 / (2 * p.sigma);

export function saddleMetrics(p, x, y) {
  const primalObjective = objective(p, x);
  const dualMinimizer = projectBox(p, p.a - y / p.q);
  const dualObjective = lagrangian(p, dualMinimizer, y);
  const rowViolation = p.constraint === 'equality' ? Math.abs(x - p.b) : Math.max(0, x - p.b);
  const primal = Math.max(rowViolation, p.lower - x, x - p.upper) / (1 + Math.abs(p.b));
  const dualCone = p.constraint === 'equality' ? 0 : Math.max(0, -y) / (1 + Math.abs(y));
  const stationarity = Math.abs(x - projectBox(p, x - (p.q * (x - p.a) + y))) /
    (1 + Math.abs(p.q * p.a));
  const complementarity = p.constraint === 'equality' ? 0 : Math.abs(y * (x - p.b)) /
    (1 + Math.abs(y) * (1 + Math.abs(x) + Math.abs(p.b)));
  const gap = Math.abs(primalObjective - dualObjective) /
    (1 + Math.abs(primalObjective) + Math.abs(dualObjective));
  return {objective: primalObjective, dualObjective, primal, dualCone,
    stationarity, complementarity, gap,
    error: Math.max(primal, dualCone, stationarity, complementarity, gap)};
}

function innerBB(p, x, y) {
  const gradient = u => p.q * (u - p.a) + y + (u - x) / p.tau;
  const curvature = p.q + 1 / p.tau;
  const path = [x], steps = [];
  let u = x, g = gradient(x), alpha = p.innerInitialScale * p.tau;
  // In one dimension BB1 learns the sole curvature after its first step, so
  // its second step solves this quadratic. Retain the real computed iterates.
  for (let j = 0; j < 3 && Math.abs(u - projectBox(p, u - g)) > 1e-12; j++) {
    const trial = u - alpha * g, next = projectBox(p, trial), nextG = gradient(next);
    const displacement = next - u, change = nextG - g;
    path.push(next);
    steps.push({from: u, trial, to: next, alpha, gradient: g, projected: Math.abs(trial - next) > 1e-12});
    const quotient = displacement / change;
    alpha = Number.isFinite(quotient) && quotient > 0 ? quotient : 1 / curvature;
    u = next; g = nextG;
  }
  return {path, steps};
}

export function solveSaddle(p, {limit = 1000, tolerance = 1e-6} = {}) {
  if (!Number.isInteger(limit) || limit < 1 || !Number.isFinite(tolerance) || tolerance <= 0)
    throw new Error('Invalid saddle solve limits');
  let [x, y] = p.initial, anchor = [x, y], cycle = 0;
  let fixedError = Infinity, initialError = Infinity, previousTrial = Infinity;
  const records = [];
  for (let k = 0; k < limit; k++) {
    let restart = false;
    if (p.halpern && p.restarts && k > 0 && k % 10 === 0) {
      restart = k === 10 || fixedError <= 0.2 * initialError ||
        (fixedError <= 0.8 * initialError && fixedError > previousTrial) || cycle >= 0.36 * k;
      previousTrial = fixedError;
      if (restart) {
        x = records[k - 1].xp; y = records[k - 1].yp;
        anchor = [x, y]; cycle = 0; previousTrial = Infinity;
      }
    }
    const inner = innerBB(p, x, y), primalPath = inner.path, xp = primalPath.at(-1);
    const xbar = 2 * xp - x;
    // A=-1: an equality row has a free multiplier; a lower-bound row has y>=0.
    const trialY = y + p.sigma * (xbar - p.b);
    const yp = p.constraint === 'equality' ? trialY : Math.max(0, trialY);
    const reflection = [xbar, 2 * yp - y];
    const theta = p.halpern ? (cycle + 1) / (cycle + 2) : 1;
    const next = p.halpern ? reflection.map((v, i) => (1 - theta) * anchor[i] + theta * v) : [xp, yp];
    const dx = reflection[0] - x, dy = reflection[1] - y;
    fixedError = Math.sqrt(Math.max(0, p.weight * dx * dx + dy * dy / p.weight -
      2 * p.eta * dx * dy));
    if (k === 0 || restart) initialError = fixedError;
    const metrics = saddleMetrics(p, xp, yp);
    records.push({iteration: k + 1, cycle, x, y, anchor: [...anchor], primalPath, innerSteps: inner.steps,
      xp, yp, xbar, trialY, reflection, next, theta, restart, fixedError, metrics});
    if (![...next, xp, yp, fixedError, metrics.error].every(Number.isFinite))
      return {records, status: 'numerical-error', tolerance};
    if (metrics.error <= tolerance) return {records, status: 'converged', tolerance};
    [x, y] = next; cycle++;
  }
  return {records, status: 'iteration-limit', tolerance};
}
