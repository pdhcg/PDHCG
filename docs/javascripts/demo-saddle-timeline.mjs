// Expand the first primal and dual solves, then follow accepted primal-dual candidates.
// Reflections, anchors and restart working points remain solver details.
const clamp = value => Math.max(0, Math.min(1, value));
const ease = value => value * value * (3 - 2 * value);
const mix = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);

function event(r, scope, stage, part, title, caption, from, to, duration, extra = {}) {
  return {r, scope, mode: 'detail', iteration: r.iteration - 1, stage, part, title, caption,
    from, to, duration, completed: r.iteration - 1,
    localFrom: 0, localTo: 0, liftFrom: 0, liftTo: 0,
    blendFrom: 0, blendTo: 0, dualFrom: 0, dualTo: 0, ...extra};
}

export function buildIntroTimeline(p, r) {
  if (!r) return [];
  const dual = buildDualTimeline(p, r);
  Object.assign(dual[0], {transitionFrom: 'inner', localFrom: 1, liftFrom: 1, blendFrom: 1});
  return [
    event(r, 'outer', 0, 'primal', 'Primal update',
      'Solve the primal proximal subproblem with yₖ fixed.', [r.x, r.y], [r.x, r.y], 450),
    ...buildInnerTimeline(p, r).slice(0, -1),
    ...dual
  ];
}

export function buildTimeline(p, result) {
  const first = result.records[0];
  const events = buildIntroTimeline(p, first);
  events.scope = 'outer'; events.status = result.status;
  const referenceLength = first ? Math.max(1e-12,
    Math.hypot(first.xp - p.initial[0], first.yp - p.initial[1])) : 1;
  for (let i = 1; i < result.records.length; i++) {
    const r = result.records[i];
    const previous = result.records[i - 1];
    const distance = Math.hypot(r.xp - previous.xp, r.yp - previous.yp);
    const duration = Math.max(8, Math.min(160, 160 * Math.sqrt(distance / referenceLength)));
    const title = p.halpern ? 'PDHG + Halpern · primal–dual trajectory' : 'PDHG · primal–dual trajectory';
    events.push(event(r, 'outer', 2, 'trajectory', title,
      'Follow the primal–dual iterates toward a saddle point.',
      [previous.xp, previous.yp], [r.xp, r.yp], duration, {mode: 'trajectory'}));
  }
  return events;
}

function buildSubproblemTimeline(r, dual) {
  const scope = dual ? 'dual' : 'inner', stage = dual ? 1 : 0;
  const from = dual ? [r.xp, r.y] : [r.x, r.y];
  const to = dual ? [r.xp, r.yp] : [r.xp, r.y];
  const open = dual ? {dualTo: 1} : {localTo: 1, liftTo: 1, blendTo: 1};
  const hold = dual ? {dualFrom: 1, dualTo: 1} :
    {localFrom: 1, localTo: 1, liftFrom: 1, liftTo: 1, blendFrom: 1, blendTo: 1};
  const exit = dual ? {dualFrom: 1} : {localFrom: 1, liftFrom: 1, blendFrom: 1};
  const title = `${dual ? 'Dual' : 'Primal'} proximal subproblem`;
  const caption = dual ? 'Compute y⁺ using the extrapolated primal point.' :
    'Minimize the quadratic proximal objective over the primal domain.';
  const events = [
    event(r, scope, stage, 'enter', title, caption, from, from, dual ? 650 : 450, open),
    event(r, scope, stage, 'optimum', title, caption, from, to, 1800, hold),
    event(r, scope, stage, 'settle', title, caption, to, to, 800, hold),
    event(r, scope, stage, 'exit', `${dual ? 'Dual' : 'Primal'} update complete`,
      'Continue with the updated primal–dual pair.', to, to, 500, exit)
  ];
  events.scope = scope; events.status = 'solved';
  return events;
}

export const buildInnerTimeline = (p, r) => buildSubproblemTimeline(r, false);
export const buildDualTimeline = (p, r) => buildSubproblemTimeline(r, true);

export function sampleTimeline(events, cursor = 0) {
  if (!events.length) return {done: true, scope: events.scope ?? 'outer', cursor: 0,
    mode: ['inner', 'dual'].includes(events.scope) ? 'detail' : 'trajectory',
    point: null, motionPath: [], completed: 0, lift: 0, localOpacity: 0, innerBlend: 0, dualBlend: 0,
    eventProgress: 1, easedProgress: 1, solveProgress: 1,
    title: 'No iterations', caption: ''};
  const position = Math.max(0, Math.min(events.length, Number.isNaN(cursor) ? 0 : cursor));
  if (position === events.length) {
    const e = events.at(-1), inner = events.scope === 'inner', dual = events.scope === 'dual';
    const converged = events.status === 'converged';
    const point = inner ? [e.r.xp, e.r.y] : [e.r.xp, e.r.yp];
    return {...e, scope: events.scope ?? e.scope, done: true, mode: inner || dual ? 'detail' : 'trajectory',
      cursor: position, point, motionPath: [point],
      lift: 0, localOpacity: 0, innerBlend: 0, dualBlend: 0,
      eventProgress: 1, easedProgress: 1, solveProgress: 1,
      completed: e.iteration + (inner ? 0 : 1),
      title: inner ? 'Primal update complete' : dual ? 'Dual update complete' :
        converged ? 'Converged to a saddle point' : 'Run complete',
      caption: inner ? 'Use x⁺ in the dual update.' : dual ? 'Continue with the updated primal–dual pair.' : converged ?
        'The stopping tolerance is satisfied.' :
        events.status === 'iteration-limit' ? 'Iteration limit reached.' : 'The numerical run stopped.'};
  }
  const e = events[Math.floor(position)], progress = position - Math.floor(position);
  const linear = e.mode === 'trajectory';
  const moving = linear ? progress : clamp((progress - 0.12) / 0.76);
  const eased = linear ? progress : ease(moving);
  const point = mix(e.from, e.to, eased);
  return {...e, done: false, cursor: position, point,
    motionPath: [e.from, point],
    eventProgress: progress, easedProgress: eased,
    solveProgress: e.part === 'optimum' ? eased : ['settle', 'exit', 'trajectory'].includes(e.part) ? 1 : 0,
    lift: e.liftFrom + (e.liftTo - e.liftFrom) * eased,
    localOpacity: e.localFrom + (e.localTo - e.localFrom) * eased,
    innerBlend: e.blendFrom + (e.blendTo - e.blendFrom) * eased,
    dualBlend: (e.dualFrom ?? 0) + ((e.dualTo ?? 0) - (e.dualFrom ?? 0)) * eased};
}
