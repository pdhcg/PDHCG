// Expand the first primal and dual solves, then follow accepted primal-dual candidates.
// Reflections, anchors and restart working points remain solver details.
const clamp = value => Math.max(0, Math.min(1, value));
const ease = value => value * value * (3 - 2 * value);
const mix = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);

function event(r, scope, stage, part, title, caption, from, to, duration, extra = {}) {
  return {r, scope, mode: 'detail', iteration: r.iteration - 1, stage, part, title, caption,
    detail: title, from, to, duration, completed: r.iteration - 1,
    localFrom: 0, localTo: 0, liftFrom: 0, liftTo: 0,
    blendFrom: 0, blendTo: 0, dualFrom: 0, dualTo: 0, ...extra};
}

function phases(events) {
  let start = 0;
  while (start < events.length) {
    let end = start + 1;
    while (end < events.length && events[end].iteration === events[start].iteration &&
      events[end].scope === events[start].scope &&
      events[end].stage === events[start].stage) end++;
    const total = events.slice(start, end).reduce((sum, e) => sum + e.duration, 0);
    let offset = 0;
    for (let j = start; j < end; j++) {
      events[j].phaseOffset = offset; events[j].phaseDuration = total;
      offset += events[j].duration;
    }
    start = end;
  }
  return events;
}

export function buildTimeline(p, result) {
  const events = [];
  events.scope = 'outer'; events.status = result.status; events.tolerance = result.tolerance;
  const first = result.records[0];
  const referenceLength = first ? Math.max(1e-12,
    Math.hypot(first.xp - p.initial[0], first.yp - p.initial[1])) : 1;
  for (let i = 0; i < result.records.length; i++) {
    const r = result.records[i];
    if (i === 0) {
      events.push(event(r, 'outer', 0, 'primal', 'Primal · call the inner solver',
        'Pass xₖ and fixed yₖ into the primal subproblem.', [r.x, r.y], [r.x, r.y], 900));
      events.push(...buildInnerTimeline(p, r));
      events.push(event(r, 'outer', 1, 'dual', 'Dual · open the proximal update',
        'Use extrapolated x̄ to update y; x⁺ waits to be paired with the result.',
        [r.xp, r.y], [r.xp, r.y], 700));
      events.push(...buildDualTimeline(p, r));
    } else {
      const previous = result.records[i - 1];
      const distance = Math.hypot(r.xp - previous.xp, r.yp - previous.yp);
      const duration = Math.max(8, Math.min(160, 160 * Math.sqrt(distance / referenceLength)));
      const title = p.halpern ? 'PDHG + Halpern · primal–dual trajectory' : 'PDHG · primal–dual trajectory';
      events.push(event(r, 'outer', 2, 'trajectory', title,
        'Follow the accepted primal–dual candidates toward the saddle.',
        [previous.xp, previous.yp], [r.xp, r.yp], duration, {mode: 'trajectory'}));
    }
  }
  return phases(events);
}

export function buildInnerTimeline(p, r) {
  const events = [];
  events.scope = 'inner'; events.status = 'solved';
  const title = 'Inside primal · projected BB';
  const caption = 'Take a primal gradient step, project onto the box, and refine the subproblem solution.';
  events.push(event(r, 'inner', 0, 'enter', 'Outer Primal → inner solve',
    `Iteration ${r.iteration} supplies xₖ and fixed yₖ. The outer iteration waits for x⁺.`,
    [r.x, r.y], [r.x, r.y], 900,
    {localTo: 1, liftTo: 1, blendTo: 1, detail: 'The outer Primal step calls its inner solver'}));
  for (let j = 0; j < r.innerSteps.length; j++) {
    const s = r.innerSteps[j];
    const common = {localFrom: 1, localTo: 1, liftFrom: 1, liftTo: 1,
      blendFrom: 1, blendTo: 1, innerStep: j, innerStepIndex: j};
    events.push(event(r, 'inner', 0, 'gradient', title, caption,
      [s.from, r.y], [s.trial, r.y], 1400,
      {...common, detail: `BB step ${j + 1} · primal gradient update`}));
    events.push(event(r, 'inner', 0, 'project', title, caption,
      [s.trial, r.y], [s.to, r.y], s.projected ? 900 : 450,
      {...common, detail: s.projected ? 'Project onto the variable bounds' : 'The primal step is already inside the box'}));
  }
  events.push(event(r, 'inner', 0, 'settle', 'Inner output: x⁺',
    'This solves the waiting Primal step. y stays fixed.',
    [r.xp, r.y], [r.xp, r.y], 1400,
    {localFrom: 1, localTo: 1, liftFrom: 1, liftTo: 1, blendFrom: 1, blendTo: 1,
      detail: 'The inner solver returns only x⁺'}));
  events.push(event(r, 'inner', 0, 'exit', 'Return x⁺ to the outer iteration',
    'Return to this iteration in the primal–dual trajectory.',
    [r.xp, r.y], [r.xp, r.y], 1300,
    {localFrom: 1, liftFrom: 1, blendFrom: 1, detail: 'Pass x⁺ to the waiting outer Dual step'}));
  return phases(events);
}

export function buildDualTimeline(p, r) {
  const events = [];
  events.scope = 'dual'; events.status = 'solved';
  const projected = Math.abs(r.trialY - r.yp) > 1e-12;
  const common = {dualFrom: 1, dualTo: 1};
  events.push(event(r, 'dual', 1, 'enter', 'Outer Dual → proximal update',
    `Iteration ${r.iteration} supplies yₖ and extrapolated x̄. x⁺ waits for y⁺.`,
    [r.xp, r.y], [r.xp, r.y], 900,
    {dualTo: 1, detail: 'Open the dual proximal objective at x̄'}));
  events.push(event(r, 'dual', 1, 'ascent', 'Dual · take an ascent step',
    'The dual ascent step gives yₖ + σ(x̄ − b), followed by projection onto the dual domain.',
    [r.xp, r.y], [r.xp, r.trialY], 1600,
    {...common, detail: 'Find the unconstrained maximum of the dual proximal objective'}));
  events.push(event(r, 'dual', 1, 'project', 'Dual · enforce the dual domain',
    p.constraint === 'equality' ? 'The equality multiplier is free: no projection is needed.' :
      projected ? 'Project the dual step onto y ≥ 0.' : 'The dual step is already in y ≥ 0.',
    [r.xp, r.trialY], [r.xp, r.yp], projected ? 700 : 350,
    {...common, detail: projected ? 'Project onto the nonnegative dual cone' : 'The dual projection leaves this step unchanged'}));
  events.push(event(r, 'dual', 1, 'settle', 'Dual output: y⁺',
    'Pair y⁺ with the solved x⁺ to complete this primal–dual update.',
    [r.xp, r.yp], [r.xp, r.yp], 800,
    {...common, detail: 'The dual update returns y⁺'}));
  events.push(event(r, 'dual', 1, 'exit', 'Return (x⁺, y⁺) to the trajectory',
    'The completed pair is ready for the next outer iteration.',
    [r.xp, r.yp], [r.xp, r.yp], 1000,
    {dualFrom: 1, detail: 'Return the completed primal–dual candidate'}));
  return phases(events);
}

export function sampleTimeline(events, cursor = 0) {
  if (!events.length) return {done: true, scope: events.scope ?? 'outer', cursor: 0,
    mode: ['inner', 'dual'].includes(events.scope) ? 'detail' : 'trajectory',
    point: null, motionPath: [], completed: 0, lift: 0, localOpacity: 0, innerBlend: 0, dualBlend: 0,
    halpernPhase: null, substepProgress: 1, anchorWeight: 0,
    eventProgress: 1, easedProgress: 1, phaseProgress: 1, title: 'No iterations', caption: ''};
  const position = Math.max(0, Math.min(events.length, Number.isNaN(cursor) ? 0 : cursor));
  if (position === events.length) {
    const e = events.at(-1), inner = events.scope === 'inner', dual = events.scope === 'dual';
    const converged = events.status === 'converged';
    const point = inner ? [e.r.xp, e.r.y] : [e.r.xp, e.r.yp];
    return {...e, scope: events.scope ?? e.scope, done: true, mode: inner || dual ? 'detail' : 'trajectory',
      cursor: position, point, motionPath: [point],
      lift: 0, localOpacity: 0, innerBlend: 0, dualBlend: 0, eventProgress: 1, easedProgress: 1, phaseProgress: 1,
      halpernPhase: null, substepProgress: 1, anchorWeight: 1 - e.r.theta,
      completed: e.iteration + (inner ? 0 : 1),
      title: inner ? 'x⁺ returned to the outer iteration' : dual ? '(x⁺, y⁺) returned to the trajectory' :
        converged ? 'At the saddle point' : 'Run complete',
      caption: inner || dual ? 'Return to the outer primal–dual trajectory.' : converged ?
        'Primal feasibility and dual stationarity meet at this point.' :
        events.status === 'iteration-limit' ? 'Iteration limit reached.' : 'The numerical run stopped.'};
  }
  const e = events[Math.floor(position)], progress = position - Math.floor(position);
  const linear = e.mode === 'trajectory';
  const moving = linear ? progress : clamp((progress - 0.12) / 0.76);
  const eased = linear ? progress : ease(moving);
  const waypoints = e.waypoints ?? [e.from, e.to];
  const segmentPosition = moving * (waypoints.length - 1);
  const segment = Math.min(waypoints.length - 2, Math.floor(segmentPosition));
  const fraction = linear ? clamp(segmentPosition - segment) : ease(clamp(segmentPosition - segment));
  const point = mix(waypoints[segment], waypoints[segment + 1], fraction);
  return {...e, done: false, cursor: position, point,
    motionPath: [...waypoints.slice(0, segment + 1), point],
    lift: e.liftFrom + (e.liftTo - e.liftFrom) * eased,
    localOpacity: e.localFrom + (e.localTo - e.localFrom) * eased,
    innerBlend: e.blendFrom + (e.blendTo - e.blendFrom) * eased,
    dualBlend: (e.dualFrom ?? 0) + ((e.dualTo ?? 0) - (e.dualFrom ?? 0)) * eased,
    halpernPhase: null, substepProgress: fraction, anchorWeight: 1 - e.r.theta,
    eventProgress: progress, easedProgress: eased,
    phaseProgress: (e.phaseOffset + e.duration * progress) / e.phaseDuration};
}
