import {makeSaddleProblem, solveSaddle} from './demo-saddle-solver.mjs';
import {buildIntroTimeline, sampleTimeline} from './demo-saddle-timeline.mjs';

// Both methods solve the same problem from the same starting point. A shared
// cursor counts outer updates, never a separately normalized animation time.
export function createComparison(options = {}) {
  const baselineProblem = makeSaddleProblem({...options, halpern: false});
  const halpernProblem = makeSaddleProblem({...options, halpern: true});
  const baseline = {problem: baselineProblem, result: solveSaddle(baselineProblem)};
  const halpern = {problem: halpernProblem, result: solveSaddle(halpernProblem)};
  return {baseline, halpern,
    iterations: Math.max(baseline.result.records.length, halpern.result.records.length)};
}

const candidate = r => [r.xp, r.yp];

function sampleRun(run, cursor) {
  const {problem, result} = run;
  const count = result.records.length;
  const position = Math.min(cursor, count);
  const done = position === count;
  const completed = Math.floor(position);
  const r = result.records[Math.min(completed, count - 1)];
  const from = completed === 0 ? [...problem.initial] : candidate(result.records[completed - 1]);
  const to = done ? [...from] : candidate(r);
  const progress = done ? 1 : position - completed;
  const point = from.map((value, i) => value + progress * (to[i] - value));
  return {problem, result, r, cursor, iteration: r.iteration - 1, completed, done,
    status: result.status, converged: done && result.status === 'converged',
    scope: 'outer', mode: 'trajectory', stage: 2, part: 'trajectory',
    title: problem.halpern ? 'PDHG + Halpern · primal–dual trajectory' : 'PDHG · primal–dual trajectory',
    caption: 'Both methods advance at the same outer iteration count.',
    from, to, point, motionPath: done ? [point] : [from, point],
    lift: 0, localOpacity: 0, innerBlend: 0, dualBlend: 0};
}

// Integer k is the kth computed primal-dual candidate (xp, yp). Fractional
// positions linearly connect those candidates; Halpern's auxiliary reflection
// and next working point are intentionally not presented as candidate points.
// A finished method holds its actual final candidate while the other proceeds.
export function sampleComparison(comparison, cursor = 0) {
  const position = Math.max(0, Math.min(comparison.iterations, Number.isNaN(cursor) ? 0 : cursor));
  return {...sampleRun(comparison.baseline, position),
    comparison: sampleRun(comparison.halpern, position)};
}

// Explain the common first update before separating the two computed paths.
// With theta=1/2, the first reflected Halpern update is the same candidate as
// plain PDHG; the first iteration can therefore be shown once for both runs.
export function buildComparisonStory(comparison) {
  const events = buildIntroTimeline(comparison.baseline.problem, comparison.baseline.result.records[0]);
  events.introLength = events.length;
  events.scope = 'outer';
  events.status = comparison.baseline.result.status;
  for (let k = 1; k < comparison.iterations; k++) {
    const duration = 1000 / Math.min(45, 5 + 0.16 * k);
    events.push({mode: 'trajectory', duration});
  }
  return events;
}

export function sampleComparisonStory(comparison, events, cursor = 0) {
  const position = Math.max(0, Math.min(events.length, Number.isNaN(cursor) ? 0 : cursor));
  const common = {cursor: position, storyDone: position === events.length, showFirstUpdate: true,
    comparisonBounds: comparison.halpern.result};
  if (position < events.introLength) {
    const model = sampleTimeline(events, position);
    const returnedDual = model.scope === 'dual' && ['settle', 'exit'].includes(model.part);
    return {...model, ...common, storyPhase: 'intro', compareCursor: returnedDual ? 1 : 0};
  }
  const compareCursor = Math.min(comparison.iterations, 1 + position - events.introLength);
  return {...sampleComparison(comparison, compareCursor), ...common,
    storyPhase: 'trajectory', compareCursor};
}
