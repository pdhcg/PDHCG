import test from 'node:test';
import assert from 'node:assert/strict';
import {createComparison, sampleComparison, buildComparisonStory, sampleComparisonStory} from '../docs/javascripts/demo-saddle-comparison.mjs';
import {makeSaddleProblem, solveSaddle} from '../docs/javascripts/demo-saddle-solver.mjs';

const near = (a,b,tolerance=1e-10) => assert.ok(Math.abs(a-b)<=tolerance,`${a} differs from ${b}`);
const nearPoint = (a,b,tolerance=1e-10) => a.forEach((v,i)=>near(v,b[i],tolerance));
const candidate = r => [r.xp,r.yp];
const presets = [{},{tau:.3,sigma:.04/.3},
  {q:1,a:1.8,b:.5,tau:.6,sigma:.8,lower:-1.2,upper:.54,initial:[-.8,0],constraint:'inequality'}];

test('comparison solves the same three problems from the same initial state with only Halpern changed',()=>{
  for(const options of presets){
    const before=JSON.stringify(options),comparison=createComparison(options);
    const {baseline,halpern}=comparison;
    assert.equal(baseline.problem.halpern,false);assert.equal(halpern.problem.halpern,true);
    assert.deepEqual({...baseline.problem,halpern:true},halpern.problem);
    assert.notEqual(baseline.problem.initial,halpern.problem.initial);
    for(const run of [baseline,halpern]){
      assert.equal(run.result.status,'converged');
      assert.deepEqual(run.result,solveSaddle(makeSaddleProblem({...options,halpern:run.problem.halpern})));
      const last=run.result.records.at(-1);
      assert.ok(last.metrics.error<=run.result.tolerance);
      nearPoint(candidate(last),run.problem.saddle,1e-5);
    }
    assert.equal(comparison.iterations,Math.max(baseline.result.records.length,halpern.result.records.length));
    assert.equal(JSON.stringify(options),before);
  }
  const forced=createComparison({halpern:true});
  assert.equal(forced.baseline.problem.halpern,false);
  assert.equal(forced.halpern.problem.halpern,true);
});

test('shared integer cursor k samples each actual kth candidate and gives both curves the same iteration clock',()=>{
  for(const options of presets){
    const comparison=createComparison(options);
    for(let k=0;k<=comparison.iterations;k++){
      const main=sampleComparison(comparison,k);
      for(const [run,state] of [[comparison.baseline,main],[comparison.halpern,main.comparison]]){
        const completed=Math.min(k,run.result.records.length);
        nearPoint(state.point,completed?candidate(run.result.records[completed-1]):run.problem.initial);
        assert.equal(state.completed,completed);
        assert.equal(state.cursor,k);
        assert.equal(state.done,k>=run.result.records.length);
        assert.equal(state.problem,run.problem);assert.equal(state.result,run.result);
        assert.equal(state.scope,'outer');assert.equal(state.mode,'trajectory');
        assert.equal(state.stage,2);assert.equal(state.part,'trajectory');
        near(state.innerBlend,0);near(state.dualBlend,0);near(state.lift,0);
      }
    }
  }
});

test('fractional progress linearly connects genuine adjacent candidates without inventing a reflection path',()=>{
  const comparison=createComparison(),before=JSON.stringify(comparison);
  for(const t of [.17,.5,.83])for(let k=0;k<comparison.iterations;k++){
    const model=sampleComparison(comparison,k+t);
    for(const [run,state] of [[comparison.baseline,model],[comparison.halpern,model.comparison]]){
      const n=run.result.records.length;
      if(k>=n){nearPoint(state.point,candidate(run.result.records.at(-1)));continue;}
      const from=k?candidate(run.result.records[k-1]):run.problem.initial;
      const to=candidate(run.result.records[k]);
      nearPoint(state.from,from);nearPoint(state.to,to);
      nearPoint(state.point,from.map((v,i)=>v+t*(to[i]-v)));
      assert.equal(state.motionPath.length,2);
      nearPoint(state.motionPath[0],from);nearPoint(state.motionPath[1],state.point);
    }
  }
  const records=comparison.halpern.result.records;
  const distinct=records.findIndex(r=>Math.hypot(r.next[0]-r.xp,r.next[1]-r.yp)>.05);
  assert.ok(distinct>0);
  const sampled=sampleComparison(comparison,distinct+1).comparison;
  nearPoint(sampled.point,candidate(records[distinct]));
  assert.ok(Math.hypot(sampled.point[0]-records[distinct].next[0],sampled.point[1]-records[distinct].next[1])>.05);
  assert.equal(JSON.stringify(comparison),before,'Sampling must not mutate either computed run');
});

test('whichever method converges first holds its actual endpoint while the other keeps moving',()=>{
  let baselineFirst=false,halpernFirst=false;
  for(const options of presets){
    const comparison=createComparison(options);
    const nb=comparison.baseline.result.records.length,nh=comparison.halpern.result.records.length;
    const first=Math.min(nb,nh),last=Math.max(nb,nh);
    assert.notEqual(first,last);
    const a=sampleComparison(comparison,first+.25),b=sampleComparison(comparison,first+.75);
    const stoppedA=nb<nh?a:a.comparison,stoppedB=nb<nh?b:b.comparison;
    const movingA=nb<nh?a.comparison:a,movingB=nb<nh?b.comparison:b;
    assert.equal(stoppedA.done,true);assert.equal(stoppedB.done,true);
    assert.equal(movingA.done,false);assert.equal(movingB.done,false);
    nearPoint(stoppedA.point,stoppedB.point);nearPoint(stoppedA.point,candidate(stoppedA.result.records.at(-1)));
    assert.ok(Math.hypot(movingA.point[0]-movingB.point[0],movingA.point[1]-movingB.point[1])>0);
    assert.equal(stoppedA.completed,first);assert.equal(movingA.completed,first);
    near(a.cursor,a.comparison.cursor);near(b.cursor,b.comparison.cursor);
    baselineFirst ||= nb<nh;halpernFirst ||= nh<nb;
  }
  assert.ok(baselineFirst&&halpernFirst,'Cover both orders of convergence across the actual presets');
});

test('initial, final and out-of-range cursors remain well-defined scene models',()=>{
  const comparison=createComparison();
  for(const cursor of [undefined,0,-5,-Infinity,NaN]){
    const state=sampleComparison(comparison,cursor);
    nearPoint(state.point,comparison.baseline.problem.initial);
    nearPoint(state.comparison.point,comparison.halpern.problem.initial);
    assert.equal(state.completed,0);assert.equal(state.comparison.completed,0);
    assert.equal(state.cursor,0);assert.equal(state.done,false);assert.equal(state.comparison.done,false);
  }
  for(const cursor of [comparison.iterations,comparison.iterations+10,Infinity]){
    const state=sampleComparison(comparison,cursor);
    for(const model of [state,state.comparison]){
      assert.equal(model.done,true);assert.equal(model.converged,true);
      nearPoint(model.point,candidate(model.result.records.at(-1)));
      nearPoint(model.from,model.to);nearPoint(model.motionPath[0],model.point);
      assert.equal(model.completed,model.result.records.length);
      assert.equal(model.cursor,comparison.iterations);
    }
  }
});

test('the shared story starts with the complete common primal and dual update before comparing trajectories',()=>{
  for(const options of presets){
    const comparison=createComparison(options),events=buildComparisonStory(comparison);
    assert.equal(events.introLength,14);
    assert.deepEqual(events.slice(0,events.introLength).map(e=>`${e.scope}:${e.part}`),[
      'outer:primal','inner:enter','inner:gradient','inner:project','inner:gradient','inner:project',
      'inner:settle','inner:exit','outer:dual','dual:enter','dual:ascent','dual:project','dual:settle','dual:exit']);
    assert.equal(events.length,events.introLength+comparison.iterations-1);
    nearPoint(candidate(comparison.baseline.result.records[0]),candidate(comparison.halpern.result.records[0]));
    for(let i=0;i<events.introLength;i++){
      const model=sampleComparisonStory(comparison,events,i+.5);
      assert.equal(model.storyPhase,'intro');assert.equal(model.comparison,undefined);
      assert.equal(model.storyDone,false);assert.equal(model.scope,events[i].scope);
      assert.equal(model.comparisonBounds,comparison.halpern.result);
      assert.equal(model.showFirstUpdate,true);
      assert.equal(model.cursor,i+.5);
      assert.equal(model.r,comparison.baseline.result.records[0]);
    }
    assert.ok(events.reduce((total,e)=>total+e.duration,0)<40000);
  }
});

test('both paths begin continuously at the completed first candidate when the close-up returns',()=>{
  for(const options of presets){
    const comparison=createComparison(options),events=buildComparisonStory(comparison);
    for(let i=1;i<events.length;i++){
      const previous=sampleComparisonStory(comparison,events,i-1e-8);
      const next=sampleComparisonStory(comparison,events,i);
      nearPoint(previous.point,next.point,1e-7);
    }
    const before=sampleComparisonStory(comparison,events,events.introLength-1e-5);
    const after=sampleComparisonStory(comparison,events,events.introLength);
    nearPoint(before.point,after.point);nearPoint(after.point,after.comparison.point);
    nearPoint(after.point,candidate(comparison.baseline.result.records[0]));
    near(before.innerBlend,0);near(before.dualBlend,0);near(after.innerBlend,0);near(after.dualBlend,0);
    assert.equal(before.comparisonBounds,after.comparisonBounds);
    assert.equal(before.showFirstUpdate,true);assert.equal(after.showFirstUpdate,true);
    assert.equal(before.compareCursor,1);assert.equal(after.compareCursor,1);
    assert.equal(after.storyPhase,'trajectory');assert.equal(after.scope,'outer');
  }
});

test('story trajectory events use one shared iteration clock and preserve every real candidate segment',()=>{
  for(const options of presets){
    const comparison=createComparison(options),before=JSON.stringify(comparison),events=buildComparisonStory(comparison);
    for(let k=1;k<comparison.iterations;k++){
      const i=events.introLength+k-1;
      near(events[i].duration,1000/Math.min(45,5+.16*k));
      for(const fraction of [0,.37,.91]){
        const story=sampleComparisonStory(comparison,events,i+fraction);
        const expected=sampleComparison(comparison,k+fraction);
        near(story.compareCursor,k+fraction);near(story.cursor,i+fraction);
        nearPoint(story.point,expected.point);nearPoint(story.comparison.point,expected.comparison.point);
        assert.equal(story.completed,expected.completed);assert.equal(story.done,expected.done);
        assert.equal(story.comparison.completed,expected.comparison.completed);
        assert.equal(story.comparison.done,expected.comparison.done);
        assert.equal(story.storyDone,false);assert.equal(story.comparisonBounds,comparison.halpern.result);
      }
    }
    assert.equal(JSON.stringify(comparison),before);
  }
});

test('story completion waits for both runs and keeps a stable initial and final state',()=>{
  for(const options of [...presets,{initial:[3,0]}]){
    const comparison=createComparison(options),events=buildComparisonStory(comparison);
    for(const cursor of [0,-1,NaN]){
      const model=sampleComparisonStory(comparison,events,cursor);
      nearPoint(model.point,comparison.baseline.problem.initial);
      assert.equal(model.storyPhase,'intro');assert.equal(model.compareCursor,0);assert.equal(model.cursor,0);
    }
    for(const cursor of [events.length,events.length+1,Infinity]){
      const model=sampleComparisonStory(comparison,events,cursor);
      assert.equal(model.storyDone,true);assert.equal(model.done,true);assert.equal(model.comparison.done,true);
      assert.equal(model.storyPhase,'trajectory');assert.equal(model.compareCursor,comparison.iterations);
      nearPoint(model.point,candidate(comparison.baseline.result.records.at(-1)));
      nearPoint(model.comparison.point,candidate(comparison.halpern.result.records.at(-1)));
      assert.equal(model.cursor,events.length);assert.equal(model.comparisonBounds,comparison.halpern.result);
      assert.equal(model.showFirstUpdate,true);
    }
  }
  const comparison=createComparison(presets[2]),events=buildComparisonStory(comparison);
  const early=sampleComparisonStory(comparison,events,events.introLength+comparison.baseline.result.records.length-.5);
  assert.equal(early.done,true);assert.equal(early.comparison.done,false);assert.equal(early.storyDone,false);
});
