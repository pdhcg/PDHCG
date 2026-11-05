import test from 'node:test';
import assert from 'node:assert/strict';
import {demoExamples, createComparison, sampleComparison, buildComparisonStory, sampleComparisonStory} from '../docs/javascripts/demo-saddle-comparison.mjs';
import {makeSaddleProblem, solveSaddle} from '../docs/javascripts/demo-saddle-solver.mjs';
import {makeConicProblem, solveConic, projectSOC, conicMetrics, conicLagrangian} from '../docs/javascripts/demo-conic-solver.mjs';
import {subproblemGeometry} from '../docs/javascripts/demo-subproblem.mjs';

const near = (a,b,tolerance=1e-10) => assert.ok(Math.abs(a-b)<=tolerance,`${a} differs from ${b}`);
const nearPoint = (a,b,tolerance=1e-10) => a.forEach((v,i)=>near(v,b[i],tolerance));
const candidate = r => [r.xp,r.yp];
const presets = [{},{tau:.3,sigma:.04/.3},
  {q:1,a:1.8,b:.5,tau:.6,sigma:.8,lower:-1.2,upper:.54,initial:[-.8,0],constraint:'inequality'}];
const norm=v=>Math.hypot(...v);
const difference=(a,b)=>a.map((v,i)=>v-b[i]);
const dot=(a,b)=>a.reduce((sum,v,i)=>sum+v*b[i],0);
const projection=(x,y)=>[x[0],-y[1]];
const inSOC=v=>norm(v.slice(1))<=v[0]+1e-10;

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
    assert.equal(events.introLength,8);
    assert.deepEqual(events.slice(0,events.introLength).map(e=>`${e.scope}:${e.part}`),[
      'outer:primal','inner:enter','inner:optimum',
      'inner:settle','dual:enter','dual:optimum','dual:settle','dual:exit']);
    assert.equal(events[4].transitionFrom,'inner');
    assert.ok(events.slice(1,events.introLength).every(e=>e.scope!=='outer'));
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

test('SOC projection covers the cone, polar cone and boundary with an orthogonal Moreau decomposition',()=>{
  for(const [input,expected] of [[[3,1,2],[3,1,2]],[[-3,1,2],[0,0,0]],
    [[1,3,4],[3,1.8,2.4]],[[0,0,0],[0,0,0]]]){
    const original=[...input],projected=projectSOC(input),residual=difference(input,projected);
    nearPoint(projected,expected);assert.deepEqual(input,original);
    assert.ok(inSOC(projected));assert.ok(inSOC(residual.map(v=>-v)));
    near(dot(projected,residual),0);
    near(norm(input)**2,norm(projected)**2+norm(residual)**2);
  }
});

test('the conic reference saddle independently satisfies feasibility, stationarity and complementarity',()=>{
  for(const options of [{},{target:[1,0]},{q:.2,target:[-8,2],offset:[.2,.3],coneRadius:2}]){
    const p=makeConicProblem(options),x=p.saddleX,y=p.saddleY;
    const slack=[p.coneRadius,...difference(x,p.offset)];
    assert.ok(norm(x)<p.radius);assert.ok(inSOC(slack));assert.ok(inSOC(y));
    nearPoint(x.map((v,i)=>p.q*(v-p.target[i])-y[i+1]),[0,0]);
    near(dot(y,slack),0);
    assert.ok(conicMetrics(p,x,y).error<1e-12);
    assert.ok(conicMetrics(p,[x[0]+.2,x[1]],y).error>1e-4);
  }
});

test('both conic methods converge with genuinely two-dimensional motion and active first projections',()=>{
  const comparison=createComparison(demoExamples.conic);
  for(const {problem:p,result} of [comparison.baseline,comparison.halpern]){
    assert.equal(result.status,'converged');
    const first=result.records[0],last=result.records.at(-1);
    assert.ok(norm(difference(first.primalCenter,first.vectorXp))>.1);
    assert.ok(norm(difference(first.dualCenter,first.vectorYp))>.1);
    near(norm(first.vectorXp),p.radius);near(norm(first.vectorYp.slice(1)),first.vectorYp[0]);
    for(const r of result.records){
      assert.ok(norm(r.vectorXp)<=p.radius+1e-10);assert.ok(inSOC(r.vectorYp));
      assert.ok([...r.vectorXp,...r.vectorYp,...r.vectorNext,r.metrics.error].every(Number.isFinite));
    }
    const cross=(a,b)=>a[0]*b[1]-a[1]*b[0];
    const xDirection=difference(first.vectorXp,p.saddleX),yDirection=difference(first.vectorYp,p.saddleY).slice(1);
    assert.ok(result.records.some(r=>Math.abs(cross(xDirection,difference(r.vectorXp,p.saddleX)))>.1));
    assert.ok(result.records.some(r=>Math.abs(cross(yDirection,difference(r.vectorYp,p.saddleY).slice(1)))>.1));
    assert.ok(norm(difference(last.vectorXp,p.saddleX))<3e-5);
    assert.ok(norm(difference(last.vectorYp,p.saddleY))<1e-5);
    assert.ok(conicMetrics(p,last.vectorXp,last.vectorYp).error<=result.tolerance);
  }
  assert.equal(solveConic(makeConicProblem(),{limit:1}).status,'iteration-limit');
});

test('the visible conic spiral consists of shrinking projected solver candidates for at least three turns',()=>{
  const {problem:p,result}=createComparison(demoExamples.conic).baseline;
  const initialRadius=norm(difference(p.initial,p.saddle)),turnRadii=[];
  let angle=Math.atan2(p.initial[1]-p.saddle[1],p.initial[0]-p.saddle[0]),rotation=0,visibleTurns=0;
  for(const r of result.records){
    const point=projection(r.vectorXp,r.vectorYp),radius=norm(difference(point,p.saddle));
    const nextAngle=Math.atan2(point[1]-p.saddle[1],point[0]-p.saddle[0]);
    rotation+=Math.atan2(Math.sin(nextAngle-angle),Math.cos(nextAngle-angle));angle=nextAngle;
    const turns=Math.abs(rotation)/(2*Math.PI);
    if(radius>.05*initialRadius)visibleTurns=turns;
    if(turns>=turnRadii.length+1)turnRadii.push(radius);
  }
  assert.ok(visibleTurns>=3,`Only ${visibleTurns} turns remain above 5% of the initial radius`);
  assert.ok(turnRadii[0]<initialRadius);
  for(let i=1;i<turnRadii.length;i++)assert.ok(turnRadii[i]<turnRadii[i-1]);
  assert.ok(norm(difference(candidate(result.records.at(-1)),p.saddle))<initialRadius*1e-4);
});

test('conic aliases, shared comparison clock and intro all preserve the same fixed vector projection',()=>{
  const comparison=createComparison(demoExamples.conic),events=buildComparisonStory(comparison);
  for(const {problem:p,result} of [comparison.baseline,comparison.halpern]){
    assert.deepEqual(p.initial,projection(p.initialX,p.initialY));
    assert.deepEqual(p.saddle,projection(p.saddleX,p.saddleY));
    for(const r of result.records){
      assert.deepEqual([r.x,r.y],projection(r.vectorX,r.vectorY));
      assert.deepEqual(candidate(r),projection(r.vectorXp,r.vectorYp));
      assert.equal(r.xbar,r.vectorXbar[0]);assert.equal(r.trialY,-r.dualCenter[1]);
      for(const name of ['anchor','reflection','next']){
        const full=r[`vector${name[0].toUpperCase()}${name.slice(1)}`];
        assert.deepEqual(r[name],projection(full.slice(0,2),full.slice(2)));
      }
    }
  }
  for(let k=0;k<=comparison.iterations;k++){
    const model=sampleComparison(comparison,k);
    for(const state of [model,model.comparison]){
      const n=Math.min(k,state.result.records.length),r=state.result.records[n-1];
      nearPoint(state.point,r?projection(r.vectorXp,r.vectorYp):projection(state.problem.initialX,state.problem.initialY));
      assert.equal(state.cursor,k);assert.equal(state.completed,n);
    }
  }
  const first=comparison.baseline.result.records[0];
  for(const e of events.slice(0,events.introLength)){
    const r=e.r;
    nearPoint(e.to,e.scope==='inner'?projection(e.part==='enter'?r.vectorX:r.vectorXp,r.vectorY):
      e.scope==='dual'?projection(r.vectorXp,e.part==='enter'?r.vectorY:r.vectorYp):projection(r.vectorX,r.vectorY));
  }
  const before=sampleComparisonStory(comparison,events,events.introLength-1e-7);
  const after=sampleComparisonStory(comparison,events,events.introLength);
  nearPoint(before.point,projection(first.vectorXp,first.vectorYp));
  nearPoint(after.point,before.point);nearPoint(after.comparison.point,before.point);
  for(const cursor of [1.37,22.6,comparison.halpern.result.records.length+.5]){
    const story=sampleComparisonStory(comparison,events,events.introLength+cursor-1);
    const shared=sampleComparison(comparison,cursor);
    near(story.compareCursor,cursor);nearPoint(story.point,shared.point);nearPoint(story.comparison.point,shared.comparison.point);
  }
});

test('conic contour geometry equals the true primal objective and an exact dual cone section',()=>{
  const comparison=createComparison(demoExamples.conic);
  for(const {problem:p,result} of [comparison.baseline,comparison.halpern]){
    for(const r of [result.records[0],result.records[Math.floor(result.records.length/2)],result.records.at(-1)]){
      const primal=subproblemGeometry(p,r,false),dual=subproblemGeometry(p,r,true);
      const primalValue=x=>conicLagrangian(p,x,r.vectorY)+norm(difference(x,r.vectorX))**2/(2*p.tau);
      const dualValue=y=>-conicLagrangian(p,r.vectorXbar,y)+norm(difference(y,r.vectorY))**2/(2*p.sigma);
      nearPoint(primal.optimum,r.vectorXp);assert.equal(primal.shape,'disk');
      for(const x of [[0,0],[p.radius,0],[-p.radius,0],[0,p.radius],[0,-p.radius]]){
        near(primal.value(...x)-primal.value(...primal.optimum),primalValue(x)-primalValue(r.vectorXp),1e-9);
        assert.ok(primalValue(x)>=primalValue(r.vectorXp)-1e-9);
      }
      // A disk normal and a polar-cone residual certify global optimality,
      // independently of the finite points used to compare contour values.
      const gradient=r.vectorXp.map((v,i)=>p.q*(v-p.target[i])-r.vectorY[i+1]+(v-r.vectorX[i])/p.tau);
      near(dot(gradient,r.vectorXp)+p.radius*norm(gradient),0,1e-8);
      const residual=difference(r.dualCenter,r.vectorYp);
      assert.ok(inSOC(residual.map(v=>-v)));near(dot(residual,r.vectorYp),0,1e-9);
      const lift=([s,t])=>[t,...dual.direction.map(v=>s*v)];
      near(norm(dual.direction),1);nearPoint(lift(dual.center),r.dualCenter);
      nearPoint(lift(dual.optimum),r.vectorYp);assert.equal(dual.shape,'cone');
      assert.ok(dual.optimum[1]>=Math.abs(dual.optimum[0])-1e-10);
      for(const uv of [[0,0],[1,1],[-1,1],[.3,2],[3,-2]]){
        near(dual.value(...uv)-dual.value(...dual.optimum),dualValue(lift(uv))-dualValue(r.vectorYp),1e-9);
        if(uv[1]>=Math.abs(uv[0]))assert.ok(dualValue(lift(uv))>=dualValue(r.vectorYp)-1e-9);
      }
    }
  }
});
