import test from 'node:test';
import assert from 'node:assert/strict';
import {makeSaddleProblem, solveSaddle, objective, lagrangian, localPrimal, localDual,
  saddleMetrics, projectBox} from '../docs/javascripts/demo-saddle-solver.mjs';
import {buildTimeline, buildInnerTimeline, buildDualTimeline, sampleTimeline} from '../docs/javascripts/demo-saddle-timeline.mjs';

const near = (a, b, tol = 1e-10) => assert.ok(Math.abs(a - b) <= tol, `${a} differs from ${b}`);
const nearPoint = (a,b,tol=1e-10) => a.forEach((v,i)=>near(v,b[i],tol));
const boundary = {q:1,a:1.8,b:.5,lower:-1.2,upper:.54,tau:.6,sigma:.8,
  initial:[-.8,0],constraint:'inequality'};
const presets = [{},{tau:.3,sigma:.04/.3},boundary];

function winding(p, records) {
  let previousAngle=Math.atan2(p.initial[1]-p.saddle[1],p.initial[0]-p.saddle[0]);
  let turns=0, visible=0;
  const radii=[Math.hypot(p.initial[0]-p.saddle[0],p.initial[1]-p.saddle[1])];
  for(const r of records){
    const radius=Math.hypot(r.xp-p.saddle[0],r.yp-p.saddle[1]);
    const angle=Math.atan2(r.yp-p.saddle[1],r.xp-p.saddle[0]);
    let delta=angle-previousAngle;
    while(delta>Math.PI)delta-=2*Math.PI;
    while(delta<-Math.PI)delta+=2*Math.PI;
    turns+=delta/(2*Math.PI);previousAngle=angle;
    if(radius>.05*radii[0])visible=turns;
    if(turns>=radii.length)radii.push(radius);
  }
  return {turns,visible,radii};
}

test('projected BB exposes two real default steps and the exact box-constrained proximal solution', () => {
  const p = makeSaddleProblem(), result = solveSaddle(p), r = result.records[0];
  assert.equal(result.status, 'converged');
  assert.equal(p.constraint,'equality');assert.equal(p.halpern,false);
  assert.equal(r.innerSteps.length,2);
  nearPoint(r.primalPath,[2,1.805,1.613861386138614]);near(r.yp,1.6455445544554457);
  assert.ok(r.innerSteps.every(s=>!s.projected),'An inactive box must not create a fake projection');
  for(const options of [...presets,{...boundary,innerInitialScale:1}]){
    const example=makeSaddleProblem(options);
    for(const row of solveSaddle(example).records){
      near(row.xp,projectBox(example,(row.x+example.tau*(example.q*example.a-row.y))/(1+example.tau*example.q)));
      near(row.primalPath[0],row.x);
      row.innerSteps.forEach((s,j)=>{
        near(s.trial,s.from-s.alpha*s.gradient);near(s.to,projectBox(example,s.trial));
        near(s.from,row.primalPath[j]);near(s.to,row.primalPath[j+1]);
        near(s.gradient,example.q*(s.from-example.a)+row.y+(s.from-row.x)/example.tau);
      });
    }
  }
});

test('plain PDHG follows the exact linear recurrence and makes several genuinely contracting turns', () => {
  for(const options of presets.slice(0,2)){
    const p=makeSaddleProblem(options),result=solveSaddle(p),alpha=1/(1+p.tau*p.q);
    assert.equal(result.status,'converged');assert.ok(result.records.every(r=>!r.restart));
    for(const r of result.records){
      const ex=r.x-p.b,ey=r.y-p.saddle[1];
      near(r.xp-p.b,alpha*(ex-p.tau*ey));
      near(r.yp-p.saddle[1],p.sigma*(2*alpha-1)*ex+(1-2*p.sigma*alpha*p.tau)*ey);
      nearPoint(r.next,[r.xp,r.yp]);
      assert.ok(r.xp>p.lower&&r.xp<p.upper);
      assert.ok(r.innerSteps.every(s=>!s.projected));
    }
    const {visible,radii}=winding(p,result.records);
    assert.ok(visible>3,`Only ${visible} turns remain above 5% of the initial radius`);
    for(let j=1;j<=3;j++)assert.ok(radii[j]<.65*radii[j-1],'Each full turn should visibly contract');
    assert.ok(result.records.some(r=>r.yp<0),'Equality multipliers must be free to cross zero');
    assert.ok(result.records.at(-1).metrics.error<=result.tolerance);
  }
});

test('dual signs, optional reflected Halpern and native-style restarts remain numerically genuine', () => {
  for(const options of presets)for(const halpern of [false,true]){
    const p=makeSaddleProblem({...options,halpern}),result=solveSaddle(p);
    assert.equal(result.status,'converged');
    assert.equal(result.records.some(r=>r.restart),halpern);
    for(const r of result.records){
      near(r.xbar,2*r.xp-r.x);near(r.trialY,r.y+p.sigma*(r.xbar-p.b));
      near(r.yp,p.constraint==='equality'?r.trialY:Math.max(0,r.trialY));
      nearPoint(r.reflection,[2*r.xp-r.x,2*r.yp-r.y]);
      if(halpern){
        near(r.theta,(r.cycle+1)/(r.cycle+2));
        nearPoint(r.next,r.reflection.map((v,i)=>(1-r.theta)*r.anchor[i]+r.theta*v));
      } else nearPoint(r.next,[r.xp,r.yp]);
      if(r.restart){const prev=result.records[r.iteration-2];nearPoint(r.anchor,[prev.xp,prev.yp]);assert.equal(r.cycle,0);}
    }
    nearPoint([result.records.at(-1).xp,result.records.at(-1).yp],p.saddle,1e-5);
  }
});

test('the displayed proximal curve is the true objective, including solutions on the box boundary', () => {
  for(const options of [{},{...boundary,q:3,a:2,b:1},{...boundary,a:-2,b:1}]){
    const p=makeSaddleProblem(options);
    for(const r of solveSaddle(p).records){
      near(localPrimal(p,r,r.x),lagrangian(p,r.x,r.y));
      const g=p.q*(r.xp-p.a)+r.y+(r.xp-r.x)/p.tau;
      near(r.xp,projectBox(p,r.xp-g));
      for(const u of [p.lower,p.upper,(p.lower+p.upper)/2])
        assert.ok(localPrimal(p,r,u)>=localPrimal(p,r,r.xp)-1e-10);
    }
  }
});

test('the dual proximal curve is concave, uses extrapolated x, and is maximized by the true dual update', () => {
  for(const options of [...presets,{...boundary,a:-.3,initial:[-.8,1],halpern:true}]){
    const p=makeSaddleProblem(options),result=solveSaddle(p);
    for(const r of result.records){
      near(localDual(p,r,r.y),lagrangian(p,r.xbar,r.y));
      const slope=v=>(r.xbar-p.b)-(v-r.y)/p.sigma;
      near(slope(r.trialY),0);
      const project=v=>p.constraint==='equality'?v:Math.max(0,v);
      near(r.yp,project(r.yp+slope(r.yp)));
      for(const v of [-3,-1,0,1,3,r.yp-1,r.yp+1].filter(v=>project(v)===v))
        assert.ok(localDual(p,r,r.yp)>=localDual(p,r,v)-1e-10);
      near(localDual(p,r,r.trialY-1)+localDual(p,r,r.trialY+1)-2*localDual(p,r,r.trialY),-1/p.sigma);
    }
  }
  const p=makeSaddleProblem(),r=solveSaddle(p).records[0];
  assert.ok(Math.abs(r.trialY-(r.y+p.sigma*(r.xp-p.b)))>.05,
    'The displayed pair coordinate x+ must not replace extrapolated x in the dual gradient');
});

test('inequality and equality modes converge to independent KKT solutions', () => {
  for(const halpern of [false,true])for(const q of [.5,1,2,4])for(const a of [-2,-.3,.5,1.8,2.5])for(const b of [.5,1]){
    const p=makeSaddleProblem({...boundary,q,a,b,halpern}),result=solveSaddle(p),r=result.records.at(-1);
    assert.equal(result.status,'converged',`q=${q},a=${a},b=${b},halpern=${halpern}`);
    assert.ok(r.metrics.error<=result.tolerance);
    near(r.xp,Math.max(p.lower,Math.min(a,b,p.upper)),1e-5);
    near(r.yp,b<=p.upper?q*Math.max(0,a-b):0,3e-5);
    near(objective(p,r.xp),objective(p,p.saddle[0]),4e-5);
    assert.ok(saddleMetrics(p,...p.saddle).error<1e-12);
  }
  for(const a of [1,3,5]){
    const p=makeSaddleProblem({a}),result=solveSaddle(p),r=result.records.at(-1);
    assert.equal(result.status,'converged');near(r.xp,p.b,1e-5);near(r.yp,p.q*(a-p.b),1e-5);
    near(saddleMetrics(p,...p.saddle).error,0);
  }
});

test('limits, feasibility and invalid inputs are checked', () => {
  assert.equal(solveSaddle(makeSaddleProblem(),{limit:1}).status,'iteration-limit');
  for(const options of [{q:0},{q:NaN},{a:Infinity},{tau:1,sigma:1},{sigma:-1},
    {initial:[0]},{initial:[0,NaN]},{lower:1,upper:0},{lower:0,b:-1},{b:7},
    {constraint:'unknown'},{halpern:'false'},{innerInitialScale:0}])
    assert.throws(()=>makeSaddleProblem(options));
  assert.throws(()=>solveSaddle(makeSaddleProblem(),{limit:0}));
});

test('the main story automatically expands primal and dual, then follows real candidates without jumps', () => {
  for(const options of presets)for(const halpern of [false,true]){
    const p=makeSaddleProblem({...options,halpern}),result=solveSaddle(p),events=buildTimeline(p,result);
    const firstInner=buildInnerTimeline(p,result.records[0]),firstDual=buildDualTimeline(p,result.records[0]);
    assert.equal(events.length,result.records.length+1+firstInner.length+firstDual.length);
    assert.equal(events.scope,'outer');assert.equal(events[0].part,'primal');
    nearPoint(events[0].from,events[0].to);assert.equal(events[0].duration,900);
    const dualIndex=events.findIndex(e=>e.part==='dual');
    assert.equal(dualIndex,firstInner.length+1);
    assert.deepEqual(events.slice(1,dualIndex).map(e=>e.part),firstInner.map(e=>e.part));
    assert.ok(events.slice(1,dualIndex).every(e=>e.scope==='inner'&&e.iteration===0));
    const trajectoryIndex=dualIndex+1+firstDual.length;
    assert.deepEqual(events.slice(dualIndex+1,trajectoryIndex).map(e=>e.part),firstDual.map(e=>e.part));
    assert.ok(events.slice(dualIndex+1,trajectoryIndex).every(e=>e.scope==='dual'&&e.iteration===0));
    assert.ok(events.slice(trajectoryIndex).every(e=>e.scope==='outer'));
    for(let i=0;i<events.length;i++){
      const e=events[i];
      if(i){
        nearPoint(e.from,events[i-1].to);near(e.liftFrom,events[i-1].liftTo);
        near(e.blendFrom,events[i-1].blendTo);near(e.dualFrom,events[i-1].dualTo);
      }
      if(e.part==='dual'){
        nearPoint(e.from,[e.r.xp,e.r.y]);nearPoint(e.to,e.from);
        assert.equal(e.iteration,0);assert.equal(e.mode,'detail');assert.equal(e.duration,700);
      }
      if(e.part==='trajectory'){
        const previous=result.records[e.iteration-1];
        nearPoint(e.from,[previous.xp,previous.yp]);nearPoint(e.to,[e.r.xp,e.r.yp]);
        assert.equal(e.mode,'trajectory');assert.equal(e.completed,e.iteration);
        assert.ok(e.duration>=8&&e.duration<=160);assert.match(e.title,/PDHG/);
        assert.equal(e.title.includes('Halpern'),halpern);
      }
    }
    const segments=events.filter(e=>e.mode==='trajectory');
    assert.ok(segments.at(-1).duration<segments[0].duration,'Small late steps should play faster');
    assert.ok(events.reduce((sum,e)=>sum+e.duration,0)<30000);
    near(sampleTimeline(events,1).innerBlend,0);
    near(sampleTimeline(events,dualIndex).innerBlend,0);
    near(sampleTimeline(events,dualIndex+1).dualBlend,0);
    near(sampleTimeline(events,trajectoryIndex).dualBlend,0);
  }
});

test('dual close-up shows real ascent and projection while holding the pairing coordinate x+ fixed', () => {
  const examples=[{}, {...boundary,a:-.3,initial:[-.8,1]}];
  let hasProjection=false;
  for(const options of examples){
    const p=makeSaddleProblem(options),result=solveSaddle(p);
    const r=p.constraint==='equality'?result.records[0]:result.records.find(r=>r.trialY<0);
    assert.ok(r);
    const events=buildDualTimeline(p,r);
    assert.deepEqual(events.map(e=>e.part),['enter','ascent','project','settle','exit']);
    assert.equal(events.scope,'dual');
    const ascent=events[1],projection=events[2];
    nearPoint(ascent.from,[r.xp,r.y]);nearPoint(ascent.to,[r.xp,r.trialY]);
    nearPoint(projection.from,[r.xp,r.trialY]);nearPoint(projection.to,[r.xp,r.yp]);
    if(p.constraint==='equality'){
      nearPoint(projection.from,projection.to);assert.match(projection.caption,/free/);
    } else {
      assert.ok(projection.from[1]<0);near(projection.to[1],0);hasProjection=true;
    }
    for(let i=0;i<events.length;i++){
      const e=events[i];assert.equal(e.scope,'dual');assert.equal(e.stage,1);
      near(e.from[0],r.xp);near(e.to[0],r.xp);
      if(i){nearPoint(e.from,events[i-1].to);near(e.dualFrom,events[i-1].dualTo);}
      const state=sampleTimeline(events,i+.5);
      near(state.innerBlend,0);near(state.lift,0);near(state.localOpacity,0);
      near(state.dualBlend,(e.dualFrom+e.dualTo)/2);
    }
    const final=sampleTimeline(events,events.length);
    assert.equal(final.done,true);assert.equal(final.scope,'dual');assert.equal(final.mode,'detail');
    assert.equal(final.completed,r.iteration);near(final.dualBlend,0);nearPoint(final.point,[r.xp,r.yp]);
  }
  assert.ok(hasProjection,'Test an actual dual-cone projection, not only an identity');
});

test('standalone inner inspection retains true projection steps and returns only the solved primal coordinate', () => {
  const p=makeSaddleProblem(boundary),result=solveSaddle(p),r=result.records[1];
  const events=buildInnerTimeline(p,r),outer=buildTimeline(p,result);
  assert.equal(events[0].part,'enter');assert.equal(events.at(-1).part,'exit');
  assert.match(events[0].caption,/outer iteration waits for x⁺/);
  assert.equal(events.find(e=>e.part==='settle').duration,1400);
  assert.equal(events.at(-1).duration,1300);
  for(let i=0;i<events.length;i++){
    const e=events[i];assert.equal(e.scope,'inner');near(e.from[1],r.y);near(e.to[1],r.y);
    if(i){nearPoint(e.from,events[i-1].to);near(e.liftFrom,events[i-1].liftTo);near(e.blendFrom,events[i-1].blendTo);}
    if(e.part==='gradient'){const s=r.innerSteps[e.innerStep];near(e.from[0],s.from);near(e.to[0],s.trial);}
    if(e.part==='project'){const s=r.innerSteps[e.innerStep];near(e.from[0],s.trial);near(e.to[0],s.to);}
  }
  const final=sampleTimeline(events,events.length);
  assert.equal(final.done,true);assert.equal(final.innerBlend,0);nearPoint(final.point,[r.xp,r.y]);
  assert.equal(final.completed,1);
  const firstInner=buildInnerTimeline(p,result.records[0]);
  nearPoint(sampleTimeline(firstInner,firstInner.length).point,outer.find(e=>e.part==='dual').from);
  const ownSegment=outer.find(e=>e.iteration===1&&e.part==='trajectory');
  near(ownSegment.to[0],final.point[0]);
});

test('trajectory sampling is linear and continuous while detail and inner sampling retain endpoint holds', () => {
  const p=makeSaddleProblem(),result=solveSaddle(p);
  for(const events of [buildTimeline(p,result),buildInnerTimeline(p,result.records[1]),buildDualTimeline(p,result.records[1])]){
    for(let i=0;i<events.length;i++){
      const e=events[i],start=sampleTimeline(events,i),middle=sampleTimeline(events,i+.5);
      nearPoint(start.point,e.from);
      for(const t of [.05,.2,.8,.95]){
        const sampled=sampleTimeline(events,i+t);
        if(e.mode==='trajectory')nearPoint(sampled.point,e.from.map((v,j)=>v+t*(e.to[j]-v)));
        else if(t===.05)nearPoint(sampled.point,e.from);
        else if(t===.95)nearPoint(sampled.point,e.to);
      }
      nearPoint(middle.point,e.from.map((v,j)=>(v+e.to[j])/2));
      near(middle.lift,(e.liftFrom+e.liftTo)/2);near(middle.innerBlend,(e.blendFrom+e.blendTo)/2);
      near(middle.dualBlend,(e.dualFrom+e.dualTo)/2);
      nearPoint(middle.motionPath.at(-1),middle.point);
    }
    assert.equal(sampleTimeline(events,-1).cursor,0);assert.equal(sampleTimeline(events,NaN).cursor,0);
  }
  assert.equal(sampleTimeline([],0).done,true);
});

test('the main outcome is the final computed candidate, including a single-record run', () => {
  for(const limit of [1,1000]){
    const p=makeSaddleProblem(),result=solveSaddle(p,{limit}),events=buildTimeline(p,result),last=result.records.at(-1);
    const final=sampleTimeline(events,events.length+10);
    assert.equal(final.done,true);nearPoint(final.point,[last.xp,last.yp]);
    assert.equal(final.completed,result.records.length);assert.equal(final.scope,'outer');
    near(final.dualBlend,0);
    assert.equal(final.title,result.status==='converged'?'At the saddle point':'Run complete');
    assert.equal(final.mode,'trajectory');
  }
});

test('the animation hides Halpern constructions without changing any solver records', () => {
  const p=makeSaddleProblem({halpern:true}),result=solveSaddle(p),before=JSON.stringify(result),events=buildTimeline(p,result);
  assert.ok(result.records.some(r=>r.restart));
  assert.ok(result.records.some(r=>r.reflection[0]!==r.xp));
  events.forEach((e,i)=>{
    assert.notEqual(e.part,'halpern');assert.notEqual(e.part,'restart');
    for(const t of [.05,.25,.5,.75,.95]){
      const state=sampleTimeline(events,i+t);
      assert.equal(state.mode,e.mode);
      nearPoint(state.motionPath.at(-1),state.point);
    }
  });
  assert.equal(JSON.stringify(result),before,'Playback must not alter any numerical solver record');
});

test('the inequality example retains substantive box projection and genuine dual cone projection', () => {
  const p=makeSaddleProblem(boundary),result=solveSaddle(p),r=result.records[1];
  const project=buildInnerTimeline(p,r).find(e=>e.part==='project'&&e.r.innerSteps[e.innerStep].projected);
  assert.ok(project);assert.ok(project.from[0]>project.to[0]);near(project.to[0],p.upper);
  const original=solveSaddle(makeSaddleProblem({...boundary,innerInitialScale:1})).records[0];
  near(original.innerSteps[0].trial,.76);near(original.innerSteps[0].to,.54);near(original.xp,.175);
  const interior=solveSaddle(makeSaddleProblem({...boundary,a:-.3,initial:[-.8,1],halpern:true}));
  assert.ok(interior.records.some(r=>r.trialY<0&&r.yp===0));
  assert.ok(interior.records.some(r=>r.reflection[1]<0));
  assert.equal(interior.status,'converged');
});
