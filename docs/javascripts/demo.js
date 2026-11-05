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

import {buildTimeline, buildInnerTimeline, buildDualTimeline, sampleTimeline} from './demo-saddle-timeline.mjs';
import {SaddleScene} from './demo-saddle-scene.mjs';
import {demoExamples, createComparison, buildComparisonStory, sampleComparisonStory} from './demo-saddle-comparison.mjs';

const root=document.getElementById('pdhcg-demo');
if(root)initialize();

function initialize(){
  const $=id=>document.getElementById(`demo-${id}`);
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const presets=demoExamples;
  const number=value=>Number(value.toFixed(2)).toFixed(2).replace('-', '−');
  const vector=values=>`(${values.map(number).join(', ')})`;
  const vectorMarkup=values=>`<span class="demo-vector">${values.map((value,i)=>
    `<span>${i===0?'(':''}${number(value)}${i===values.length-1?')':','}</span>`).join('')}</span>`;
  const vectorInput=(label,values)=>`<span class="demo-vector-input"><span class="demo-vector-symbol">${label} =</span>${vectorMarkup(values)}</span>`;
  const statusLabel=status=>({'converged':'Converged','iteration-limit':'Iteration limit',
    'numerical-error':'Numerical error'}[status]??'Run complete');
  const completedLabel=run=>`${statusLabel(run.result.status)} · ${run.result.records.length}`;
  let scene;
  try{scene=new SaddleScene($('scene'));}catch(error){
    $('alert').hidden=false;$('alert').textContent='The 3D view needs WebGL. Enable hardware acceleration or try a different browser.';console.error(error);return;
  }
  let p,result,outerEvents,innerEvents=null,outerCursor=0,innerCursor=0,resumeCursor=0;
  let playing=false,stepTarget=null,frame,lastTime=0,lastText='';
  let view='compare',comparison,comparisonEvents,compareCursor=0;
  const mainEvents=()=>view==='compare'?comparisonEvents:outerEvents;
  const mainCursor=()=>view==='compare'?compareCursor:outerCursor;
  const activeEvents=()=>innerEvents||mainEvents();
  const activeCursor=()=>innerEvents?innerCursor:mainCursor();
  const activeProblem=()=>view==='compare'?comparison.baseline.problem:p;
  const activeResult=()=>view==='compare'?comparison.baseline.result:result;
  const activeState=()=>view==='compare'&&!innerEvents?sampleComparisonStory(comparison,comparisonEvents,compareCursor):sampleTimeline(activeEvents(),activeCursor());
  function setMainCursor(value){if(view==='compare')compareCursor=value;else outerCursor=value;}
  function setCursor(value){if(innerEvents)innerCursor=value;else setMainCursor(value);}
  function pause(){playing=false;stepTarget=null;cancelAnimationFrame(frame);lastTime=0;}
  function renderComparison(dt,state){
    const done=state.storyDone,conic=comparison.baseline.problem.kind==='conic';
    const statuses=[comparison.baseline.result.status,comparison.halpern.result.status];
    const finalStatus=statuses.includes('numerical-error')?'numerical-error':
      statuses.find(status=>status!=='converged')??'converged';
    root.dataset.level='outer';root.dataset.mode='trajectory';root.dataset.operation='trajectory';
    root.dataset.transfer='';root.dataset.innerReady='false';root.dataset.innerPart='';root.dataset.innerCursor='';root.dataset.phase='2';
    root.dataset.status=done?finalStatus:playing?'playing':'paused';
    root.dataset.iteration=String(Math.floor(state.compareCursor));root.dataset.compareCursor=state.compareCursor.toFixed(4);
    $('play').textContent=playing?'Ⅱ Pause':done?'↻ Replay':'▶ Play';
    $('step').disabled=done;$('timeline').value=String(compareCursor);
    $('trajectory-title').textContent='Primal–dual iterates';
    $('trajectory-iteration').textContent=`Iteration ${Math.floor(state.compareCursor)} · same starting point`;
    $('plain-status').textContent=state.done?completedLabel(comparison.baseline):'';
    $('halpern-status').textContent=state.comparison.done?completedLabel(comparison.halpern):'';
    $('scene').setAttribute('aria-label',`${conic?'Fixed two-dimensional projection of the primal and dual vectors':'PDHG trajectories on the same saddle'}: teal without Halpern, purple with Halpern. Both advance at the same iteration count. Drag or use arrow keys to rotate. R resets the view.`);
    state.animateCamera=playing||stepTarget!==null;
    scene.update(comparison.baseline.problem,comparison.baseline.result,state,dt);
  }
  function render(dt=16){
    if(!outerEvents)return;
    const state=activeState(),primal=state.scope==='inner',dual=state.scope==='dual',inside=primal||dual;
    const problem=activeProblem(),run=activeResult(),conic=problem.kind==='conic';
    root.dataset.storyPhase=view==='compare'&&!innerEvents?state.storyPhase:'detail';
    root.dataset.cursor=mainCursor().toFixed(4);root.dataset.part=state.part;
    root.dataset.compareCursor=state.compareCursor===undefined?'':state.compareCursor.toFixed(4);
    if(view==='compare'&&state.storyPhase==='trajectory'){renderComparison(dt,state);return;}
    root.dataset.level=inside?state.scope:'outer';
    root.dataset.mode=inside?'detail':state.mode;
    root.dataset.operation=state.done&&!inside?'complete':primal||state.stage===0?'primal':dual||state.stage===1?'dual':'trajectory';
    root.dataset.transfer=inside?(['settle','exit'].includes(state.part)?'return':state.part==='enter'?'enter':'working'):'';
    const ready=inside&&state.solveProgress>=1-1e-7;
    root.dataset.innerReady=String(ready);
    $('inspect').setAttribute('aria-expanded',String(primal));
    $('inspect-dual').setAttribute('aria-expanded',String(dual));
    $('inspect').disabled=false;$('inspect-dual').disabled=false;
    if(inside){
      const inputs=conic?vectorInput(dual?'x̄':'xₖ',dual?state.r.vectorXbar:state.r.vectorX)+vectorInput('yₖ',state.r.vectorY):
        dual?`<span>x̄ = ${state.r.xbar.toFixed(3)}</span><span>yₖ = ${state.r.y.toFixed(3)}</span>`:
        `<span>xₖ = ${state.r.x.toFixed(3)}</span><span>yₖ = ${state.r.y.toFixed(3)}</span>`;
      if($('inner-input').innerHTML!==inputs)$('inner-input').innerHTML=inputs;
      const coordinate=`${dual?'y':'x'}⁺`,values=dual?state.r.vectorYp:state.r.vectorXp;
      const output=!ready?coordinate:conic?`<span class="demo-vector-symbol">${coordinate} ∈ ${dual?'𝒬₃*':'C'}</span>${vectorMarkup(values)}`:
        `${coordinate} = ${(dual?state.r.yp:state.r.xp).toFixed(3)}`;
      if($('inner-output').innerHTML!==output)$('inner-output').innerHTML=output;
      $('inner-output').setAttribute('aria-label',ready&&conic?`${coordinate} = ${vector(values)}`:output);
      $('subproblem-label').textContent=dual?'Dual proximal subproblem':'Primal proximal subproblem';
      $('subproblem-module').textContent=dual?'Dual':'Primal';
      $('return').firstElementChild.textContent=primal?'Continue to dual':'View trajectory';
    }
    root.dataset.phase=String(state.stage);root.dataset.iteration=String(state.iteration);root.dataset.part=state.part;
    root.dataset.innerCursor=inside?(innerEvents?innerCursor:mainCursor()-mainEvents().findIndex(e=>e.scope===state.scope)).toFixed(4):'';
    root.dataset.status=state.done&&!inside?run.status:playing?'playing':'paused';
    root.dataset.halpern=String(problem.halpern);
    $('play').textContent=playing?'Ⅱ Pause':state.done&&!inside?'↻ Replay':'▶ Play';
    $('step').disabled=state.done&&!inside;
    $('timeline').value=String(mainCursor());
    $('iteration').textContent=`Iteration ${state.iteration+1}`;
    $('trajectory-title').textContent='Primal–dual iterates';
    $('trajectory-iteration').textContent=`${problem.halpern?'PDHG + Halpern':'PDHG'} · ${state.iteration+1} / ${run.records.length}${state.done&&!inside?` · ${statusLabel(run.status)}`:''}`;
    let title=state.title,caption=state.caption;
    if(lastText!==title+caption){
      lastText=title+caption;$('stage-title').textContent=title;$('stage-caption').textContent=caption;
      const description=conic?(primal?'Primal proximal objective over the primal domain':dual?'Dual proximal objective on a two-dimensional cross-section of the second-order dual cone':'Primal–dual iterates shown in the x₁ and −y₁ coordinates'):
        primal?'Primal proximal subproblem':dual?'Dual proximal subproblem':'Primal–dual iterates';
      $('scene').setAttribute('aria-label',`${description}. ${title}. ${caption}${inside?'':' Drag or use arrow keys to rotate the outer view. R resets the view.'}`);
    }
    root.dataset.innerPart=inside?state.part:'';
    state.animateCamera=playing||stepTarget!==null;
    if(view==='compare')state.comparisonBounds=comparison.halpern.result;
    scene.update(activeProblem(),activeResult(),state,dt);
  }
  function resumeAfter(scope){
    const index=mainEvents().findIndex(e=>scope==='inner'?e.scope==='dual':e.mode==='trajectory');
    return index<0?mainEvents().length:index;
  }
  function enterInner(scope='inner'){
    const index=mainEvents().findIndex(e=>e.scope===scope);
    if(index<0)return;
    const previous=activeState(),before=previous.point;
    setMainCursor(index);resumeCursor=resumeAfter(scope);
    innerEvents=(scope==='dual'?buildDualTimeline:buildInnerTimeline)(activeProblem(),mainEvents()[index].r);innerCursor=0;
    if(scope==='inner')innerEvents.pop();
    const entry=innerEvents[0];
    if(['inner','dual'].includes(previous.scope)&&previous.scope!==scope){
      Object.assign(entry,{transitionFrom:previous.scope},previous.scope==='inner'?
        {localFrom:1,liftFrom:1,blendFrom:1}:{dualFrom:1});
    }else if(!reduced.matches&&Math.hypot(...before.map((v,i)=>v-entry.from[i]))>1e-8){
      innerEvents.unshift({...entry,part:'revisit',from:[...before],to:[...entry.from],duration:450,
        localTo:0,liftTo:0,blendTo:0,dualTo:0});
    }
    stepTarget=null;lastText='';render();
  }
  function finishInner(){innerEvents=null;setMainCursor(resumeCursor);innerCursor=0;lastText='';}
  function tick(now){
    const dt=lastTime?Math.min(75,now-lastTime):16;lastTime=now;
    let remaining=dt*Number($('speed').value);
    while(remaining>0&&activeCursor()<activeEvents().length){
      const events=activeEvents(),cursor=activeCursor(),index=Math.floor(cursor);
      const amount=Math.min(remaining/events[index].duration,1-(cursor-index));
      setCursor(cursor+amount);remaining-=amount*events[index].duration;
      if(stepTarget!==null&&activeCursor()>=stepTarget){setCursor(stepTarget);pause();break;}
      if(innerEvents&&innerCursor>=innerEvents.length)finishInner();
      if(amount<1e-9)break;
    }
    if(innerEvents&&innerCursor>=innerEvents.length)finishInner();
    if(!innerEvents&&mainCursor()>=mainEvents().length){setMainCursor(mainEvents().length);pause();}
    render(dt);if(playing||stepTarget!==null)frame=requestAnimationFrame(tick);
  }
  function start(){
    if(!innerEvents&&mainCursor()>=mainEvents().length){setMainCursor(0);scene.reset();}
    playing=true;stepTarget=null;lastTime=0;cancelAnimationFrame(frame);frame=requestAnimationFrame(tick);render();
  }
  function compute(){
    pause();
    try{
      comparison=createComparison(presets[$('example').value]);comparisonEvents=buildComparisonStory(comparison);
      compareCursor=reduced.matches?comparisonEvents.length:0;
      ({problem:p,result}=comparison[$('halpern').checked?'halpern':'baseline']);
      root.dataset.problem=p.kind==='conic'?'conic':'scalar';
      $('outer').textContent='PDHG iteration';
      outerEvents=buildTimeline(p,result);
      innerEvents=null;outerCursor=0;lastText='';
      $('timeline').max=String(mainEvents().length);$('timeline').step='0.001';
      const coefficient=p.q===1?'½':Number((p.q/2).toPrecision(5));
      if(p.kind==='conic'){
        const tuple=values=>`(${values.map(value=>Number(value.toFixed(2))).join(', ')})`;
        $('problem-formula').innerHTML=`<span class="demo-min">min<sub>‖x‖ ≤ ${p.radius}</sub></span><span>${coefficient}‖x − ${tuple(p.target)}‖²</span><span class="demo-subject">subject to</span><span>(${p.coneRadius}, x − ${tuple(p.offset)}) ∈ 𝒬₃</span>`;
        $('problem-formula').setAttribute('aria-label',`Minimize ${p.q/2} times the squared Euclidean distance from x to ${p.target.join(', ')}, with norm x at most ${p.radius}, subject to (${p.coneRadius}, x minus (${p.offset.join(', ')})) in the three-dimensional second-order cone.`);
      }else{
        const equality=p.constraint==='equality',relation=equality?'=':'≤';
        $('problem-formula').innerHTML=`<span class="demo-min">min<sub>${p.lower} ≤ x ≤ ${p.upper}</sub></span><span>${coefficient}(x − ${p.a})²</span><span class="demo-subject">subject to</span><span>x ${relation} ${p.b}</span>`;
        $('problem-formula').setAttribute('aria-label',`Minimize ${p.q/2} times x minus ${p.a} squared, with x in [${p.lower}, ${p.upper}], subject to x ${equality?'equal to':'less than or equal to'} ${p.b}`);
      }
      const error=[comparison.baseline,comparison.halpern].some(run=>run.result.status==='numerical-error');
      $('alert').hidden=!error;
      if(error)$('alert').textContent='The numerical reference stopped unexpectedly. Choose another example.';
      render();
    }catch(error){$('alert').hidden=false;$('alert').textContent=`Unable to compute this example: ${error.message}`;console.error(error);}
  }
  $('play').addEventListener('click',()=>{if(playing){pause();render();}else start();});
  $('step').addEventListener('click',()=>{
    pause();
    const target=Math.min(activeEvents().length,Math.floor(activeCursor()+1e-6)+1);
    if(reduced.matches){setCursor(target);if(innerEvents&&innerCursor>=innerEvents.length)finishInner();render();}
    else{stepTarget=target;lastTime=0;frame=requestAnimationFrame(tick);}
  });
  function replaySteps(){pause();innerEvents=null;outerCursor=0;compareCursor=0;scene.reset();render();}
  $('rewind').addEventListener('click',replaySteps);
  $('replay-steps').addEventListener('click',()=>{replaySteps();if(!reduced.matches)start();});
  $('finish').addEventListener('click',()=>{pause();innerEvents=null;setMainCursor(mainEvents().length);render();});
  $('timeline').addEventListener('input',()=>{pause();innerEvents=null;setMainCursor(Number($('timeline').value));render();});
  function switchView(next){
    if(view===next)return;
    pause();innerEvents=null;view=next;root.dataset.view=view;
    $('show-comparison').setAttribute('aria-pressed',String(view==='compare'));
    $('show-steps').setAttribute('aria-pressed',String(view==='steps'));
    $('timeline').max=String(mainEvents().length);
    if(view==='steps'){innerEvents=null;outerCursor=0;lastText='';}
    scene.reset();render();
    if(!reduced.matches&&mainCursor()<mainEvents().length)start();
  }
  $('show-comparison').addEventListener('click',()=>switchView('compare'));
  $('show-steps').addEventListener('click',()=>switchView('steps'));
  $('example').addEventListener('change',()=>{compute();if(!reduced.matches)start();});
  $('halpern').addEventListener('change',()=>{
    const compareTrajectory=activeState().mode==='trajectory';
    compute();
    if(compareTrajectory){outerCursor=outerEvents.findIndex(e=>e.mode==='trajectory');render();}
    if(!reduced.matches)start();
  });
  function inspect(scope){
    if(activeState().scope===scope){returnOuter();return;}
    pause();enterInner(scope);
    if(!reduced.matches)start();else{innerCursor=2;render();}
  }
  $('inspect').addEventListener('click',()=>inspect('inner'));
  $('inspect-dual').addEventListener('click',()=>inspect('dual'));
  function returnOuter(skipDual=false){
    const state=activeState();
    if(!['inner','dual'].includes(state.scope))return;
    const dual=state.scope==='dual';
    pause();
    if(!dual&&!skipDual){
      enterInner('dual');
      if(reduced.matches){innerCursor=2;render();}else start();
      return;
    }
    if(skipDual){innerEvents=null;setMainCursor(resumeAfter('dual'));render();return;}
    if(!innerEvents){
      resumeCursor=resumeAfter(state.scope);
      innerEvents=(dual?buildDualTimeline:buildInnerTimeline)(activeProblem(),state.r);
    }
    if(reduced.matches){finishInner();render();return;}
    // Return the computed subproblem optimum through the same transition.
    const exit=innerEvents.at(-1),settle={...exit,part:'settle',from:[...state.point],to:[state.r.xp,dual?state.r.yp:state.r.y],
      duration:500,localFrom:state.localOpacity,localTo:dual?0:1,liftFrom:state.lift,liftTo:dual?0:1,
      blendFrom:state.innerBlend,blendTo:dual?0:1,dualFrom:state.dualBlend||0,dualTo:dual?1:0};
    innerEvents=[settle,exit];innerEvents.scope=state.scope;innerEvents.status='solved';innerCursor=0;start();
  }
  $('return').addEventListener('click',()=>returnOuter());$('outer').addEventListener('click',()=>returnOuter(true));
  scene.onViewChange=({magnification,overview})=>{
    const scale=Number(magnification.toPrecision(3)).toLocaleString('en-US');
    const label=overview?'Full trajectory':`Local view · ${scale}×`;
    if($('zoom-label').textContent!==label)$('zoom-label').textContent=label;
    $('overview').textContent=overview?'Zoom in':'Full trajectory';
    $('overview').setAttribute('aria-pressed',String(!overview));
  };
  $('overview').addEventListener('click',()=>scene.toggleOverview());
  $('view').addEventListener('click',()=>scene.reset());
  const theater=root.querySelector('.demo-theater');
  $('fullscreen').addEventListener('click',async()=>{
    try{if(document.fullscreenElement)await document.exitFullscreen();else await theater.requestFullscreen();}catch{}
  });
  document.addEventListener('fullscreenchange',()=>$('fullscreen').setAttribute('aria-label',document.fullscreenElement?'Exit fullscreen':'Enter fullscreen'));
  document.addEventListener('visibilitychange',()=>{if(document.hidden){pause();render();}});
  reduced.addEventListener('change',()=>{if(reduced.matches){pause();render();}});
  root.addEventListener('keydown',e=>{
    if(e.code==='Space'&&!['INPUT','SELECT','BUTTON','SUMMARY','A'].includes(e.target.tagName)){e.preventDefault();$('play').click();}
  });
  compute();
  if(!reduced.matches)start();
  new IntersectionObserver(entries=>{
    if(!entries[0].isIntersecting&&playing){pause();render();}
  },{threshold:.2}).observe(theater);
}
