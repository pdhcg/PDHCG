import {makeSaddleProblem, solveSaddle} from './demo-saddle-solver.mjs';
import {buildTimeline, buildInnerTimeline, buildDualTimeline, sampleTimeline} from './demo-saddle-timeline.mjs';
import {SaddleScene} from './demo-saddle-scene.mjs';
import {createComparison, buildComparisonStory, sampleComparisonStory} from './demo-saddle-comparison.mjs';

const root=document.getElementById('pdhcg-demo');
if(root)initialize();

function initialize(){
  const $=id=>document.getElementById(`demo-${id}`);
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const presets={spiral:{},elliptic:{tau:.3,sigma:.04/.3},
    boundary:{q:1,a:1.8,b:.5,tau:.6,sigma:.8,lower:-1.2,upper:.54,initial:[-.8,0],constraint:'inequality'}};
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
    const done=state.storyDone;
    root.dataset.level='outer';root.dataset.mode='trajectory';root.dataset.operation='trajectory';
    root.dataset.transfer='';root.dataset.innerReady='false';root.dataset.innerPart='';root.dataset.innerCursor='';root.dataset.phase='2';
    root.dataset.status=done?'converged':playing?'playing':'paused';
    root.dataset.iteration=String(Math.floor(state.compareCursor));root.dataset.compareCursor=state.compareCursor.toFixed(4);
    $('play').textContent=playing?'Ⅱ Pause':done?'↻ Replay':'▶ Play';
    $('step').disabled=done;$('timeline').value=String(compareCursor);
    $('trajectory-title').textContent='Convergence trajectories';
    $('trajectory-iteration').textContent=`Iteration ${Math.floor(state.compareCursor)} · same starting point`;
    $('plain-status').textContent=state.done?`Converged · ${comparison.baseline.result.records.length}`:'';
    $('halpern-status').textContent=state.comparison.done?`Converged · ${comparison.halpern.result.records.length}`:'';
    $('scene').setAttribute('aria-label','PDHG convergence on the same saddle: teal without Halpern, purple with Halpern. Both advance at the same iteration count. Drag or use arrow keys to rotate. R resets the view.');
    state.animateCamera=playing||stepTarget!==null;
    scene.update(comparison.baseline.problem,comparison.baseline.result,state,dt);
  }
  function render(dt=16){
    if(!outerEvents)return;
    const state=activeState(),primal=state.scope==='inner',dual=state.scope==='dual',inside=primal||dual;
    root.dataset.storyPhase=view==='compare'&&!innerEvents?state.storyPhase:'detail';
    root.dataset.cursor=mainCursor().toFixed(4);root.dataset.part=state.part;
    root.dataset.compareCursor=state.compareCursor===undefined?'':state.compareCursor.toFixed(4);
    if(view==='compare'&&state.storyPhase==='trajectory'){renderComparison(dt,state);return;}
    root.dataset.level=inside?state.scope:'outer';
    root.dataset.mode=inside?'detail':state.mode;
    root.dataset.operation=state.done&&!inside?'complete':primal||state.stage===0?'primal':dual||state.stage===1?'dual':'trajectory';
    root.dataset.transfer=inside?(['settle','exit'].includes(state.part)?'return':state.part==='enter'?'enter':'working'):'';
    const ready=inside&&['settle','exit'].includes(state.part)&&Math.abs(state.point[dual?1:0]-(dual?state.r.yp:state.r.xp))<1e-7;
    root.dataset.innerReady=String(ready);
    $('inspect').setAttribute('aria-expanded',String(primal));
    $('inspect-dual').setAttribute('aria-expanded',String(dual));
    $('inspect').disabled=dual;$('inspect-dual').disabled=primal;
    if(inside){
      const inputs=dual?`<span>x̄ = ${state.r.xbar.toFixed(3)}</span><span>yₖ = ${state.r.y.toFixed(3)}</span>`:
        `<span>xₖ = ${state.r.x.toFixed(3)}</span><span>yₖ = ${state.r.y.toFixed(3)}</span>`;
      if($('inner-input').innerHTML!==inputs)$('inner-input').innerHTML=inputs;
      const output=ready?`${dual?'y':'x'}⁺ = ${(dual?state.r.yp:state.r.xp).toFixed(3)}`:`Computing ${dual?'y':'x'}⁺…`;
      if($('inner-output').textContent!==output)$('inner-output').textContent=output;
      $('subproblem-label').textContent=dual?'Dual subproblem':'Primal subproblem';
      $('subproblem-module').textContent=dual?'Dual update':'Inner solve';
      $('substep-first').textContent=dual?'Ascent step':'Gradient step';
      $('substep-second').textContent=dual&&p.constraint==='equality'?'Free dual':'Project';
    }
    root.dataset.phase=String(state.stage);root.dataset.iteration=String(state.iteration);root.dataset.part=state.part;
    root.dataset.innerCursor=inside?(innerEvents?innerCursor:mainCursor()-mainEvents().findIndex(e=>e.scope===state.scope)).toFixed(4):'';
    root.dataset.status=state.done&&!inside?result.status:playing?'playing':'paused';
    root.dataset.halpern=String(activeProblem().halpern);
    $('play').textContent=playing?'Ⅱ Pause':state.done&&!inside?'↻ Replay':'▶ Play';
    $('step').disabled=state.done&&!inside;
    $('timeline').value=String(mainCursor());
    $('iteration').textContent=`Iteration ${state.iteration+1}`;
    $('trajectory-title').textContent='Convergence trajectory';
    $('trajectory-iteration').textContent=`${p.halpern?'PDHG + Halpern':'PDHG'} · ${state.iteration+1} / ${result.records.length}`;
    $('inner-count').textContent=primal&&state.innerStep!==undefined?`${state.innerStep+1}/${state.r.innerSteps.length}`:'';
    let title=state.title,caption=state.caption;
    if(primal){
      if(['settle','exit'].includes(state.part)){
        title='Return x⁺. The outer iteration continues.';
        caption='The inner solver supplies x⁺ to the primal step. Next comes the dual update.';
      }else{
        title='The outer loop waits for this primal solve.';
        caption='Keep xₖ and yₖ fixed. The inner solver computes only x⁺.';
      }
    }
    if(lastText!==title+caption){
      lastText=title+caption;$('stage-title').textContent=title;$('stage-caption').textContent=caption;
      $('scene').setAttribute('aria-label',`${primal?'Inner primal subproblem':dual?'Dual subproblem':'Outer primal–dual iteration'}. ${title}. ${caption} Drag or use arrow keys to rotate the outer view. R resets the view.`);
    }
    root.querySelectorAll('[data-phase]').forEach(b=>{b.disabled=inside;b.setAttribute('aria-pressed',String(!inside&&state.mode==='detail'&&b.dataset.phase===(state.stage===1?'dual':'pdhg')));});
    root.dataset.innerPart=inside?state.part:'';
    state.animateCamera=playing||stepTarget!==null;
    if(view==='compare')state.comparisonBounds=comparison.halpern.result;
    scene.update(activeProblem(),activeResult(),state,dt);
  }
  function resumeAfter(scope){
    if(scope==='inner')return mainEvents().findIndex(e=>e.scope==='outer'&&e.part==='dual');
    const index=mainEvents().findIndex(e=>e.mode==='trajectory');
    return index<0?mainEvents().length:index;
  }
  function enterInner(scope='inner'){
    const index=mainEvents().findIndex(e=>e.scope==='outer'&&e.part===(scope==='dual'?'dual':'primal'));
    if(index<0)return;
    const before=activeState().point;
    setMainCursor(index);resumeCursor=resumeAfter(scope);
    innerEvents=(scope==='dual'?buildDualTimeline:buildInnerTimeline)(activeProblem(),mainEvents()[index].r);innerCursor=0;
    const entry=innerEvents[0];
    if(!reduced.matches&&Math.hypot(...before.map((v,i)=>v-entry.from[i]))>1e-8){
      innerEvents.unshift({...entry,part:'revisit',from:[...before],to:[...entry.from],duration:450,
        localTo:0,liftTo:0,blendTo:0,dualTo:0,detail:`Revisit this ${scope==='dual'?'dual':'primal'} update`});
      const total=innerEvents.reduce((sum,e)=>sum+e.duration,0);let elapsed=0;
      for(const e of innerEvents){e.phaseOffset=elapsed;e.phaseDuration=total;elapsed+=e.duration;}
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
      p=makeSaddleProblem({...presets[$('example').value],halpern:$('halpern').checked});result=solveSaddle(p);outerEvents=buildTimeline(p,result);
      innerEvents=null;outerCursor=0;lastText='';
      $('timeline').max=String(mainEvents().length);$('timeline').step='0.001';
      const coefficient=p.q===1?'½':Number((p.q/2).toPrecision(5));
      const equality=p.constraint==='equality',relation=equality?'=':'≤';
      $('problem-formula').innerHTML=`<span class="demo-min">min<sub>${p.lower} ≤ x ≤ ${p.upper}</sub></span><span>${coefficient}(x − ${p.a})²</span><span class="demo-subject">subject to</span><span>x ${relation} ${p.b}</span>`;
      $('problem-formula').setAttribute('aria-label',`Minimize ${p.q/2} times x minus ${p.a} squared, with x in [${p.lower}, ${p.upper}], subject to x ${equality?'equal to':'less than or equal to'} ${p.b}`);
      $('alert').hidden=result.status!=='numerical-error';
      if(result.status==='numerical-error')$('alert').textContent='The numerical reference stopped unexpectedly. Choose another example.';
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
    if(!reduced.matches)start();else{innerCursor=1;render();}
  }
  $('inspect').addEventListener('click',()=>inspect('inner'));
  $('inspect-dual').addEventListener('click',()=>inspect('dual'));
  function returnOuter(){
    const state=activeState();
    if(!['inner','dual'].includes(state.scope))return;
    const dual=state.scope==='dual';
    pause();
    if(!innerEvents){
      resumeCursor=resumeAfter(state.scope);
      innerEvents=(dual?buildDualTimeline:buildInnerTimeline)(activeProblem(),state.r);
    }
    if(reduced.matches){finishInner();render();return;}
    // Finish this solve and reverse the same spatial transition, even if the
    // reader returns halfway through a trial or a projection.
    const exit=innerEvents.at(-1),settle={...exit,part:'settle',from:[...state.point],to:[state.r.xp,dual?state.r.yp:state.r.y],
      duration:500,localFrom:state.localOpacity,localTo:dual?0:1,liftFrom:state.lift,liftTo:dual?0:1,
      blendFrom:state.innerBlend,blendTo:dual?0:1,dualFrom:state.dualBlend||0,dualTo:dual?1:0,
      phaseOffset:0,phaseDuration:500+exit.duration};
    innerEvents=[settle,{...exit,phaseOffset:500,phaseDuration:500+exit.duration}];innerEvents.scope=state.scope;innerEvents.status='solved';innerCursor=0;start();
  }
  $('return').addEventListener('click',returnOuter);$('outer').addEventListener('click',returnOuter);
  root.querySelectorAll('[data-phase]').forEach(b=>b.addEventListener('click',()=>{
    pause();const part=b.dataset.phase==='dual'?'dual':'primal';
    const index=mainEvents().findIndex(e=>e.part===part);
    innerEvents=null;setMainCursor(Math.max(0,index));render();
  }));
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
