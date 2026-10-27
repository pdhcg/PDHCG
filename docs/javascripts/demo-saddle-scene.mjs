import * as T from './vendor/three.mjs';
import {lagrangian, localPrimal, localDual, solveSaddle} from './demo-saddle-solver.mjs';

const mix=(a,b,t)=>a.map((v,i)=>v+(b[i]-v)*t);
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));

export class SaddleScene {
  constructor(canvas,{compact=false}={}){
    this.canvas=canvas;this.ctx=canvas.getContext('2d');this.model=null;this.problem=null;
    this.compact=compact;
    this.yaw=-1.75;this.pitch=1.06;this.zoom=1;this.overview=true;this.viewWindow=null;
    this.renderer=new T.WebGLRenderer({alpha:true,antialias:true,powerPreference:'low-power'});
    this.renderer.setClearColor(0,0);this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    this.renderer.domElement.className='demo-webgl';
    this.renderer.domElement.setAttribute('aria-hidden','true');
    Object.assign(this.renderer.domElement.style,{position:'absolute',inset:'0',width:'100%',height:'100%',pointerEvents:'none',zIndex:'0'});
    canvas.parentElement.insertBefore(this.renderer.domElement,canvas);
    Object.assign(canvas.style,{position:'relative',zIndex:'1'});
    this.scene=new T.Scene();this.camera=new T.OrthographicCamera(-4,4,3,-3,.1,100);this.camera.up.set(0,0,1);
    this.scene.add(new T.AmbientLight(0xffffff,1.5));
    const light=new T.DirectionalLight(0xffffff,1.4);light.position.set(-3,-4,7);this.scene.add(light);
    const fill=new T.DirectionalLight(0xb7d9ed,.8);fill.position.set(5,3,3);this.scene.add(fill);
    this.observer=new ResizeObserver(()=>this.draw());this.observer.observe(canvas);
    new MutationObserver(()=>{this.setTheme();this.draw();}).observe(document.body,{attributes:true,attributeFilter:['data-md-color-scheme']});
    let drag=null;
    canvas.addEventListener('pointerdown',e=>{if(['inner','dual'].includes(this.model?.scope))return;drag={id:e.pointerId,x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);canvas.focus({preventScroll:true});});
    canvas.addEventListener('pointermove',e=>{if(drag?.id!==e.pointerId)return;this.yaw-=(e.clientX-drag.x)*.007;this.pitch=clamp(this.pitch+(e.clientY-drag.y)*.005,.23,1.3);drag.x=e.clientX;drag.y=e.clientY;this.viewWindow=null;this.draw();});
    const end=e=>{if(drag?.id===e.pointerId){drag=null;if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);}};
    canvas.addEventListener('pointerup',end);canvas.addEventListener('pointercancel',end);
    canvas.addEventListener('wheel',e=>{if(['inner','dual'].includes(this.model?.scope))return;if(!e.ctrlKey&&!e.metaKey)return;e.preventDefault();this.zoom=clamp(this.zoom*Math.exp(-e.deltaY*.002),.7,1.6);this.viewWindow=null;this.draw();},{passive:false});
    canvas.addEventListener('keydown',e=>{
      if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','r','R'].includes(e.key))return;
      if(['inner','dual'].includes(this.model?.scope)&&e.key.toLowerCase()!=='r')return;
      e.preventDefault();e.stopPropagation();
      if(e.key==='ArrowLeft')this.yaw+=.12;if(e.key==='ArrowRight')this.yaw-=.12;
      if(e.key==='ArrowUp')this.pitch=clamp(this.pitch+.1,.23,1.3);if(e.key==='ArrowDown')this.pitch=clamp(this.pitch-.1,.23,1.3);
      if(e.key.toLowerCase()==='r')this.reset();else{this.viewWindow=null;this.draw();}
    });
    this.setTheme();
  }
  reset(){this.yaw=-1.75;this.pitch=1.06;this.zoom=1;this.overview=true;this.viewWindow=null;this.draw();}
  toggleOverview(){this.overview=!this.overview;this.viewWindow=null;this.draw();}
  setTheme(){
    const dark=document.body.dataset.mdColorScheme==='slate';this.dark=dark;
    this.colors={ink:dark?'#dceaf4':'#173b55',orange:dark?'#ffb67c':'#c4793c',teal:dark?'#5bd5cb':'#007d76',halpern:dark?'#c1afff':'#7460bd',paper:dark?'#0d2030':'#fbfcfd'};
    if(this.surface){this.surface.material.color.set(dark?'#276d7e':'#6aadb7');this.surface.material.opacity=dark?.61:.55;}
  }
  setup(p,result,comparisonResult){
    this.problem=p;this.result=result;this.scaleZ=.24/Math.max(1,p.q);
    this.comparisonResult=comparisonResult;
    const star=p.saddle;
    // Both Halpern settings share the unaccelerated view scale.
    const baseline=p.halpern?solveSaddle({...p,halpern:false}):result;
    const all=[p.initial,star,...baseline.records.map(r=>[r.xp,r.yp]),
      ...(comparisonResult?.records??[]).map(r=>[r.xp,r.yp])];
    const low=[0,1].map(i=>Math.min(...all.map(v=>v[i]))),high=[0,1].map(i=>Math.max(...all.map(v=>v[i])));
    this.outerBounds={low,high};
    this.center=[(low[0]+high[0])/2,(low[1]+high[1])/2,lagrangian(p,...star)*this.scaleZ];
    const patchMargin=p.constraint==='equality'?.58:.8;
    this.rx=Math.max(.4,(high[0]-low[0])*patchMargin);
    this.ry=Math.max(.55,(high[1]-low[1])*patchMargin);
    // An elliptical patch avoids a boxed chart. It samples the exact Lagrangian.
    const positions=[],indices=[],rings=52,sectors=120;
    for(let j=0;j<=rings;j++)for(let i=0;i<=sectors;i++){
      const t=i/sectors*Math.PI*2,r=j/rings;
      const x=this.center[0]+this.rx*r*Math.cos(t),y=this.center[1]+this.ry*r*Math.sin(t);
      positions.push(x,y,lagrangian(p,x,y)*this.scaleZ);
    }
    for(let j=0;j<rings;j++)for(let i=0;i<sectors;i++){
      const a=j*(sectors+1)+i,b=a+sectors+1;indices.push(a,b,a+1,b,b+1,a+1);
    }
    const geometry=new T.BufferGeometry();geometry.setAttribute('position',new T.Float32BufferAttribute(positions,3));geometry.setIndex(indices);geometry.computeVertexNormals();
    if(this.surface){this.scene.remove(this.surface);this.surface.geometry.dispose();this.surface.material.dispose();}
    this.surface=new T.Mesh(geometry,new T.MeshPhongMaterial({color:'#6aadb7',specular:'#173b42',shininess:8,side:T.DoubleSide,transparent:true,opacity:.55,depthWrite:false}));
    this.scene.add(this.surface);this.setTheme();this.viewWindow=null;this.overview=true;
    // Fit the trajectory itself, not the decorative surface or its floor.
    this.fitPoints=all.map(xy=>new T.Vector3(...xy,lagrangian(p,...xy)*this.scaleZ));
  }
  update(p,result,model,dt=16){
    const comparisonResult=model.comparisonBounds??model.comparison?.result;
    if(this.problem!==p||this.comparisonResult!==comparisonResult)this.setup(p,result,comparisonResult);
    const previous=this.model;
    this.cameraAction=!previous||(!model.animateCamera&&
      (previous.cursor!==model.cursor||previous.scope!==model.scope))?'snap':
      model.animateCamera?'animate':'hold';
    this.model=model;this.frameDt=dt;this.draw();
  }
  draw(){
    if(!this.model||!this.surface)return;
    const p=this.problem,m=this.model,r=m.r,C=this.colors;
    const W=this.canvas.clientWidth,H=this.canvas.clientHeight;if(!W||!H)return;
    const ctx=this.ctx,dpr=Math.min(devicePixelRatio,2);
    const resized=this.canvas.width!==Math.round(W*dpr)||this.canvas.height!==Math.round(H*dpr);
    if(resized){this.canvas.width=Math.round(W*dpr);this.canvas.height=Math.round(H*dpr);this.renderer.setSize(W,H,false);}
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);
    const focus=this.center;
    const inner=clamp(m.innerBlend||0,0,1),dual=clamp(m.dualBlend||0,0,1),close=Math.max(inner,dual),outer=1-close;

    this.camera.position.set(focus[0]+10*Math.cos(this.yaw)*Math.cos(this.pitch),focus[1]+10*Math.sin(this.yaw)*Math.cos(this.pitch),focus[2]+10*Math.sin(this.pitch));
    this.camera.lookAt(...focus);this.camera.updateMatrixWorld();
    const projectedBounds=vertices=>{
      let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
      for(const vertex of vertices){const q=vertex.clone().applyMatrix4(this.camera.matrixWorldInverse);
        xmin=Math.min(xmin,q.x);xmax=Math.max(xmax,q.x);ymin=Math.min(ymin,q.y);ymax=Math.max(ymax,q.y);}
      return {xmin,xmax,ymin,ymax};
    };
    const compact=this.compact&&W<500;
    const top=compact?42:62,bottom=compact?45:66,side=compact?20:W<500?24:48;
    const frame=b=>{
      const dx=Math.max(1e-9,b.xmax-b.xmin),dy=Math.max(1e-9,b.ymax-b.ymin);
      const unit=Math.min((W-2*side)/dx,(H-top-bottom)/dy);
      return {unit,cx:(b.xmin+b.xmax)/2,cy:(b.ymin+b.ymax)/2+(top-bottom)/(2*unit)};
    };
    const fullBounds=projectedBounds(this.fitPoints);
    const marginX=(fullBounds.xmax-fullBounds.xmin)*.1||.1,marginY=(fullBounds.ymax-fullBounds.ymin)*.1||.1;
    const globalFrame=frame({xmin:fullBounds.xmin-marginX,xmax:fullBounds.xmax+marginX,ymin:fullBounds.ymin-marginY,ymax:fullBounds.ymax+marginY});
    const recent=this.result.records.slice(Math.max(0,m.iteration-2),m.iteration+1);
    const localPoints=[m.point,p.saddle,...recent.map(v=>[v.xp,v.yp])];
    if(m.comparison){
      localPoints.push(m.comparison.point,...m.comparison.result.records.slice(Math.max(0,m.comparison.completed-2),m.comparison.completed).map(v=>[v.xp,v.yp]));
    }
    if(m.mode==='detail')localPoints.push([r.x,r.y],[r.xp,r.y]);
    const lo=[0,1].map(i=>Math.min(...localPoints.map(v=>v[i]))),hi=[0,1].map(i=>Math.max(...localPoints.map(v=>v[i])));
    const localSpan=Math.max(1e-8,hi[0]-lo[0],hi[1]-lo[1]);
    this.localBounds={lo,hi};
    const vertices=localPoints.flatMap(xy=>[
      new T.Vector3(...xy,lagrangian(p,...xy)*this.scaleZ),
      new T.Vector3(...xy,lagrangian(p,r.xp,r.yp)*this.scaleZ)]);
    const bounds=projectedBounds(vertices),cx=(bounds.xmin+bounds.xmax)/2,cy=(bounds.ymin+bounds.ymax)/2;
    const dx=Math.max(bounds.xmax-bounds.xmin,localSpan*.25)*1.5,dy=Math.max(bounds.ymax-bounds.ymin,localSpan*.25)*1.5;
    const localFrame=frame({xmin:cx-dx/2,xmax:cx+dx/2,ymin:cy-dy/2,ymax:cy+dy/2});
    const target=this.overview?globalFrame:localFrame;
    const targetWindow={...target,unit:Math.max(globalFrame.unit,Math.min(globalFrame.unit*1e7,target.unit))*this.zoom};
    if(!this.viewWindow||this.cameraAction==='snap'||resized)this.viewWindow={...targetWindow};
    else if(this.cameraAction==='animate'){
      const move=1-Math.exp(-Math.min(this.frameDt||16,75)/90),scale=1-Math.exp(-Math.min(this.frameDt||16,75)/240);
      this.viewWindow.cx+=(targetWindow.cx-this.viewWindow.cx)*move;
      this.viewWindow.cy+=(targetWindow.cy-this.viewWindow.cy)*move;
      this.viewWindow.unit=Math.exp(Math.log(this.viewWindow.unit)+(Math.log(targetWindow.unit)-Math.log(this.viewWindow.unit))*scale);
    }
    this.cameraAction='hold';
    const {unit,cx:viewX,cy:viewY}=this.viewWindow;
    this.magnification=unit/globalFrame.unit;
    this.camera.left=viewX-W/(2*unit);this.camera.right=viewX+W/(2*unit);
    this.camera.top=viewY+H/(2*unit);this.camera.bottom=viewY-H/(2*unit);this.camera.updateProjectionMatrix();
    this.surface.material.opacity=(this.dark?.61:.55)*outer*outer/Math.max(1,Math.log10(this.magnification+1));
    this.renderer.render(this.scene,this.camera);
    const P3=v=>{const q=new T.Vector3(...v).project(this.camera);return[(q.x+1)*W/2,(1-q.y)*H/2];};
    // Keep the same subproblem curve while bringing its fixed-y slice forward.
    // Independent plot scales make a small, late subproblem readable as well.
    const innerXs=[r.x,r.xp,...r.innerSteps.flatMap(s=>[s.from,s.trial,s.to])];
    const span=Math.max(.05,Math.max(...innerXs)-Math.min(...innerXs));
    const ilow=Math.min(...innerXs)-span*.3,ihigh=Math.max(...innerXs)+span*.3;
    const isamples=Array.from({length:80},(_,i)=>ilow+(ihigh-ilow)*i/79);
    const izs=isamples.flatMap(x=>[localPrimal(p,r,x),lagrangian(p,x,r.y)]);
    const zlow=Math.min(...izs),zhigh=Math.max(...izs),zrange=Math.max(1e-8,zhigh-zlow);
    // The dual close-up maximizes the true proximal objective at fixed x-bar.
    // It has its own y-axis and scale; its pair still returns to (x-plus, y-plus).
    const dualRadius=Math.max(.05,Math.abs(r.y-r.trialY),Math.abs(r.yp-r.trialY));
    const dlow=r.trialY-dualRadius*1.2,dhigh=r.trialY+dualRadius*1.2;
    const dsamples=Array.from({length:81},(_,i)=>dlow+(dhigh-dlow)*i/80);
    const dzs=dsamples.map(y=>localDual(p,r,y)),dzlow=Math.min(...dzs),dzhigh=localDual(p,r,r.trialY),dzrange=Math.max(1e-9,dzhigh-dzlow);
    const iw=this.compact?W-56:Math.min(W-100,660),ih=H-(this.compact?116:146),left=(W-iw)/2,bottomPlot=H-(this.compact?52:62);
    const P=v=>{
      if(v[3]==='dual'){
        const q=P3([v[0],v[1],lagrangian(p,v[0],v[1])*this.scaleZ]);
        const flat=[left+(v[1]-dlow)/(dhigh-dlow)*iw,bottomPlot-(v[2]/this.scaleZ-dzlow)/dzrange*ih];
        return [q[0]*(1-dual)+flat[0]*dual,q[1]*(1-dual)+flat[1]*dual];
      }
      const q=P3(v),flat=[left+(v[0]-ilow)/(ihigh-ilow)*iw,
        bottomPlot-(v[2]/this.scaleZ-zlow)/zrange*ih];
      return [q[0]*(1-inner)+flat[0]*inner,q[1]*(1-inner)+flat[1]*inner];
    };
    const lifted=(xy,lift=0)=>[...xy,(lagrangian(p,...xy)+lift*(xy[0]-r.x)**2/(2*p.tau))*this.scaleZ];
    const onDual=y=>[r.xp,y,localDual(p,r,y)*this.scaleZ,'dual'];
    const displayPoint=xy=>m.scope==='dual'?onDual(xy[1]):lifted(xy,m.lift);
    const path=(points,color,width=1,alpha=1,dash=[])=>{
      if(points.length<2)return;ctx.beginPath();points.forEach((v,i)=>{const q=P(v);i?ctx.lineTo(...q):ctx.moveTo(...q);});ctx.strokeStyle=color;ctx.lineWidth=width;ctx.lineCap='round';ctx.lineJoin='round';ctx.globalAlpha=alpha;ctx.setLineDash(dash);ctx.stroke();ctx.globalAlpha=1;ctx.setLineDash([]);
    };
    const line=(a,b,fn,n=70)=>Array.from({length:n+1},(_,i)=>fn(a+(b-a)*i/n));
    const onSaddle=xy=>[...xy,lagrangian(p,...xy)*this.scaleZ];
    const visible=q=>q[0]>8&&q[0]<W-8&&q[1]>top&&q[1]<H-bottom+6;

    const labels=[];
    const point=(v,color,label='',radius=5,alpha=1)=>{
      const q=P(v);if(!visible(q))return;ctx.globalAlpha=alpha;
      const glow=ctx.createRadialGradient(q[0],q[1],0,q[0],q[1],radius*3.3);glow.addColorStop(0,color+'38');glow.addColorStop(1,color+'00');ctx.fillStyle=glow;ctx.beginPath();ctx.arc(...q,radius*3.3,0,2*Math.PI);ctx.fill();
      ctx.beginPath();ctx.arc(...q,radius,0,2*Math.PI);ctx.fillStyle=color;ctx.strokeStyle=C.paper;ctx.lineWidth=2;ctx.fill();ctx.stroke();ctx.globalAlpha=1;
      if(label)labels.push({p:q,text:label,color,dx:12,dy:-13});
    };
    const textAt=(v,text,color,dx=10,dy=0)=>{const q=P(v);if(visible(q))labels.push({p:q,text,color,dx,dy});};
    if(outer>.01){
      // The full view stays fixed so the complete convergence path remains visible.
      if(this.magnification<2){
        const {low,high}=this.outerBounds;
        path(line(low[0],high[0],x=>onSaddle([x,p.saddle[1]])),C.orange,1.2,.4*outer);
        path(line(low[1],high[1],y=>onSaddle([p.saddle[0],y])),C.teal,1.2,.4*outer);
        if(close<.4){
          textAt(onSaddle([high[0],p.saddle[1]]),'primal x',C.orange,12,16);
          textAt(onSaddle([p.saddle[0],p.saddle[1]+.82*(high[1]-p.saddle[1])]),'dual y',C.teal,9,-14);
        }
        for(const f of [-.6,0,.6]){
          const x=this.center[0]+f*this.rx,yr=this.ry*Math.sqrt(1-f*f);
          path(line(this.center[1]-yr,this.center[1]+yr,y=>onSaddle([x,y])),C.ink,.8,.075*outer);
        }
      }else{
        const {lo,hi}=this.localBounds,margin=Math.max(hi[0]-lo[0],hi[1]-lo[1])*.15;
        const xa=onSaddle([lo[0]-margin,p.saddle[1]]),xb=onSaddle([hi[0]+margin,p.saddle[1]]);
        const ya=onSaddle([p.saddle[0],lo[1]-margin]),yb=onSaddle([p.saddle[0],hi[1]+margin]);
        path([xa,xb],C.orange,1,.18*outer);path([ya,yb],C.teal,1,.18*outer);
        textAt(xb,'x',C.orange,6,13);textAt(yb,'y',C.teal,5,-10);
      }
      // Keep the introductory primal/dual elbow in the accumulated path.
      const candidates=[p.initial];
      if((!m.comparison||m.showFirstUpdate)&&(m.completed>0||m.part==='dual'||m.scope==='dual'))candidates.push([this.result.records[0].xp,p.initial[1]]);
      candidates.push(...this.result.records.slice(0,m.completed).map(v=>[v.xp,v.yp]));
      const trail=candidates.flatMap((v,i)=>i?Array.from({length:8},(_,j)=>onSaddle(mix(candidates[i-1],v,(j+1)/8))):[onSaddle(v)]);
      path(trail,C.paper,5.5,.6*outer);
      path(trail,C.teal,2.8,.78*outer);
      // As a close-up folds back, retain the completed coordinate update on
      // the outer surface rather than adding its trail only at the next event.
      if(m.scope==='inner')path(line(r.x,m.point[0],x=>onSaddle([x,r.y])),C.orange,2.8,.78*outer);
      if(m.scope==='dual')path(line(r.y,m.point[1],y=>onSaddle([r.xp,y])),C.teal,2.8,.78*outer);
      // Sparse markers leave the winding path readable even after many turns.
      for(const [i,xy] of candidates.slice(2).entries())if(i<20||i%8===0)point(onSaddle(xy),C.teal,'',1.8,.5*outer);
      if(m.comparison){
        const other=m.comparison;
        const firstUpdate=m.showFirstUpdate&&other.completed>0?[[other.result.records[0].xp,other.problem.initial[1]]]:[];
        const points=[other.problem.initial,...firstUpdate,...other.result.records.slice(0,other.completed).map(v=>[v.xp,v.yp]),other.point];
        const overlay=points.flatMap((v,i)=>i?Array.from({length:8},(_,j)=>onSaddle(mix(points[i-1],v,(j+1)/8))):[onSaddle(v)]);
        path(overlay,C.paper,5.5,.65);path(overlay,C.halpern,3,.95);
        point(onSaddle(other.point),C.halpern,'',6);
      }
      point(onSaddle(p.initial),C.ink,close<.4?'start':'',4.5,.8*outer);
      point(onSaddle(p.saddle),C.ink,close<.4?'solution':'',4.5,.85*outer);
    }
    if(m.scope==='inner'){
      const curve=line(ilow,ihigh,x=>[x,r.y,localPrimal(p,r,x)*this.scaleZ]);
      const slice=line(ilow,ihigh,x=>onSaddle([x,r.y]));
      const opacity=Math.max(inner,m.localOpacity||0);
      path(slice,C.teal,1.5,.45*opacity);
      // The box is a constraint of this primal subproblem; the separate row
      // constraint remains the responsibility of the outer dual update.
      if(inner>.01){
        const low=Math.max(ilow,p.lower),high=Math.min(ihigh,p.upper);
        const rect=[[low,r.y,zlow*this.scaleZ],[high,r.y,zlow*this.scaleZ],
          [high,r.y,zhigh*this.scaleZ],[low,r.y,zhigh*this.scaleZ]].map(P);
        ctx.beginPath();rect.forEach((q,i)=>i?ctx.lineTo(...q):ctx.moveTo(...q));ctx.closePath();
        ctx.fillStyle=C.orange;ctx.globalAlpha=.045*inner;ctx.fill();ctx.globalAlpha=1;
        for(const bound of [p.lower,p.upper])if(bound>ilow&&bound<ihigh){
          path([[bound,r.y,zlow*this.scaleZ],[bound,r.y,zhigh*this.scaleZ]],C.orange,1.1,.4*inner,[4,5]);
          if(inner>.7)textAt([bound,r.y,zlow*this.scaleZ],bound===p.upper?'upper bound':'lower bound',C.orange,8,18);
        }
      }
      path(curve,C.orange,2.5,opacity);
      if(inner>.5){
        textAt(curve[Math.floor(curve.length*.84)],'primal subproblem',C.orange,-100,-18);
        textAt(slice[Math.floor(slice.length*.18)],'L(x, yₖ)',C.teal,-8,19);
      }
      for(const step of r.innerSteps.slice(0,m.innerStep||0))point([step.to,r.y,localPrimal(p,r,step.to)*this.scaleZ],C.orange,'',2.5,.35*inner);
      if(m.part==='project'){
        const trial=[m.from[0],r.y,localPrimal(p,r,m.from[0])*this.scaleZ];
        const accepted=[m.to[0],r.y,localPrimal(p,r,m.to[0])*this.scaleZ];
        path([trial,accepted],C.orange,1.5,.45,[4,5]);point(trial,C.orange,inner>.6?'primal step':'',4,.5);
      }
    }
    if(m.scope==='dual'){
      const curve=dsamples.map(onDual);
      if(p.constraint==='inequality'&&dlow<0&&dhigh>0){
        const rect=[[r.xp,0,dzlow*this.scaleZ,'dual'],[r.xp,dhigh,dzlow*this.scaleZ,'dual'],
          [r.xp,dhigh,dzhigh*this.scaleZ,'dual'],[r.xp,0,dzhigh*this.scaleZ,'dual']].map(P);
        ctx.beginPath();rect.forEach((q,i)=>i?ctx.lineTo(...q):ctx.moveTo(...q));ctx.closePath();
        ctx.fillStyle=C.teal;ctx.globalAlpha=.045*dual;ctx.fill();ctx.globalAlpha=1;
        path([[r.xp,0,dzlow*this.scaleZ,'dual'],[r.xp,0,dzhigh*this.scaleZ,'dual']],C.teal,1.1,.5*dual,[4,5]);
        if(dual>.7)textAt([r.xp,0,dzlow*this.scaleZ,'dual'],'y ≥ 0',C.teal,8,-10);
      }
      path(curve,C.teal,2.5,dual);
      if(dual>.5){
        textAt(curve[Math.floor(curve.length*.2)],'dual subproblem',C.teal,-75,-18);
        textAt(curve.at(-1),'dual y',C.teal,-38,-12);
        point(onDual(r.y),C.ink,m.part==='enter'||m.part==='ascent'?'yₖ':'',3.5,.45*dual);
      }
      if(m.part==='project'&&Math.abs(r.trialY-r.yp)>1e-10){
        const trial=onDual(r.trialY),accepted=onDual(r.yp);
        path([trial,accepted],C.teal,1.5,.5,[4,5]);point(trial,C.teal,dual>.6?'dual step':'',4,.5);
      }
    }
    if(m.done&&m.scope==='outer')point(onSaddle(m.point),C.teal,'',7);
    else{
      const color=m.scope==='inner'||m.stage===0?C.orange:C.teal;
      const visited=m.motionPath||[m.from,m.point];
      const motion=visited.flatMap((v,i)=>i?Array.from({length:30},(_,j)=>
        displayPoint(mix(visited[i-1],v,(j+1)/30))):[displayPoint(v)]);
      path(motion,C.paper,5.5,.7);path(motion,color,3,.95);
      const returned=['inner','dual'].includes(m.scope)&&['settle','exit'].includes(m.part);
      point(displayPoint(m.point),color,returned?(m.scope==='dual'?'y⁺ → Outer':'x⁺ → Dual'):'',7);
    }
    // The two current coordinates remain visible, so dual movement is explicit.
    const pad=W<500?18:28,font=W<500?12:14;
    ctx.font=`500 ${font}px system-ui, sans-serif`;ctx.textAlign='left';
    const digits=this.magnification>10000?7:this.magnification>100?5:3;
    const fmt=v=>v.toFixed(digits);
    const baseline=H-44;
    const fixedX=m.scope==='dual'||m.scope!=='inner'&&m.part==='dual',fixedY=m.scope==='inner'||m.stage===0&&m.part!=='restart';
    if(!m.comparison&&!this.compact){
      ctx.fillStyle=C.orange;ctx.fillText(`${m.scope==='inner'?'inner u':m.scope==='dual'?'x̄':'x'}  ${fmt(m.scope==='dual'?r.xbar:m.point[0])}${fixedX?'  · fixed':''}`,pad,baseline-23);
      ctx.fillStyle=C.teal;ctx.fillText(`${m.scope==='inner'?'yₖ':'y'}  ${fmt(m.point[1])}${fixedY?'  · fixed':''}`,pad,baseline);
    }
    // Only a few labels, positioned outside one another as the scene rotates.
    const boxes=[];ctx.font=`500 ${W<500?11:13}px system-ui, sans-serif`;
    const semantic=l=>['start','solution'].includes(l.text);
    labels.sort((a,b)=>Number(semantic(b))-Number(semantic(a)));
    for(const l of labels){
      const width=ctx.measureText(l.text).width;
      const offsets=[[l.dx,l.dy],[-width/2,-22],[-width/2,27],[-width-14,4],[14,4],[-width/2,-42],[-width/2,47]];
      const candidates=offsets.map(([dx,dy])=>({x:clamp(l.p[0]+dx,12,W-width-12),y:clamp(l.p[1]+dy,top+13,H-bottom+1)}));
      const chosen=candidates.find(q=>!boxes.some(b=>q.x<b.x+b.w+8&&q.x+width+8>b.x&&Math.abs(q.y-b.y)<21))||candidates[0];
      const {x,y}=chosen;
      const end=[clamp(l.p[0],x,x+width),clamp(l.p[1],y-11,y+2)];
      if(semantic(l)&&Math.hypot(end[0]-l.p[0],end[1]-l.p[1])>13){
        ctx.beginPath();ctx.moveTo(...l.p);ctx.lineTo(...end);ctx.strokeStyle=l.color;ctx.globalAlpha=.5;ctx.lineWidth=.9;ctx.stroke();ctx.globalAlpha=1;
      }
      boxes.push({x,y,w:width});ctx.lineWidth=4;ctx.strokeStyle=C.paper;ctx.strokeText(l.text,x,y);ctx.fillStyle=l.color;ctx.fillText(l.text,x,y);
    }
    this.canvas.dataset.camera=`${this.yaw.toFixed(3)},${this.pitch.toFixed(3)},${this.zoom.toFixed(3)}`;
    this.canvas.dataset.point=m.point.join(',');this.canvas.dataset.part=m.part;this.canvas.dataset.inner=inner.toFixed(3);
    this.canvas.dataset.dual=dual.toFixed(3);
    this.canvas.dataset.zoom=this.magnification.toFixed(2);this.canvas.dataset.screenPoint=P(displayPoint(m.point)).join(',');
    this.canvas.dataset.trailCount=String(m.completed);
    this.canvas.dataset.halpernPoint=m.comparison?m.comparison.point.join(','):'';
    this.canvas.dataset.halpernTrailCount=m.comparison?String(m.comparison.completed):'';
    this.canvas.dataset.solution=P(onSaddle(p.saddle)).join(',');
    this.canvas.dataset.start=P(onSaddle(p.initial)).join(',');
    this.onViewChange?.({magnification:this.magnification,overview:this.overview});
  }
}
