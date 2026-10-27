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

const preview = document.querySelector('[data-home-demo]');
if (preview) initializePreview(preview);

async function initializePreview(root) {
  const canvas = root.querySelector('#home-demo-scene');
  const toggle = root.querySelector('#home-demo-toggle');
  const status = root.querySelector('#home-demo-status');
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  let scene, comparison, events, sampleComparisonStory;
  let stageLabel = 'Demo preview';
  let cursor = 0, endHold = 0, frame = 0, lastTime = 0;
  let playing = false, inView = false, userPaused = false, allowMotion = false;
  let available = false, pageActive = true;

  function updateControl() {
    root.dataset.playing = String(playing);
    const label = playing ? 'Pause demo preview' : 'Play demo preview';
    toggle.setAttribute('aria-label', label);
    toggle.title = label;
    toggle.firstElementChild.textContent = playing ? 'Ⅱ' : '▶';
    status.textContent = stageLabel;
  }

  function stop() {
    cancelAnimationFrame(frame);
    frame = 0;
    lastTime = 0;
    endHold = 0;
    playing = false;
    root.dataset.resetting = 'false';
    updateControl();
  }

  function draw(dt = 16) {
    const model = sampleComparisonStory(comparison, events, cursor);
    model.animateCamera = playing;
    scene.update(comparison.baseline.problem, comparison.baseline.result, model, dt);
    root.dataset.storyPhase = model.storyPhase;
    root.dataset.scope = model.scope;
    root.dataset.part = model.part;
    root.dataset.cursor = cursor.toFixed(3);
    root.dataset.iteration = model.compareCursor.toFixed(3);
    stageLabel = model.storyPhase === 'trajectory' ? 'Convergence' :
      model.scope === 'dual' || model.stage === 1 ? 'Dual update' : 'Primal · inner solve';
    status.textContent = stageLabel;
  }

  function fallback() {
    available = false;
    stageLabel = 'Demo preview';
    stop();
    root.dataset.ready = 'false';
    root.dataset.resetting = 'false';
    toggle.hidden = true;
    scene?.renderer?.dispose();
    scene?.renderer?.domElement.remove();
    scene?.observer?.disconnect();
  }

  function tick(now) {
    if (!playing) return;
    frame = requestAnimationFrame(tick);
    // A 30 fps preview keeps the two real trajectories smooth without running
    // the full renderer at the display refresh rate throughout the homepage.
    if (lastTime && now - lastTime < 1000 / 30) return;
    const dt = lastTime ? Math.min(100, now - lastTime) : 1000 / 30;
    lastTime = now;
    if (cursor >= events.length) {
      endHold += dt;
      if (endHold > 1300) root.dataset.resetting = 'true';
      if (endHold <= 1650) return;
      cursor = 0;
      endHold = 0;
      root.dataset.resetting = 'false';
    } else {
      let remaining = dt;
      while (remaining > 0 && cursor < events.length) {
        const index = Math.floor(cursor);
        const amount = Math.min(remaining / events[index].duration, 1 - (cursor - index));
        cursor += amount;
        remaining -= amount * events[index].duration;
        if (amount < 1e-9) break;
      }
    }
    try { draw(dt); } catch { fallback(); }
  }

  function synchronizePlayback() {
    const shouldPlay = available && inView && pageActive && !document.hidden &&
      !userPaused && (!motion.matches || allowMotion);
    if (!shouldPlay) {
      stop();
      return;
    }
    if (playing) return;
    playing = true;
    lastTime = 0;
    updateControl();
    frame = requestAnimationFrame(tick);
  }

  try {
    const [sceneModule, comparisonModule] = await Promise.all([
      import('./demo-saddle-scene.mjs'),
      import('./demo-saddle-comparison.mjs')
    ]);
    sampleComparisonStory = comparisonModule.sampleComparisonStory;
    comparison = comparisonModule.createComparison();
    events = comparisonModule.buildComparisonStory(comparison);
    scene = new sceneModule.SaddleScene(canvas, {compact: true});
    cursor = motion.matches ? events.length : 0;
    draw();
    available = true;
    root.dataset.ready = 'true';
    toggle.hidden = false;
    scene.renderer.domElement.addEventListener('webglcontextlost', fallback);
  } catch {
    fallback();
    return;
  }

  toggle.addEventListener('click', () => {
    if (playing) {
      userPaused = true;
    } else {
      userPaused = false;
      allowMotion = true;
      if (cursor >= events.length) {
        cursor = 0;
        endHold = 0;
        root.dataset.resetting = 'false';
      }
    }
    synchronizePlayback();
  });
  motion.addEventListener('change', () => {
    if (!available) return;
    allowMotion = false;
    if (motion.matches) {
      cursor = events.length;
      endHold = 0;
      root.dataset.resetting = 'false';
      draw();
    }
    synchronizePlayback();
  });
  document.addEventListener('visibilitychange', synchronizePlayback);
  window.addEventListener('pagehide', () => { pageActive = false; synchronizePlayback(); });
  window.addEventListener('pageshow', () => { pageActive = true; synchronizePlayback(); });
  new IntersectionObserver(entries => {
    inView = entries[0].isIntersecting && entries[0].intersectionRatio >= 0.15;
    synchronizePlayback();
  }, {threshold: [0, 0.15]}).observe(root);
  updateControl();
}
