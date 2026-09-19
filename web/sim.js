/**
 * SpudLenia Continuous Cellular Automata Simulation Engine
 * Real-time 2D multi-channel PDE integration:
 * - A_lipid: Lipid bilayer density (Teal / Cyan)
 * - A_crowd: Membrane protein crowding (Coral / Amber)
 * - A_feed: Ambient nutrient feeder liposomes (Golden Yellow)
 */

class SpudLeniaEngine {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');

    // Simulation grid dimensions
    this.width = 160;
    this.height = 106;
    this.size = this.width * this.height;

    // Field arrays [0, 1]
    this.lipid = new Float32Array(this.size);
    this.crowd = new Float32Array(this.size);
    this.feed = new Float32Array(this.size);

    // Buffers for updates
    this.nextLipid = new Float32Array(this.size);
    this.nextCrowd = new Float32Array(this.size);
    this.nextFeed = new Float32Array(this.size);

    // Convolved potential buffers
    this.uShort = new Float32Array(this.size);
    this.uCrowd = new Float32Array(this.size);

    // Default Biophysical Parameters
    this.params = {
      muLip: 0.15,
      sigmaLip: 0.038,
      dt: 0.15,
      alphaNeck: 1.15,     // Furrow brake parameter (controlled by Fader 1)
      gammaAssim: 0.85,    // Nutrient flux (controlled by Fader 2)
      etaExpr: 0.12,       // Internal protein synthesis
      deltaDecay: 0.04,    // Protein turnover
      betaCurve: 0.22,     // Curvature segregation
      lambdaWaste: 0.02,   // Lipid degradation
      rShort: 3.5,
      rCrowd: 7.0,
      speedMultiplier: 1.0,
    };

    // Precompute discrete convolution kernels
    this.initKernels();

    // Offscreen render buffer for pixel blitting
    this.imgData = this.ctx.createImageData(this.width, this.height);

    // Stats
    this.stepCount = 0;
    this.fps = 60;
    this.lastTime = performance.now();
    this.frameCount = 0;
    this.currentPreset = 'fission';

    // Initialize with default fission configuration
    this.loadPreset('fission');
  }

  // Generate annular kernel weights
  initKernels() {
    this.kShort = this.createAnnularKernel(this.params.rShort, 1.8);
    this.kCrowd = this.createAnnularKernel(this.params.rCrowd, 2.5);
  }

  createAnnularKernel(rBar, sigma) {
    const rad = Math.ceil(rBar + sigma * 2);
    const diam = rad * 2 + 1;
    const kernel = [];
    let sum = 0;

    for (let dy = -rad; dy <= rad; dy++) {
      for (let dx = -rad; dx <= rad; dx++) {
        const r = Math.sqrt(dx * dx + dy * dy);
        const x = (r - rBar) / sigma;
        let w = 0;
        if (Math.abs(x) < 1.0) {
          w = Math.exp(1.0 - 1.0 / (1.0 - x * x));
        }
        kernel.push({ dx, dy, w });
        sum += w;
      }
    }
    // Normalize
    if (sum > 0) {
      for (const k of kernel) k.w /= sum;
    }
    return kernel;
  }

  // Seed protocell soliton at position (cx, cy)
  seedProtocell(cx, cy, rad = 7.0, withCrowding = true) {
    for (let dy = -rad * 2; dy <= rad * 2; dy++) {
      for (let dx = -rad * 2; dx <= rad * 2; dx++) {
        const x = Math.floor(cx + dx);
        const y = Math.floor(cy + dy);
        if (x < 0 || x >= this.width || y < 0 || y >= this.height) continue;
        const idx = y * this.width + x;
        const dist = Math.sqrt(dx * dx + dy * dy);

        // Annular lipid vesicle envelope
        const lipidVal = Math.exp(-Math.pow(dist - rad * 0.7, 2) / (rad * 0.45)) +
                         0.35 * Math.exp(-Math.pow(dist, 2) / (rad * rad));
        this.lipid[idx] = Math.min(1.0, this.lipid[idx] + lipidVal * 0.95);

        // Internal protein crowding core
        if (withCrowding) {
          const crowdVal = 0.85 * Math.exp(-Math.pow(dist, 2) / (rad * 0.6 * rad * 0.6));
          this.crowd[idx] = Math.min(1.0, this.crowd[idx] + crowdVal);
        }
      }
    }
  }

  // Seed ambient nutrient feeder liposomes
  seedNutrients(cx, cy, rad = 14) {
    for (let dy = -rad; dy <= rad; dy++) {
      for (let dx = -rad; dx <= rad; dx++) {
        const x = Math.floor(cx + dx);
        const y = Math.floor(cy + dy);
        if (x < 0 || x >= this.width || y < 0 || y >= this.height) continue;
        const idx = y * this.width + x;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const feedVal = 0.75 * Math.exp(-Math.pow(dist, 2) / (rad * rad * 0.5));
        this.feed[idx] = Math.min(1.0, this.feed[idx] + feedVal);
      }
    }
  }

  // Clear all fields
  clear() {
    this.lipid.fill(0);
    this.crowd.fill(0);
    this.feed.fill(0);
    this.stepCount = 0;
  }

  // Load predefined biophysical configurations
  loadPreset(presetName) {
    this.clear();
    this.currentPreset = presetName;
    const cx = this.width / 2;
    const cy = this.height / 2;

    if (presetName === 'fission') {
      // Component_1: Autonomous cytokinesis via growth-braking
      this.params.alphaNeck = 1.15;
      this.params.gammaAssim = 0.85;
      this.params.etaExpr = 0.14;

      // Seed central protocell with nutrient pool around it
      this.seedProtocell(cx, cy, 7.5, true);
      this.seedNutrients(cx + 16, cy, 18);
      this.seedNutrients(cx - 16, cy, 18);

    } else if (presetName === 'soliton') {
      // Component_2: Motile continuous soliton
      this.params.alphaNeck = 0.0;
      this.params.gammaAssim = 0.5;
      this.params.etaExpr = 0.05;

      // Seed slightly asymmetric vesicle
      this.seedProtocell(cx - 20, cy, 6.5, false);
      // Give it asymmetric nutrient gradient to glide toward
      this.seedNutrients(cx + 30, cy, 25);

    } else if (presetName === 'bloat') {
      // Component_3: Pathological bloating (uncontrolled growth without division)
      this.params.alphaNeck = 0.05; // Furrow brake disabled
      this.params.gammaAssim = 1.8;  // Extreme nutrient flux
      this.params.etaExpr = 0.02;

      this.seedProtocell(cx, cy, 8.0, false);
      // Heavy ambient nutrient bath
      for (let i = 0; i < this.size; i++) {
        this.feed[i] = 0.8;
      }

    } else if (presetName === 'colony') {
      // Component_4: Multi-protocell Darwinian colony
      this.params.alphaNeck = 1.25;
      this.params.gammaAssim = 0.95;
      this.params.etaExpr = 0.12;

      this.seedProtocell(cx - 35, cy - 18, 6.0, true);
      this.seedProtocell(cx + 35, cy - 18, 6.0, true);
      this.seedProtocell(cx, cy + 24, 6.5, true);

      this.seedNutrients(cx, cy - 10, 20);
      this.seedNutrients(cx - 20, cy + 15, 16);
      this.seedNutrients(cx + 20, cy + 15, 16);
    }
  }

  // Gaussian Growth function G(u; mu, sigma) = 2*exp(-(u-mu)^2 / (2*sigma^2)) - 1
  growth(u, mu, sigma) {
    const d = (u - mu) / sigma;
    return 2.0 * Math.exp(-0.5 * d * d) - 1.0;
  }

  // Compute spatial convolution with periodic toroidal boundary conditions
  convolve(src, dst, kernel) {
    const W = this.width;
    const H = this.height;

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let sum = 0;
        for (let i = 0; i < kernel.length; i++) {
          const k = kernel[i];
          const nx = (x + k.dx + W) % W;
          const ny = (y + k.dy + H) % H;
          sum += src[ny * W + nx] * k.w;
        }
        dst[y * W + x] = sum;
      }
    }
  }

  // Compute squared gradient ||grad A_crowd||^2 via central differences
  computeCrowdGradientSq(x, y) {
    const W = this.width;
    const H = this.height;
    const xm1 = (x - 1 + W) % W;
    const xp1 = (x + 1) % W;
    const ym1 = (y - 1 + H) % H;
    const yp1 = (y + 1) % H;

    const gx = (this.crowd[y * W + xp1] - this.crowd[y * W + xm1]) * 0.5;
    const gy = (this.crowd[yp1 * W + x] - this.crowd[ym1 * W + x]) * 0.5;
    return gx * gx + gy * gy;
  }

  // Single PDE integration step
  step() {
    const W = this.width;
    const H = this.height;
    const p = this.params;
    const dt = p.dt * p.speedMultiplier;

    // Step 1: Spatial multi-annular convolutions
    this.convolve(this.lipid, this.uShort, this.kShort);
    this.convolve(this.lipid, this.uCrowd, this.kCrowd);

    let totalLipidMass = 0;

    // Step 2: Coupled field dynamics
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const idx = y * W + x;

        const lip = this.lipid[idx];
        const crw = this.crowd[idx];
        const fed = this.feed[idx];
        const uS = this.uShort[idx];
        const uC = this.uCrowd[idx];

        // 1. Lipid Dynamics F_lipid:
        // Growth potential + assimilation - necking furrow brake
        const gLip = this.growth(uS, p.muLip, p.sigmaLip);
        const gradSq = this.computeCrowdGradientSq(x, y);
        const furrowBrake = p.alphaNeck * gradSq * lip * 12.0;
        const assimilation = p.gammaAssim * fed * lip;
        const dLipid = gLip + assimilation - furrowBrake - p.lambdaWaste * lip;

        // 2. Crowding Dynamics F_crowd:
        // Internal expression - decay + curvature segregation
        const dCrowd = p.etaExpr * lip - p.deltaDecay * crw + p.betaCurve * crw * uC;

        // 3. Feeder Nutrient Dynamics F_feed:
        // Uptake depletion + slight slow environmental diffusion
        const dFeed = -assimilation * 0.7;

        // Euler time integration with clipping to [0, 1]
        let nextL = lip + dt * dLipid;
        let nextC = crw + dt * dCrowd;
        let nextF = fed + dt * dFeed;

        if (nextL < 0) nextL = 0; else if (nextL > 1) nextL = 1;
        if (nextC < 0) nextC = 0; else if (nextC > 1) nextC = 1;
        if (nextF < 0) nextF = 0; else if (nextF > 1) nextF = 1;

        this.nextLipid[idx] = nextL;
        this.nextCrowd[idx] = nextC;
        this.nextFeed[idx] = nextF;

        totalLipidMass += nextL;
      }
    }

    // Swap buffers
    const tempL = this.lipid; this.lipid = this.nextLipid; this.nextLipid = tempL;
    const tempC = this.crowd; this.crowd = this.nextCrowd; this.nextCrowd = tempC;
    const tempF = this.feed;  this.feed = this.nextFeed;   this.nextFeed = tempF;

    this.stepCount++;
    return totalLipidMass / this.size;
  }

  // Trigger cytokinesis expression shock (Vote button)
  triggerExpressionShock() {
    const W = this.width;
    const H = this.height;
    // Boost equatorial protein concentration across active protocells
    for (let i = 0; i < this.size; i++) {
      if (this.lipid[i] > 0.3) {
        this.crowd[i] = Math.min(1.0, this.crowd[i] + 0.45);
      }
    }
  }

  // Render continuous fields to canvas with authentic OLED aesthetic
  render() {
    const W = this.width;
    const H = this.height;
    const data = this.imgData.data;

    for (let i = 0; i < this.size; i++) {
      const l = this.lipid[i];
      const c = this.crowd[i];
      const f = this.feed[i];

      const pIdx = i * 4;

      // Biophysical Color Blending:
      // Background: Deep OLED slate #1b1c1e (27, 28, 30)
      // Lipid: Neon Teal / Cyan #00e5ff (0, 229, 255)
      // Crowd: Vivid Coral / Orange #ff5722 (255, 87, 34)
      // Feed: Amber Glow #ffc107 (255, 193, 7)

      let r = 27 + (l * 10 + c * 230 + f * 190);
      let g = 28 + (l * 215 + c * 70 + f * 150);
      let b = 30 + (l * 225 + c * 35 + f * 10);

      // Bloom on lipid boundaries
      if (l > 0.6) {
        g += 30 * l;
        b += 30 * l;
      }

      data[pIdx]     = Math.min(255, r);
      data[pIdx + 1] = Math.min(255, g);
      data[pIdx + 2] = Math.min(255, b);
      data[pIdx + 3] = 255;
    }

    // Draw to main display canvas with smooth scaling
    createImageBitmap(this.imgData).then(bmp => {
      this.ctx.imageSmoothingEnabled = true;
      this.ctx.imageSmoothingQuality = 'high';
      this.ctx.drawImage(bmp, 0, 0, this.canvas.width, this.canvas.height);
    });
  }
}

// -----------------------------------------------------------------------------
// Interactive UI Controller & Event Binding
// -----------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('simCanvas');
  const engine = new SpudLeniaEngine(canvas);

  // HUD Elements
  const hudPreset = document.getElementById('hudPreset');
  const hudStep   = document.getElementById('hudStep');
  const hudMass   = document.getElementById('hudMass');
  const hudFps    = document.getElementById('hudFps');

  // Component Preset Buttons
  const compButtons = document.querySelectorAll('.comp-item');
  compButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      compButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const preset = btn.getAttribute('data-preset');
      engine.loadPreset(preset);
      const titleSpan = btn.querySelector('.comp-title');
      const affordSpan = btn.querySelector('.comp-affordance');
      hudPreset.textContent = `${titleSpan ? titleSpan.textContent : 'Component'} (${affordSpan ? affordSpan.textContent : preset.toUpperCase()})`;
    });
  });

  // Tactile Save Button: Download high-resolution PNG
  const saveBtn = document.getElementById('saveBtn');
  saveBtn.addEventListener('click', () => {
    const link = document.createElement('a');
    link.download = `cgen_spudlenia_${engine.currentPreset}_step${engine.stepCount}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  });

  // Quality-Diversity / MAP-Elites In-Memory Specimen Archive
  const specimens = [];
  const voteBtn = document.getElementById('voteBtn');
  const voteCountBadge = document.getElementById('voteCountBadge');
  const archiveBadge = document.getElementById('archiveBadge');
  const archiveModal = document.getElementById('archiveModal');
  const archiveClose = document.getElementById('archiveClose');
  const archiveEmpty = document.getElementById('archiveEmpty');
  const specimenGrid = document.getElementById('specimenGrid');

  function renderArchive() {
    if (specimens.length === 0) {
      archiveEmpty.style.display = 'block';
      specimenGrid.innerHTML = '';
      return;
    }
    archiveEmpty.style.display = 'none';
    specimenGrid.innerHTML = '';

    specimens.forEach((spec, idx) => {
      const card = document.createElement('div');
      card.className = 'specimen-card';
      card.innerHTML = `
        <img class="specimen-thumb" src="${spec.thumb}" alt="Specimen ${idx + 1}">
        <div class="specimen-meta">
          <span class="specimen-name">#${idx + 1} · ${spec.preset.toUpperCase()}</span>
          <span class="specimen-stats">M(t): ${spec.mass.toFixed(3)} | α: ${spec.params.alphaNeck.toFixed(2)}</span>
          <button class="specimen-restore-btn">Recall Genome</button>
        </div>
      `;
      card.querySelector('.specimen-restore-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        restoreSpecimen(spec);
      });
      card.addEventListener('click', () => {
        restoreSpecimen(spec);
      });
      specimenGrid.appendChild(card);
    });
  }

  function restoreSpecimen(spec) {
    Object.assign(engine.params, spec.params);
    engine.loadPreset(spec.preset);
    hudPreset.textContent = `RECALLED #${spec.id} (${spec.preset.toUpperCase()})`;
    archiveModal.classList.remove('open');
  }

  // Vote Button: Curate Specimen into MAP-Elites Archive + Trigger Expression Shock
  voteBtn.addEventListener('click', () => {
    // 1. Biological kinetic shock feedback
    engine.triggerExpressionShock();
    voteBtn.style.transform = 'translateY(2px)';
    setTimeout(() => { voteBtn.style.transform = ''; }, 120);

    // 2. Curate state into archive
    const thumbUrl = canvas.toDataURL('image/png');
    const mass = engine.step();
    const newSpecimen = {
      id: specimens.length + 1,
      preset: engine.currentPreset,
      step: engine.stepCount,
      mass: mass,
      params: { ...engine.params },
      thumb: thumbUrl,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    };
    specimens.unshift(newSpecimen);

    // Update Badges
    voteCountBadge.textContent = specimens.length;
    archiveBadge.textContent = specimens.length;
    renderArchive();
  });

  // Module Switcher Tabs
  const moduleSwitcher = document.getElementById('moduleSwitcher');
  const moduleTabs = moduleSwitcher.querySelectorAll('.module-tab');
  moduleTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      moduleTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const mode = tab.getAttribute('data-mode');
      if (mode === 'archive') {
        renderArchive();
        archiveModal.classList.add('open');
      } else if (mode === 'compute') {
        // Reservoir perturbation mode: seed frequency waves
        engine.triggerExpressionShock();
        hudPreset.textContent = 'MODE: RESERVOIR COMPUTATION';
      } else {
        hudPreset.textContent = `Component_1 (FISSION)`;
      }
    });
  });

  archiveClose.addEventListener('click', () => archiveModal.classList.remove('open'));
  archiveModal.addEventListener('click', (e) => {
    if (e.target === archiveModal) archiveModal.classList.remove('open');
  });

  // Speed / Rate Step Controls (+) and (-)
  const zoomInBtn  = document.getElementById('zoomInBtn');
  const zoomOutBtn = document.getElementById('zoomOutBtn');
  const rateIndicator = document.getElementById('rateIndicator');

  function updateRate(delta) {
    engine.params.speedMultiplier = Math.max(0.25, Math.min(3.0, engine.params.speedMultiplier + delta));
    rateIndicator.textContent = `${engine.params.speedMultiplier.toFixed(1)}×`;
  }

  zoomInBtn.addEventListener('click', () => updateRate(0.25));
  zoomOutBtn.addEventListener('click', () => updateRate(-0.25));

  // Vertical Capacitive Fader 1: alpha_neck (Furrow Growth Brake)
  setupFader('faderTrack1', 'faderPill1', 'faderTip1', (val) => {
    engine.params.alphaNeck = val * 2.5;
    return `Furrow Brake α: ${engine.params.alphaNeck.toFixed(2)}`;
  }, 0.50);

  // Vertical Capacitive Fader 2: gamma_assim (Nutrient Assimilation Flux)
  setupFader('faderTrack2', 'faderPill2', 'faderTip2', (val) => {
    engine.params.gammaAssim = val * 2.0;
    return `Feeder Flux γ: ${engine.params.gammaAssim.toFixed(2)}`;
  }, 0.42);

  function setupFader(trackId, pillId, tipId, onChange, initialRatio = 0.5) {
    const track = document.getElementById(trackId);
    const pill  = document.getElementById(pillId);
    const tip   = document.getElementById(tipId);

    function updateFromY(clientY) {
      const rect = track.getBoundingClientRect();
      const pillHeight = pill.offsetHeight;
      const maxY = rect.height - pillHeight;
      let y = clientY - rect.top - pillHeight / 2;
      y = Math.max(0, Math.min(maxY, y));

      pill.style.top = `${y}px`;
      const ratio = 1.0 - (y / maxY); // Inverted: top is maximum value
      const label = onChange(ratio);
      tip.textContent = label;
    }

    // Set initial position
    setTimeout(() => {
      const rect = track.getBoundingClientRect();
      const pillHeight = pill.offsetHeight;
      const maxY = rect.height - pillHeight;
      const y = (1.0 - initialRatio) * maxY;
      pill.style.top = `${y}px`;
      tip.textContent = onChange(initialRatio);
    }, 50);

    let isDragging = false;
    track.addEventListener('mousedown', (e) => {
      isDragging = true;
      track.classList.add('dragging');
      updateFromY(e.clientY);
    });

    window.addEventListener('mousemove', (e) => {
      if (isDragging) updateFromY(e.clientY);
    });

    window.addEventListener('mouseup', () => {
      if (isDragging) {
        isDragging = false;
        track.classList.remove('dragging');
      }
    });
  }

  // Dynamic Contextual Canvas Action Hint on Keydown
  const canvasHint = document.getElementById('canvasHint');
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Shift') {
      canvasHint.innerHTML = `<span class="hint-glyph" style="color: #fbc02d;">●</span> <span class="hint-text">SHIFT ACTIVE: Painting Nutrient Feeder Gradient</span>`;
    }
  });
  window.addEventListener('keyup', (e) => {
    if (e.key === 'Shift') {
      canvasHint.innerHTML = `<span class="hint-glyph">✛</span> <span class="hint-text">CLICK: Seed Soliton &nbsp;·&nbsp; SHIFT+DRAG: Feed Nutrients</span>`;
    }
  });

  // Canvas Drawing Interaction: Click/drag to seed protocells or nutrients
  let isInteracting = false;
  canvas.addEventListener('mousedown', (e) => {
    isInteracting = true;
    handleCanvasInteraction(e);
  });
  window.addEventListener('mousemove', (e) => {
    if (isInteracting) handleCanvasInteraction(e);
  });
  window.addEventListener('mouseup', () => { isInteracting = false; });

  function handleCanvasInteraction(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = engine.width / rect.width;
    const scaleY = engine.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    if (e.shiftKey || e.button === 2) {
      engine.seedNutrients(x, y, 12);
    } else {
      engine.seedProtocell(x, y, 6.0, true);
    }
  }

  // Keyboard Shortcuts for Rapid Prototyping & Affordance
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key === '1') compButtons[0].click();
    if (e.key === '2') compButtons[1].click();
    if (e.key === '3') compButtons[2].click();
    if (e.key === '4') compButtons[3].click();
    if (e.key === 'v' || e.key === 'V') voteBtn.click();
    if (e.key === 's' || e.key === 'S') saveBtn.click();
  });

  // About Modal Handlers
  const aboutBtn   = document.getElementById('aboutBtn');
  const aboutModal = document.getElementById('aboutModal');
  const modalClose = document.getElementById('modalClose');
  aboutBtn.addEventListener('click', () => aboutModal.classList.add('open'));
  modalClose.addEventListener('click', () => aboutModal.classList.remove('open'));
  aboutModal.addEventListener('click', (e) => {
    if (e.target === aboutModal) aboutModal.classList.remove('open');
  });

  // Main 60 FPS Simulation & Animation Loop
  let frameCount = 0;
  let lastFpsUpdate = performance.now();

  function animate(now) {
    const mass = engine.step();
    engine.render();

    // Update Telemetry
    hudStep.textContent = engine.stepCount;
    hudMass.textContent = mass.toFixed(3);

    frameCount++;
    if (now - lastFpsUpdate >= 500) {
      const fps = Math.round((frameCount * 1000) / (now - lastFpsUpdate));
      hudFps.textContent = fps;
      frameCount = 0;
      lastFpsUpdate = now;
    }

    requestAnimationFrame(animate);
  }

  requestAnimationFrame(animate);
});
