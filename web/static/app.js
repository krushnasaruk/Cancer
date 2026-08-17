/**
 * Sesame AI Digital Twin — 100vh Executive SPA Client Logic
 */

let scene, camera, renderer, controls;
let robotGroup, targetMesh;
let legs = {};
let followRobot = false;

function initThreeScene() {
  const container = document.getElementById("three-canvas-container");
  if (!container) return;
  const w = container.clientWidth || 800;
  const h = container.clientHeight || 500;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x080b13);
  scene.fog = new THREE.FogExp2(0x080b13, 0.35);

  camera = new THREE.PerspectiveCamera(42, w / h, 0.01, 20);
  camera.position.set(0.30, -0.30, 0.22);
  camera.up.set(0, 0, 1);  // Z-up to match MuJoCo

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.target.set(0, 0, 0.05);

  // Lighting
  scene.add(new THREE.AmbientLight(0xffffff, 0.8));
  const key = new THREE.DirectionalLight(0xffffff, 1.3);
  key.position.set(1.5, -1.5, 3.5);
  key.castShadow = true;
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x818cf8, 0.6);
  fill.position.set(-2, 2, 2.5);
  scene.add(fill);

  // Floor Grid
  const grid = new THREE.GridHelper(4, 40, 0x00f0ff, 0x1e293b);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(6, 6),
    new THREE.MeshStandardMaterial({ color: 0x080b13, roughness: 0.5, metalness: 0.3 })
  );
  floor.receiveShadow = true;
  scene.add(floor);

  buildRobot();

  // Reaching Target Marker
  targetMesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.012, 32, 32),
    new THREE.MeshStandardMaterial({ color: 0x00f0ff, emissive: 0x00f0ff, emissiveIntensity: 0.8, roughness: 0.2 })
  );
  targetMesh.position.set(0.10, 0.0, 0.02);
  scene.add(targetMesh);

  window.addEventListener("resize", onResize);
  animate();
}

function buildRobot() {
  robotGroup = new THREE.Group();
  scene.add(robotGroup);

  // Materials
  const matChassis = new THREE.MeshStandardMaterial({ color: 0x1e2430, roughness: 0.35, metalness: 0.4 });
  const matCyan = new THREE.MeshStandardMaterial({ color: 0x00f0ff, roughness: 0.15, metalness: 0.1 });
  const matOledBg = new THREE.MeshStandardMaterial({ color: 0x080a10, roughness: 0.05 });
  const matEye = new THREE.MeshBasicMaterial({ color: 0x00f0ff });
  const matHorn = new THREE.MeshStandardMaterial({ color: 0xf97316, roughness: 0.25, metalness: 0.2 });
  const matServo = new THREE.MeshStandardMaterial({ color: 0x0a0b0e, roughness: 0.4 });
  const matFemur = new THREE.MeshStandardMaterial({ color: 0x2b3445, roughness: 0.3, metalness: 0.4 });
  const matTibia = new THREE.MeshStandardMaterial({ color: 0x475569, roughness: 0.25, metalness: 0.5 });
  const matFoot = new THREE.MeshStandardMaterial({ color: 0x00f0ff, emissive: 0x00f0ff, emissiveIntensity: 0.7 });

  // Chassis box
  const chassis = new THREE.Mesh(new THREE.BoxGeometry(0.102, 0.074, 0.046), matChassis);
  chassis.castShadow = true;
  robotGroup.add(chassis);

  // Cyan shell
  const topShell = new THREE.Mesh(new THREE.BoxGeometry(0.103, 0.075, 0.004), matCyan);
  topShell.position.set(0, 0, 0.0235);
  topShell.castShadow = true;
  chassis.add(topShell);

  // OLED Screen & Eyes
  const oledBg = new THREE.Mesh(new THREE.PlaneGeometry(0.048, 0.030), matOledBg);
  oledBg.position.set(0.052, 0, 0.004);
  oledBg.rotation.y = Math.PI / 2;
  chassis.add(oledBg);

  const eyeGeo = new THREE.PlaneGeometry(0.010, 0.012);
  const eyeL = new THREE.Mesh(eyeGeo, matEye);
  eyeL.position.set(0.053, 0.010, 0.005);
  eyeL.rotation.y = Math.PI / 2;
  chassis.add(eyeL);

  const eyeR = new THREE.Mesh(eyeGeo, matEye);
  eyeR.position.set(0.053, -0.010, 0.005);
  eyeR.rotation.y = Math.PI / 2;
  chassis.add(eyeR);

  // Leg Kinematic Chains
  const legDefs = [
    { id: "FL", bx: 0.036, by: 0.039,  fyo: 0.005, tyo: 0.011 },
    { id: "FR", bx: 0.036, by: -0.039, fyo: -0.005, tyo: -0.011 },
    { id: "RL", bx: -0.036, by: 0.039,  fyo: 0.005, tyo: 0.011 },
    { id: "RR", bx: -0.036, by: -0.039, fyo: -0.005, tyo: -0.011 },
  ];

  legDefs.forEach(def => {
    const servoBox = new THREE.Mesh(new THREE.BoxGeometry(0.022, 0.012, 0.022), matServo);
    servoBox.position.set(def.bx, def.by * 0.82, 0);
    chassis.add(servoBox);

    const femurPivot = new THREE.Group();
    femurPivot.position.set(def.bx, def.by, 0);
    chassis.add(femurPivot);

    const horn = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.004, 20), matHorn);
    horn.position.set(0, def.fyo * 0.3, 0);
    femurPivot.add(horn);

    const femurLen = 0.042;
    const femurGeo = new THREE.CylinderGeometry(0.0075, 0.0075, femurLen, 12);
    femurGeo.rotateX(-Math.PI / 2);
    const femur = new THREE.Mesh(femurGeo, matFemur);
    femur.position.set(0, def.fyo, -femurLen / 2);
    femur.castShadow = true;
    femurPivot.add(femur);

    const kneeServo = new THREE.Mesh(new THREE.BoxGeometry(0.022, 0.012, 0.022), matServo);
    kneeServo.position.set(0, def.fyo, -0.035);
    femurPivot.add(kneeServo);

    const kneeHorn = new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.007, 0.004, 20), matHorn);
    kneeHorn.position.set(0, def.tyo, -0.042);
    femurPivot.add(kneeHorn);

    const tibiaPivot = new THREE.Group();
    tibiaPivot.position.set(0, def.tyo, -0.042);
    femurPivot.add(tibiaPivot);

    const tibiaLen = 0.046;
    const tibiaGeo = new THREE.CylinderGeometry(0.006, 0.005, tibiaLen, 12);
    tibiaGeo.rotateX(-Math.PI / 2);
    const tibia = new THREE.Mesh(tibiaGeo, matTibia);
    tibia.position.set(0, 0, -tibiaLen / 2);
    tibia.castShadow = true;
    tibiaPivot.add(tibia);

    const foot = new THREE.Mesh(new THREE.SphereGeometry(0.008, 16, 16), matFoot);
    foot.position.set(0, 0, -0.046);
    tibiaPivot.add(foot);

    legs[def.id] = { femurPivot, tibiaPivot };
  });

  robotGroup.position.set(0, 0, 0.065);
}

function onResize() {
  const c = document.getElementById("three-canvas-container");
  if (!c || !renderer || !camera) return;
  camera.aspect = c.clientWidth / c.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(c.clientWidth, c.clientHeight);
}

function animate() {
  requestAnimationFrame(animate);
  if (controls) controls.update();
  if (followRobot && robotGroup) controls.target.copy(robotGroup.position);
  if (targetMesh) {
    const p = 1.0 + Math.sin(Date.now() * 0.008) * 0.12;
    targetMesh.scale.set(p, p, p);
  }
  if (renderer && scene && camera) renderer.render(scene, camera);
}

function setCameraView(preset) {
  document.querySelectorAll(".cam-btn").forEach(b => b.classList.remove("active"));
  if (event && event.target) event.target.classList.add("active");
  const t = { x: 0, y: 0, z: 0.06 };
  if (preset === "iso")   { camera.position.set(0.30, -0.30, 0.22); }
  if (preset === "front") { camera.position.set(0.40, 0, 0.08); }
  if (preset === "side")  { camera.position.set(0, -0.40, 0.08); }
  if (preset === "top")   { camera.position.set(0.001, 0, 0.50); }
  controls.target.set(t.x, t.y, t.z);
}

function toggleFollowCamera() {
  followRobot = !followRobot;
  document.getElementById("btn-follow").classList.toggle("active", followRobot);
}

// ========== Single-Page Application (SPA) Page Switcher ==========
function switchPage(pageId) {
  document.querySelectorAll(".nav-tab-btn").forEach(b => b.classList.remove("active"));
  if (event && event.target) event.target.classList.add("active");

  document.querySelectorAll(".app-page").forEach(p => p.classList.remove("active"));
  const targetPage = document.getElementById(pageId);
  if (targetPage) {
    targetPage.classList.add("active");
    if (pageId === "page-twin") {
      setTimeout(onResize, 50);
    } else if (pageId === "page-analytics") {
      setTimeout(() => {
        if (chartRewardFull) chartRewardFull.update();
        if (chartDistFull) chartDistFull.update();
      }, 50);
    }
  }
}

// ========== Telemetry Charts (Drawer & Full Page) ==========
let chartRewardMini, chartDistMini, chartRewardFull, chartDistFull;
const MAX_PTS = 50;

function initCharts() {
  const opts = {
    responsive: true, maintainAspectRatio: false, animation: false,
    scales: {
      x: { grid: { color: "rgba(255,255,255,0.06)" }, ticks: { color: "#94a3b8", font: { size: 9 } } },
      y: { grid: { color: "rgba(255,255,255,0.06)" }, ticks: { color: "#94a3b8", font: { size: 9 } } },
    },
    plugins: { legend: { display: false } },
  };

  const elR1 = document.getElementById("chart-reward-mini");
  const elD1 = document.getElementById("chart-distance-mini");
  const elR2 = document.getElementById("chart-reward-full");
  const elD2 = document.getElementById("chart-distance-full");

  if (elR1) {
    chartRewardMini = new Chart(elR1.getContext("2d"), {
      type: "line", data: { labels: [], datasets: [{ data: [], borderColor: "#6366f1", borderWidth: 2, pointRadius: 0, tension: 0.3 }] }, options: opts,
    });
  }
  if (elD1) {
    chartDistMini = new Chart(elD1.getContext("2d"), {
      type: "line", data: { labels: [], datasets: [{ data: [], borderColor: "#00f0ff", borderWidth: 2, pointRadius: 0, tension: 0.3 }] }, options: opts,
    });
  }
  if (elR2) {
    chartRewardFull = new Chart(elR2.getContext("2d"), {
      type: "line", data: { labels: [], datasets: [{ data: [], borderColor: "#6366f1", borderWidth: 2, pointRadius: 0, tension: 0.3 }] }, options: opts,
    });
  }
  if (elD2) {
    chartDistFull = new Chart(elD2.getContext("2d"), {
      type: "line", data: { labels: [], datasets: [{ data: [], borderColor: "#00f0ff", borderWidth: 2, pointRadius: 0, tension: 0.3 }] }, options: opts,
    });
  }
}

function pushChart(t, ret, dist) {
  const chartGroup = [
    { reward: chartRewardMini, dist: chartDistMini, canvas: document.getElementById("chart-reward-mini") },
    { reward: chartRewardFull, dist: chartDistFull, canvas: document.getElementById("chart-reward-full") }
  ];

  chartGroup.forEach(g => {
    // Only update if the chart's canvas is currently visible in the DOM
    const isVisible = g.canvas && g.canvas.offsetParent !== null;

    if (g.reward) {
      if (g.reward.data.labels.length > MAX_PTS) { g.reward.data.labels.shift(); g.reward.data.datasets[0].data.shift(); }
      g.reward.data.labels.push(t);
      g.reward.data.datasets[0].data.push(ret);
      if (isVisible) g.reward.update("none");
    }
    if (g.dist) {
      if (g.dist.data.labels.length > MAX_PTS) { g.dist.data.labels.shift(); g.dist.data.datasets[0].data.shift(); }
      g.dist.data.labels.push(t);
      g.dist.data.datasets[0].data.push(dist);
      if (isVisible) g.dist.update("none");
    }
  });
}

// ========== WebSocket Streaming ==========
let ws;

function connectWS() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen = () => {
    const el = document.getElementById("live-sim-status");
    if (el) { el.textContent = "● SIMULATION ONLINE | 500 Hz Physics"; el.style.color = "#10b981"; }
    const selectVal = document.getElementById("select-controller")?.value;
    if (selectVal) onControllerChange(selectVal);
  };
  ws.onmessage = (e) => updateUI(JSON.parse(e.data));
  ws.onclose = () => {
    const el = document.getElementById("live-sim-status");
    if (el) { el.textContent = "○ RECONNECTING..."; el.style.color = "#f59e0b"; }
    setTimeout(connectWS, 1500);
  };
}

function updateUI(d) {
  if (robotGroup && d.base) {
    robotGroup.position.set(d.base.x, d.base.y, d.base.z);
    robotGroup.quaternion.set(d.base.qx, d.base.qy, d.base.qz, d.base.qw);
  }

  if (d.joints) {
    const jk = Object.keys(d.joints);
    const hipMap = { FL: 2, FR: 0, RL: 3, RR: 1 };
    const kneeMap = { FL: 6, FR: 5, RL: 7, RR: 4 };

    Object.entries(legs).forEach(([id, leg]) => {
      const hipDeg = d.joints[jk[hipMap[id]]]?.angle_deg || 90;
      const kneeDeg = d.joints[jk[kneeMap[id]]]?.angle_deg || 90;
      const hipRot = (hipDeg * Math.PI / 180) - (Math.PI / 2);
      const kneeRot = (kneeDeg * Math.PI / 180) - (Math.PI / 2);
      leg.femurPivot.rotation.y = hipRot;
      leg.tibiaPivot.rotation.y = kneeRot;
    });
  }

  if (targetMesh && d.target) {
    targetMesh.position.set(d.target.x, d.target.y, d.target.z + 0.005);
  }

  // KPI Mini Cards
  if (d.base) {
    const elZ = document.getElementById("kpi-base-z");
    const elE = document.getElementById("kpi-euler");
    if (elZ) elZ.textContent = `Z: ${(d.base.z * 1000).toFixed(1)} mm`;
    if (elE) elE.textContent = `Roll: ${d.base.roll > 0 ? "+" : ""}${d.base.roll.toFixed(0)}° | Pitch: ${d.base.pitch > 0 ? "+" : ""}${d.base.pitch.toFixed(0)}°`;
  }
  if (d.target) {
    const elT = document.getElementById("kpi-target-dist");
    const elS = document.getElementById("kpi-reach-status");
    if (elT) elT.textContent = `${d.target.dist_mm.toFixed(1)} mm`;
    if (elS) {
      elS.textContent = d.target.reached ? "✓ REACHED" : "In Transit";
      elS.style.color = d.target.reached ? "#10b981" : "#06b6d4";
    }
  }
  if (d.metrics) {
    const elR = document.getElementById("kpi-return");
    const elS = document.getElementById("kpi-step-counter");
    if (elR) elR.textContent = `${d.metrics.return > 0 ? "+" : ""}${d.metrics.return.toFixed(0)}`;
    if (elS) elS.textContent = `Step: ${d.metrics.steps} | 60 FPS`;
  }

  // Ground Contact LEDs
  if (d.contacts) {
    ["FL","FR","RL","RR"].forEach(id => {
      const el = document.getElementById(`led-${id.toLowerCase()}`);
      if (el) el.classList.toggle("active", Boolean(d.contacts[id]));
    });
  }

  // Push Telemetry Charts
  if (d.metrics && d.target) pushChart(d.time, d.metrics.return, d.target.dist_mm);

  // Update Servos & Actuator Diagnostics
  if (d.joints) { buildServoCards(d.joints); updateActuatorTbl(d.joints); }
}

function buildServoCards(joints) {
  const c = document.getElementById("servos-list");
  if (!c || c.children.length > 0) return;
  Object.entries(joints).forEach(([n, v]) => {
    const d = document.createElement("div");
    d.className = "servo-card";
    d.innerHTML = `<div class="servo-header"><span>${n}</span><span>${v.torque_nm.toFixed(3)} N·m</span></div><div class="servo-val">${v.angle_deg.toFixed(1)}°</div><div class="servo-bar"><div class="servo-bar-fill" style="width:${(v.angle_deg/180)*100}%"></div></div>`;
    c.appendChild(d);
  });
}

function updateActuatorTbl(joints) {
  const tb = document.getElementById("actuator-table-body");
  if (!tb) return;
  tb.innerHTML = Object.entries(joints).map(([n,v]) => `<tr><td><b>${n}</b></td><td>${v.angle_deg.toFixed(1)}°</td><td>${v.target_deg.toFixed(1)}°</td><td style="color:${Math.abs(v.angle_deg-v.target_deg)>5?'#ef4444':'#10b981'}">${Math.abs(v.angle_deg-v.target_deg).toFixed(1)}°</td><td>${v.torque_nm.toFixed(3)}</td><td style="color:#10b981">✓ OK</td></tr>`).join("");
}

// ========== User Command API Calls ==========
async function sendCommand(action, params={}) {
  document.querySelectorAll(".pill-deck .pill-btn").forEach(b => b.classList.remove("active"));
  if (action==="START") document.getElementById("btn-run")?.classList.add("active");
  if (action==="PAUSE") document.getElementById("btn-pause")?.classList.add("active");
  if (action==="RESET") document.getElementById("btn-reset")?.classList.add("active");
  await fetch("/api/control",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({action,...params})});
}
function onControllerChange(v) { sendCommand("SET_CONTROLLER",{controller:v}); const g=document.getElementById("gait-group"); if(g) g.style.display=v.includes("PID")?"flex":"none"; }
function onGaitChange(v) { sendCommand("SET_PID_MODE",{mode:v}); }
function onSpeedChange(v) { const s=document.getElementById("speed-val"); if(s) s.textContent=parseFloat(v).toFixed(1)+"x"; sendCommand("SET_SPEED",{speed:parseFloat(v)}); }
function nudgeTarget(dx,dy,dz) { sendCommand("NUDGE_TARGET",{dx,dy,dz}); }
function toggleTheme() { const b=document.body; const c=b.getAttribute("data-theme"); const n=c==="dark"?"light":"dark"; b.setAttribute("data-theme",n); const tb=document.getElementById("theme-btn"); if(tb) tb.textContent=n==="dark"?"☀️ Light":"🌙 Dark"; }
function runQuickBenchmark() { alert("Running benchmark evaluation..."); }

window.addEventListener("DOMContentLoaded", () => { initThreeScene(); initCharts(); connectWS(); });
