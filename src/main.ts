import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

import {
  ScryfallError,
  accentFor,
  backUrl,
  cardByName,
  cardByPrinting,
  parseHash,
  randomCard,
  stepNumber,
  type Card,
} from './data/scryfall';
import type { TransitionMode } from './scene/morph';
import { SculptView } from './scene/sculpt';
import { VoxelCloud } from './scene/voxels';
import { sculptSubject } from './sculpt/mesh';
import { cutout, depth, hasCutout, hasDepth, onStatus, type DepthMap, type Status, type Task } from './segment/cutout';
import { Panel } from './ui/panel';
import { bindSearch } from './ui/search';
import { bindSettings, loadSettings, saveSettings } from './ui/settings';
import { createToast } from './ui/toast';
import { buildCard, riseStart, type VoxelModel } from './voxel/card';
import { loadPixels } from './voxel/image';
import { locateArt, type Pixels, type Rect } from './voxel/raster';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

const canvas = $<HTMLCanvasElement>('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000, 0);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;

const scene = new THREE.Scene();
// A soft studio environment for the reflections; the lights below do the real work.
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.35;

// `|| 1`: a page loaded in a hidden embed starts at 0×0, and 0 / 0 would poison the framing with NaN.
const camera = new THREE.PerspectiveCamera(35, window.innerWidth / window.innerHeight || 1, 1, 4000);
camera.position.set(60, 90, 330);

const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 80, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.enablePan = false;
controls.minDistance = 40;
controls.maxDistance = 1000;
controls.maxPolarAngle = Math.PI / 2 - 0.04;
controls.autoRotateSpeed = 1.2;

// Card art is painted to be seen face on under even light, so the fill is generous and the sun soft.
scene.add(new THREE.HemisphereLight(0xffffff, 0x3a3550, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 2.1);
sun.position.set(70, 160, 140);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.3;
scene.add(sun, sun.target);
// A coloured light from behind picks out the silhouette in the card's colour.
const rim = new THREE.DirectionalLight(0xffffff, 1.4);
rim.position.set(-90, 60, -140);
scene.add(rim);

// Pedestal: a dark disc with a thin glowing band, scaled to the model.
const pedestal = new THREE.Group();
const top = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 96), new THREE.MeshStandardMaterial({ color: 0x1c1b26, roughness: 0.85 }));
top.position.y = -0.5;
top.receiveShadow = true;
const bandMaterial = new THREE.MeshStandardMaterial({ color: 0x000000, emissiveIntensity: 1.4, roughness: 0.6 });
const band = new THREE.Mesh(new THREE.CylinderGeometry(1.02, 1.02, 0.25, 96, 1, true), bandMaterial);
band.position.y = -0.12;
const base = new THREE.Mesh(new THREE.CylinderGeometry(1.02, 1.1, 4, 96), new THREE.MeshStandardMaterial({ color: 0x121118, roughness: 0.9 }));
base.position.y = -3;
pedestal.add(top, band, base);
pedestal.scale.set(70, 1, 70);
scene.add(pedestal);

const cloud = new VoxelCloud();
scene.add(cloud.mesh);
const sculpt = new SculptView();
scene.add(sculpt.group);

// ---------------------------------------------------------------------------
// Camera framing
// ---------------------------------------------------------------------------

const frameGoal = { target: new THREE.Vector3(0, 80, 0), distance: 330, radius: 70, active: false };

type Bounds = Pick<VoxelModel, 'min' | 'max'>;

/** Eases the camera to fit `model`, keeping the angle it's looking from. */
function frame(model: Bounds): void {
  const size = [model.max[0] - model.min[0], model.max[1] - model.min[1], model.max[2] - model.min[2]];
  const r = 0.5 * Math.hypot(size[0], size[1], size[2]);
  const fov = THREE.MathUtils.degToRad(camera.fov);
  // The tighter of the vertical and horizontal fields of view decides the distance.
  const half = Math.min(fov, 2 * Math.atan(Math.tan(fov / 2) * camera.aspect)) / 2;
  frameGoal.target.set((model.min[0] + model.max[0]) / 2, (model.min[1] + model.max[1]) / 2, (model.min[2] + model.max[2]) / 2);
  frameGoal.distance = THREE.MathUtils.clamp((r / Math.sin(half)) * 1.1, controls.minDistance, controls.maxDistance);
  frameGoal.radius = 0.5 * Math.hypot(Math.max(Math.abs(model.min[0]), Math.abs(model.max[0])) * 2, Math.max(Math.abs(model.min[2]), Math.abs(model.max[2])) * 2) + 6;
  frameGoal.active = true;

  // The shadow camera only needs to cover the model and its pedestal.
  const s = Math.max(r, frameGoal.radius) * 1.2;
  const cam = sun.shadow.camera;
  cam.left = cam.bottom = -s;
  cam.right = cam.top = s;
  cam.near = 1;
  cam.far = 800;
  cam.updateProjectionMatrix();
  sun.target.position.copy(frameGoal.target);
  sun.position.copy(frameGoal.target).add(new THREE.Vector3(70, 160, 140));
}
controls.addEventListener('start', () => (frameGoal.active = false));

const tmp = new THREE.Vector3();
function updateFraming(dt: number): void {
  const k = 1 - Math.exp(-dt * 3.2);
  const pr = THREE.MathUtils.lerp(pedestal.scale.x, frameGoal.radius, k);
  pedestal.scale.set(pr, 1, pr);
  if (!frameGoal.active) return;
  controls.target.lerp(frameGoal.target, k);
  const dir = tmp.subVectors(camera.position, controls.target);
  const dist = THREE.MathUtils.lerp(dir.length(), frameGoal.distance, k);
  camera.position.copy(controls.target).addScaledVector(dir.normalize(), dist);
  if (Math.abs(dist - frameGoal.distance) < 0.5 && controls.target.distanceTo(frameGoal.target) < 0.1) frameGoal.active = false;
}

/** Shifts the picture away from the panel, so the model is centred in the free space. */
function updateViewOffset(): void {
  const w = window.innerWidth;
  const h = window.innerHeight;
  const r = $('dex').getBoundingClientRect();
  let x = 0;
  let y = 0;
  if (w > 760) x = -(r.right + 8) / 2;
  else y = (h - r.top + 8) / 2;
  camera.setViewOffset(w, h, x, y, w, h);
  document.documentElement.style.setProperty('--dex-space', `${w > 760 ? r.width + 16 : 0}px`);
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const settings = loadSettings();
const toast = createToast($('toast'));
const panel = new Panel($('dex'));

/** Where the art usually is on a modern frame; used when the search for it fails. */
const USUAL_ART: Rect = { x: 0.082, y: 0.112, w: 0.836, h: 0.444 };
/** Above this mean squared difference, `locateArt` found something that isn't the art. */
const MATCH_LIMIT = 0.02;

interface Current {
  card: Card;
  face: number;
  front: Pixels;
  back: Pixels | null;
  /** The cut-out subject, once there is one (and while "cut out" is on). */
  cutout: Pixels | null;
  art: Rect;
  model: VoxelModel;
  /** The art's depth, once there is one (sculpt only). */
  depth: DepthMap | null;
}
let current: Current | null = null;
let token = 0;

const accent = new THREE.Color();
function applyAccent(card: Card): void {
  const c = accentFor(card.colorIdentity);
  document.documentElement.style.setProperty('--accent', c);
  accent.set(c);
  rim.color.copy(accent);
  bandMaterial.emissive.copy(accent);
}

const artMatches = new Map<string, Promise<Rect>>();
/** Where the art is on this face's card image, memoised. */
function findArt(cardUrl: string, artUrl: string, front: Pixels): Promise<Rect> {
  const key = `${cardUrl} ${artUrl}`;
  let p = artMatches.get(key);
  if (!p) {
    p = loadPixels(artUrl).then((art) => {
      const m = locateArt(front, art);
      return m && m.error < MATCH_LIMIT ? m.rect : USUAL_ART;
    });
    artMatches.set(key, p);
    p.catch(() => artMatches.delete(key));
  }
  return p;
}

const build = (c: Pick<Current, 'front' | 'back' | 'cutout' | 'art'>) =>
  buildCard(c, { layout: settings.layout, width: settings.detail, inflate: settings.inflate });

/**
 * Shows face `face` of `card`. The card goes up straight away; the cut-out subject follows when the
 * model has made it (at once if it's been made before), rising out of the card.
 */
async function show(card: Card, face: number, mode: TransitionMode): Promise<void> {
  const mine = ++token;
  $('loading').hidden = false;
  try {
    const f = card.faces[face];
    if (!f.image) throw new Error(`Scryfall has no image of ${f.name}.`);
    const other = card.twoSided ? card.faces[1 - face].image : null;
    const [front, back] = await Promise.all([
      loadPixels(f.image.card),
      loadPixels(other ? other.card : backUrl(card.cardBackId)).catch(() => null),
    ]);
    if (mine !== token) return;

    const art = await findArt(f.image.card, f.image.art, front);
    const ready = settings.popOut && hasCutout(f.image.art);
    const cut = ready ? await cutout(f.image.art) : null;
    const dep = sculpting() && hasDepth(f.image.art) ? await depth(f.image.art) : null;
    if (mine !== token) return;

    const model = build({ front, back, cutout: cut, art });
    cloud.show(model, mode);
    const first = !current;
    current = { card, face, front, back, cutout: cut, art, model, depth: dep };

    panel.render(card, face);
    panel.cutout(cut ? subjectNote(model) : '');
    applyAccent(card);
    present(true);
    const hash = `#${card.set}/${card.collectorNumber}`;
    // Each card is a history entry, so Back returns to the last one; the very first replaces the bare URL.
    if (location.hash !== hash) history[first ? 'replaceState' : 'pushState'](null, '', hash);
    document.title = `${f.name} · Voxel MTG`;

    if (settings.popOut && !cut) void popOut(mine);
    if (sculpting() && !dep) void measure(mine);
  } catch (err) {
    if (mine === token) toast(err instanceof Error ? err.message : String(err), 'error');
  } finally {
    if (mine === token) $('loading').hidden = true;
  }
}

function subjectNote(model: VoxelModel): string {
  return model.subjectPixels ? '' : 'No single subject stood out in this art, so it stays flat.';
}

/** The sculpt style wants depth: it's on, and there's a subject to shape. */
const sculpting = () => settings.style === 'sculpt' && settings.popOut;

const modelStatus: Record<Task, Status> = { cutout: { state: 'idle' }, depth: { state: 'idle' } };
let cutting = false;
let measuring = false;
/** What the models are busy with: downloading, cutting out, estimating depth. Empty when idle. */
function cutoutLine(): string {
  const mb = (n: number) => (n / 1e6).toFixed(0);
  const jobs: [Task, boolean, string][] = [
    ['cutout', cutting, 'Cutting the subject out of the art…'],
    ['depth', measuring, 'Estimating how deep the art is…'],
  ];
  for (const [task, busy] of jobs) {
    const s = modelStatus[task];
    if (busy && s.state === 'downloading' && s.loaded < s.total) return `Downloading the ${task} model (once): ${mb(s.loaded)} / ${mb(s.total)} MB…`;
  }
  return jobs
    .filter(([, busy]) => busy)
    .map(([, , line]) => line)
    .join(' ');
}
onStatus((task, s) => {
  modelStatus[task] = s;
  if (cutting || measuring) panel.cutout(cutoutLine());
});
const deviceName = (task: Task) => {
  const s = modelStatus[task];
  return s.state === 'ready' ? ` on ${s.device === 'webgpu' ? 'the GPU' : 'the CPU'}` : '';
};

/** Shows the current card in the chosen style, frames it, and updates the stats line. */
function present(grow: boolean): void {
  if (!current) return;
  const sculpted = settings.style === 'sculpt';
  cloud.mesh.visible = !sculpted;
  sculpt.group.visible = sculpted;
  $('burst').hidden = sculpted;
  if (!sculpted) {
    panel.voxels(current.model);
    frame(current.model);
    return;
  }
  const { model, cutout: cut } = current;
  sculpt.setCard(current.front, current.back, model.width, model.height, settings.layout);
  // The subject waits for its depth, so it doesn't change shape a second later.
  const mesh =
    cut && current.depth && model.subjectPixels
      ? sculptSubject(
          { cutout: cut, depth: current.depth, art: current.art, cardWidth: model.width, cardHeight: model.height },
          { layout: settings.layout, inflate: settings.inflate, depth: settings.depth },
        )
      : null;
  sculpt.setSubject(mesh, cut, settings.layout, grow);
  const b = sculpt.bounds();
  panel.sculpt(sculpt.triangles, b);
  frame(b);
}

/** Estimates the depth of the current card's art, then shapes the sculpt with it. */
async function measure(mine: number): Promise<void> {
  if (!current) return;
  const f = current.card.faces[current.face];
  if (!f.image) return;
  measuring = true;
  panel.cutout(cutoutLine());
  const started = performance.now();
  try {
    const d = await depth(f.image.art);
    if (mine !== token || !current) return;
    measuring = false;
    current.depth = d;
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    if (cutting) panel.cutout(cutoutLine());
    else if (current.cutout) panel.cutout(current.model.subjectPixels ? `Cut out and measured in ${seconds} s${deviceName('depth')}.` : subjectNote(current.model));
    if (settings.style === 'sculpt') present(true);
  } catch (err) {
    if (mine !== token) return;
    measuring = false;
    panel.cutout(cutoutLine());
    toast(`Couldn't estimate the depth: ${err instanceof Error ? err.message : String(err)}`, 'error');
  }
}

/** Cuts the current card's subject out, then raises it out of the card. */
async function popOut(mine: number): Promise<void> {
  if (!current) return;
  const f = current.card.faces[current.face];
  if (!f.image) return;
  cutting = true;
  panel.cutout(cutoutLine());
  const started = performance.now();
  try {
    const cut = await cutout(f.image.art);
    if (mine !== token || !current || !settings.popOut) return;
    cutting = false;
    current.cutout = cut;
    const model = build(current);
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    if (measuring) panel.cutout(cutoutLine());
    else panel.cutout(model.subjectPixels ? `Cut out in ${seconds} s${deviceName('cutout')}.` : subjectNote(model));

    if (settings.style === 'sculpt') {
      // The voxels change out of sight; the sculpt's subject grows once the depth is in too.
      cloud.show(model, 'morph');
      current.model = model;
      present(true);
      return;
    }

    // Rise only from the bare card at rest, so every card voxel is where the new model expects it.
    await cloud.settled();
    if (mine !== token || !current || current.cutout !== cut) return;
    const bare = current.model.subjectPixels === 0 && current.model.count === model.cardCount;
    if (bare && model.subjectPixels) cloud.show(model, 'rise', riseStart(model));
    else cloud.show(model, 'morph');
    current.model = model;
    present(false);
  } catch (err) {
    if (mine !== token) return;
    cutting = false;
    panel.cutout('');
    toast(`Couldn't cut out the art: ${err instanceof Error ? err.message : String(err)}`, 'error');
  }
}

/** Rebuilds the current card from what's already loaded (layout, detail, inflation, cutout on/off). */
function rebuild(): void {
  if (!current) return;
  if (!settings.popOut) current.cutout = null;
  const model = build(current);
  cloud.show(model, 'morph');
  current.model = model;
  if (current.cutout) panel.cutout(subjectNote(model));
  present(false);
  if (settings.popOut && !current.cutout) void popOut(token);
  else if (!settings.popOut) panel.cutout('');
  if (sculpting() && !current.depth && !measuring) void measure(token);
}

/** Switches between sculpt and voxels. */
function restyle(): void {
  if (!current) return;
  present(true);
  if (sculpting() && !current.depth && !measuring) void measure(token);
}

async function load(fetchCard: () => Promise<Card>, notFound?: string): Promise<void> {
  const mine = ++token;
  $('loading').hidden = false;
  try {
    const card = await fetchCard();
    if (mine !== token) return;
    await show(card, 0, settings.transition);
  } catch (err) {
    if (mine !== token) return;
    $('loading').hidden = true;
    toast(err instanceof ScryfallError && err.status === 404 && notFound ? notFound : err instanceof Error ? err.message : String(err), 'error');
  }
}

const random = () => load(() => randomCard(settings.randomQuery), `Nothing matches “${settings.randomQuery}”.`);

/** The next printing in the same set, skipping gaps in the numbering. */
async function step(dir: number): Promise<void> {
  if (!current) return;
  const { set, collectorNumber } = current.card;
  let number = collectorNumber;
  for (let tries = 0; tries < 4; tries++) {
    const n = stepNumber(number, dir);
    if (!n) break;
    number = n;
    try {
      const card = await cardByPrinting(set, n);
      await show(card, 0, settings.transition);
      return;
    } catch (err) {
      if (!(err instanceof ScryfallError && err.status === 404)) {
        toast(err instanceof Error ? err.message : String(err), 'error');
        return;
      }
    }
  }
  toast(`No ${dir > 0 ? 'next' : 'previous'} card in ${current.card.setName} from #${collectorNumber}.`);
}

function fromHash(): void {
  const h = parseHash(location.hash);
  if (!h) return;
  if ('name' in h) void load(() => cardByName(h.name), `No card is called “${h.name}”.`);
  else if (!current || current.card.set !== h.set || current.card.collectorNumber !== h.number) {
    void load(() => cardByPrinting(h.set, h.number), `There's no ${h.set.toUpperCase()} #${h.number}.`);
  }
}

function flip(): void {
  if (current?.card.twoSided) void show(current.card, 1 - current.face, 'morph');
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

const syncSettings = bindSettings($('settings'), settings, (key) => {
  if (key === 'autoRotate') controls.autoRotate = settings.autoRotate;
  else if (key === 'layout' || key === 'detail' || key === 'inflate' || key === 'popOut') rebuild();
  else if (key === 'style') restyle();
  else if (key === 'depth') present(false);
});
controls.autoRotate = settings.autoRotate;

bindSearch($<HTMLFormElement>('search-form'), $<HTMLInputElement>('search'), $<HTMLDataListElement>('card-names'), (name) =>
  load(() => cardByName(name), `No card matches “${name}”.`),
);

$('prev').addEventListener('click', () => void step(-1));
$('next').addEventListener('click', () => void step(1));
$('random').addEventListener('click', () => void random());
$('flip').addEventListener('click', flip);
$('burst').addEventListener('click', burst);
$('settings-toggle').addEventListener('click', () => ($('settings').hidden = !$('settings').hidden));
$('fullscreen').addEventListener('click', toggleFullscreen);
$('collapse').addEventListener('click', () => {
  const collapsed = $('dex').classList.toggle('collapsed');
  $('collapse').textContent = collapsed ? '+' : '–';
  updateViewOffset();
});

function burst(): void {
  if (current && settings.style === 'voxels') cloud.show(current.model, 'explode');
}

function toggleFullscreen(): void {
  if (document.fullscreenElement) void document.exitFullscreen();
  else void document.documentElement.requestFullscreen?.();
}

window.addEventListener('hashchange', fromHash);

window.addEventListener('keydown', (e) => {
  const t = e.target;
  if ((t instanceof Element && t.matches('input:not([type=checkbox]):not([type=range]), select, textarea')) || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === 'arrowleft') void step(-1);
  else if (k === 'arrowright') void step(1);
  else if (k === 'r') void random();
  else if (k === 't') flip();
  else if (k === 'b') burst();
  else if (k === 'l') {
    settings.layout = settings.layout === 'relief' ? 'summon' : 'relief';
    syncSettings();
    saveSettings(settings);
    rebuild();
  } else if (k === 'v') {
    settings.style = settings.style === 'sculpt' ? 'voxels' : 'sculpt';
    syncSettings();
    saveSettings(settings);
    restyle();
  } else if (k === 'a') {
    settings.autoRotate = controls.autoRotate = !settings.autoRotate;
    syncSettings();
    saveSettings(settings);
  } else if (k === 'f') toggleFullscreen();
  else if (k === '/') $('search').focus();
  else return;
  e.preventDefault();
});

// Poke: a click (not a drag) on the model.
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
function rayFrom(e: PointerEvent): THREE.Raycaster {
  const r = canvas.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  return raycaster;
}
let down: { x: number; y: number; t: number } | null = null;
canvas.addEventListener('pointerdown', (e) => (down = { x: e.clientX, y: e.clientY, t: performance.now() }));
canvas.addEventListener('pointerup', (e) => {
  if (!down) return;
  const click = Math.hypot(e.clientX - down.x, e.clientY - down.y) < 6 && performance.now() - down.t < 500;
  down = null;
  if (!click || !cloud.mesh.visible) return;
  const hit = cloud.pick(rayFrom(e));
  if (hit) cloud.poke(hit);
});
// Hover: at most one raycast per frame, and none while dragging.
let hover: PointerEvent | null = null;
canvas.addEventListener('pointermove', (e) => (hover = e.buttons ? null : e));
canvas.addEventListener('pointerleave', () => {
  hover = null;
  canvas.classList.remove('pointer');
});

function onResize(): void {
  const w = window.innerWidth;
  const h = window.innerHeight;
  // A hidden embed can report 0×0, which would make the aspect ratio NaN.
  if (!w || !h) return;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  updateViewOffset();
  if (current) frame(settings.style === 'sculpt' ? sculpt.bounds() : current.model);
}
window.addEventListener('resize', onResize);
new ResizeObserver(updateViewOffset).observe($('dex'));
onResize();

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

if (parseHash(location.hash)) fromHash();
else void load(() => cardByName('Phenax, God of Deception'));

function tick(dt: number): void {
  cloud.update(dt);
  sculpt.update(dt);
  updateFraming(dt);
  controls.update(dt);
  if (hover) {
    canvas.classList.toggle('pointer', cloud.mesh.visible && cloud.pick(rayFrom(hover)) !== null);
    hover = null;
  }
  renderer.render(scene, camera);
}

const timer = new THREE.Timer();
timer.connect(document);
renderer.setAnimationLoop((time) => {
  timer.update(time);
  // Clamped so a backgrounded tab doesn't jump the animations on return.
  tick(Math.min(timer.getDelta(), 0.05));
});

// Dev only: lets a test harness step frames by hand (a hidden tab gets no animation frames).
if (import.meta.env.DEV) Object.assign(window, { __dbg: { tick, cloud, sculpt, camera, get current() { return current; } } });
