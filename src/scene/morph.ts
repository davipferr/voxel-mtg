/**
 * Transitions between two voxel models, planned on the CPU and played on the GPU.
 *
 * A plan pairs every voxel of the old model with one of the new model and gives each pair a flight:
 * a curved path, a delay, a tumble. The vertex shader evaluates the flight from a single time uniform,
 * so the CPU uploads the plan once and then does nothing per frame. `sampleAt` is the same maths in
 * TypeScript, used to start a new transition from wherever an interrupted one had got to.
 */

export interface Snapshot {
  count: number;
  /** xyz per voxel. */
  positions: Float32Array;
  /** rgb per voxel. */
  colors: Float32Array;
  /** Uniform scale per voxel; 0 = invisible. */
  scales: Float32Array;
}

export type TransitionMode = 'explode' | 'morph' | 'recolor' | 'rise';

export interface Plan {
  count: number;
  from: Float32Array;
  to: Float32Array;
  /** Offset of the Bézier control point from the midpoint of `from` and `to`. */
  mid: Float32Array;
  fromColor: Float32Array;
  toColor: Float32Array;
  /** from, to scale pairs. */
  scale: Float32Array;
  /** Seconds before each voxel starts moving. */
  delay: Float32Array;
  /** Tumble axis xyz (unit) and peak angle. */
  spin: Float32Array;
  /** Seconds each voxel's flight takes. */
  duration: number;
  /** Extra scale at mid-flight: 0.3 = voxels swell 30% halfway through. */
  pop: number;
  /** When the last voxel lands. */
  end: number;
  /** The new model, at rest. */
  target: Snapshot;
}

interface ModeConfig {
  duration: number;
  /** Largest stagger delay. */
  stagger: number;
  /** How far the control point is pushed away, in voxels. */
  scatter: number;
  /** Largest tumble, in radians. */
  spin: number;
  pop: number;
}

export const MODES: Record<TransitionMode, ModeConfig> = {
  // Shuffled pairs fly out past the model and fall back in.
  explode: { duration: 1.5, stagger: 0.35, scatter: 34, spin: Math.PI * 3, pop: 0 },
  // Pairs matched by position flow into the new shape, assembling bottom to top.
  morph: { duration: 1.1, stagger: 0.45, scatter: 4, spin: Math.PI * 0.5, pop: 0 },
  // Same shape, new colours: a sparkle sweeps from top to bottom.
  recolor: { duration: 0.6, stagger: 0.55, scatter: 0, spin: 0, pop: 0.45 },
  // Already paired voxel for voxel: each one rises from its origin, the shortest trips first.
  rise: { duration: 1.2, stagger: 0.9, scatter: 0, spin: 0, pop: 0 },
};

// ---------------------------------------------------------------------------
// Easing
// ---------------------------------------------------------------------------

const OMEGA = 11;
const ZETA = 0.55;
const OMEGA_D = OMEGA * Math.sqrt(1 - ZETA * ZETA);

/**
 * Step response of an underdamped spring, normalised to t ∈ [0, 1]: it overshoots about 13% and
 * rings out by t = 1 (the residue there is 0.2%, then it snaps to exactly 1).
 */
export function spring(t: number): number {
  if (t <= 0) return 0;
  if (t >= 1) return 1;
  const decay = Math.exp(-ZETA * OMEGA * t);
  return 1 - decay * (Math.cos(OMEGA_D * t) + ((ZETA * OMEGA) / OMEGA_D) * Math.sin(OMEGA_D * t));
}

export const smoothstep = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** GLSL twin of `spring`, `smoothstep` aside (GLSL has it built in). Keep the two in sync. */
export const SPRING_GLSL = /* glsl */ `
float voxSpring(float t) {
  if (t >= 1.0) return 1.0;
  float decay = exp(-${(ZETA * OMEGA).toFixed(4)} * t);
  return 1.0 - decay * (cos(${OMEGA_D.toFixed(4)} * t) + ${((ZETA * OMEGA) / OMEGA_D).toFixed(4)} * sin(${OMEGA_D.toFixed(4)} * t));
}
`;

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

export function snapshotOf(model: { count: number; positions: Float32Array; colors: Float32Array }): Snapshot {
  return { count: model.count, positions: model.positions, colors: model.colors, scales: new Float32Array(model.count).fill(1) };
}

/** A single invisible voxel at `point`: the source for the very first model, which grows out of it. */
export function seed(x: number, y: number, z: number): Snapshot {
  return { count: 1, positions: Float32Array.of(x, y, z), colors: Float32Array.of(1, 1, 1), scales: Float32Array.of(0) };
}

interface Bounds {
  min: [number, number, number];
  size: [number, number, number];
}

function bounds(s: Snapshot): Bounds {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < s.count; i++) {
    for (let k = 0; k < 3; k++) {
      const v = s.positions[i * 3 + k];
      if (v < min[k]) min[k] = v;
      if (v > max[k]) max[k] = v;
    }
  }
  return { min, size: [max[0] - min[0] || 1, max[1] - min[1] || 1, max[2] - min[2] || 1] };
}

/** Voxel indices in a spatial order: rows bottom to top, then left to right, then back to front. */
export function spatialOrder(s: Snapshot): Uint32Array {
  const b = bounds(s);
  const ROWS = 48;
  const key = new Float64Array(s.count);
  for (let i = 0; i < s.count; i++) {
    const ny = (s.positions[i * 3 + 1] - b.min[1]) / b.size[1];
    const nx = (s.positions[i * 3] - b.min[0]) / b.size[0];
    const nz = (s.positions[i * 3 + 2] - b.min[2]) / b.size[2];
    key[i] = Math.floor(ny * ROWS) * 4 + nx * 2 + nz;
  }
  return Uint32Array.from({ length: s.count }, (_, i) => i).sort((a, b) => key[a] - key[b]);
}

/** How far voxel `i` moves between two snapshots paired index for index. */
function trip(a: Snapshot, b: Snapshot, i: number): number {
  return Math.hypot(b.positions[i * 3] - a.positions[i * 3], b.positions[i * 3 + 1] - a.positions[i * 3 + 1], b.positions[i * 3 + 2] - a.positions[i * 3 + 2]);
}

function shuffledOrder(n: number, rand: () => number): Uint32Array {
  const order = Uint32Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}

function randomUnit(rand: () => number): [number, number, number] {
  // Uniform on the sphere: uniform z and uniform angle.
  const z = rand() * 2 - 1;
  const a = rand() * Math.PI * 2;
  const r = Math.sqrt(1 - z * z);
  return [r * Math.cos(a), r * Math.sin(a), z];
}

/**
 * Pairs `from` with `to`. Both are put in the same kind of order (spatial for morph/recolor, random
 * for explode) and walked in step, so the i-th quantile of one meets the i-th quantile of the other.
 * When the counts differ, the smaller side is stretched: its voxels are reused, and every reuse after
 * the first starts (or ends) at scale 0, so voxels appear out of, or vanish into, a neighbour.
 *
 * `rise` is different: the caller has already paired the two sides index for index (they must be the
 * same length, or it falls back to `morph`), and each voxel is delayed by how far it has to travel.
 */
export function planTransition(from: Snapshot, to: Snapshot, mode: TransitionMode, rand: () => number = Math.random): Plan {
  if (from.count === 0) from = seed(0, 0, 0);
  if (mode === 'rise' && from.count !== to.count) mode = 'morph';
  const cfg = MODES[mode];
  const n = from.count;
  const m = to.count;
  const M = Math.max(n, m);

  const identity = (k: number) => Uint32Array.from({ length: k }, (_, i) => i);
  const orderA = mode === 'explode' ? shuffledOrder(n, rand) : mode === 'rise' ? identity(n) : spatialOrder(from);
  const orderB = mode === 'explode' ? shuffledOrder(m, rand) : mode === 'rise' ? identity(m) : spatialOrder(to);
  let longest = 0;
  if (mode === 'rise') for (let i = 0; i < n; i++) longest = Math.max(longest, trip(from, to, i));
  const tb = bounds(to);
  const centre = [tb.min[0] + tb.size[0] / 2, tb.min[1] + tb.size[1] / 2, tb.min[2] + tb.size[2] / 2];

  const plan: Plan = {
    count: M,
    from: new Float32Array(M * 3),
    to: new Float32Array(M * 3),
    mid: new Float32Array(M * 3),
    fromColor: new Float32Array(M * 3),
    toColor: new Float32Array(M * 3),
    scale: new Float32Array(M * 2),
    delay: new Float32Array(M),
    spin: new Float32Array(M * 4),
    duration: cfg.duration,
    pop: cfg.pop,
    end: 0,
    target: to,
  };

  let maxDelay = 0;
  for (let i = 0; i < M; i++) {
    const ia = Math.floor((i * n) / M);
    const ib = Math.floor((i * m) / M);
    const firstA = i === 0 || Math.floor(((i - 1) * n) / M) !== ia;
    const firstB = i === 0 || Math.floor(((i - 1) * m) / M) !== ib;
    const a = orderA[ia];
    const b = orderB[ib];

    for (let k = 0; k < 3; k++) {
      plan.from[i * 3 + k] = from.positions[a * 3 + k];
      plan.to[i * 3 + k] = to.positions[b * 3 + k];
      plan.fromColor[i * 3 + k] = from.colors[a * 3 + k];
      plan.toColor[i * 3 + k] = to.colors[b * 3 + k];
    }
    plan.scale[i * 2] = firstA ? from.scales[a] : 0;
    plan.scale[i * 2 + 1] = firstB ? to.scales[b] : 0;

    const ny = (to.positions[b * 3 + 1] - tb.min[1]) / tb.size[1];
    const order =
      mode === 'explode' ? rand() : mode === 'morph' ? ny : mode === 'rise' ? (longest ? trip(from, to, i) / longest : 0) : 1 - ny;
    const delay = order * cfg.stagger;
    plan.delay[i] = delay;
    if (delay > maxDelay) maxDelay = delay;

    if (cfg.scatter > 0) {
      const r = randomUnit(rand);
      let dir = r;
      if (mode === 'explode') {
        // Outward from the model's centre, roughened, and biased up so the cloud rises before it falls.
        const mx = (plan.from[i * 3] + plan.to[i * 3]) / 2 - centre[0];
        const my = (plan.from[i * 3 + 1] + plan.to[i * 3 + 1]) / 2 - centre[1];
        const mz = (plan.from[i * 3 + 2] + plan.to[i * 3 + 2]) / 2 - centre[2];
        const len = Math.hypot(mx, my, mz) || 1;
        dir = [mx / len + r[0] * 0.8, my / len + r[1] * 0.8 + 0.5, mz / len + r[2] * 0.8];
      }
      const amount = cfg.scatter * (0.4 + 0.8 * rand());
      for (let k = 0; k < 3; k++) plan.mid[i * 3 + k] = dir[k] * amount;
    }

    if (cfg.spin > 0) {
      const axis = randomUnit(rand);
      plan.spin.set([axis[0], axis[1], axis[2], cfg.spin * (0.3 + 0.7 * rand())], i * 4);
    }
  }
  plan.end = maxDelay + cfg.duration;
  return plan;
}

/** A plan that holds `s` still: every voxel is already where it's going. */
export function restingPlan(s: Snapshot): Plan {
  const scale = new Float32Array(s.count * 2);
  for (let i = 0; i < s.count; i++) scale[i * 2] = scale[i * 2 + 1] = s.scales[i];
  return {
    count: s.count,
    from: s.positions,
    to: s.positions,
    mid: new Float32Array(s.count * 3),
    fromColor: s.colors,
    toColor: s.colors,
    scale,
    delay: new Float32Array(s.count),
    spin: new Float32Array(s.count * 4),
    duration: 1,
    pop: 0,
    end: 0,
    target: s,
  };
}

/**
 * Where every voxel of `plan` is at `time`: the shader's maths on the CPU. Invisible voxels are left
 * out, so an interrupted transition hands over only what's on screen. Tumble isn't carried over; the
 * next plan starts axis-aligned.
 */
export function sampleAt(plan: Plan, time: number): Snapshot {
  const positions = new Float32Array(plan.count * 3);
  const colors = new Float32Array(plan.count * 3);
  const scales = new Float32Array(plan.count);
  let n = 0;
  for (let i = 0; i < plan.count; i++) {
    const t = Math.min(1, Math.max(0, (time - plan.delay[i]) / plan.duration));
    const scale = (plan.scale[i * 2] + (plan.scale[i * 2 + 1] - plan.scale[i * 2]) * smoothstep(0, 1, t)) * (1 + plan.pop * Math.sin(Math.PI * t));
    if (scale < 0.02) continue;
    const e = spring(t);
    const u = 1 - e;
    const c = smoothstep(0.1, 0.9, t);
    for (let k = 0; k < 3; k++) {
      const a = plan.from[i * 3 + k];
      const b = plan.to[i * 3 + k];
      const ctrl = (a + b) / 2 + plan.mid[i * 3 + k];
      positions[n * 3 + k] = u * u * a + 2 * u * e * ctrl + e * e * b;
      colors[n * 3 + k] = plan.fromColor[i * 3 + k] + (plan.toColor[i * 3 + k] - plan.fromColor[i * 3 + k]) * c;
    }
    scales[n] = scale;
    n++;
  }
  return { count: n, positions: positions.subarray(0, n * 3), colors: colors.subarray(0, n * 3), scales: scales.subarray(0, n) };
}
