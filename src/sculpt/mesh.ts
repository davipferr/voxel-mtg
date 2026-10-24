/**
 * Cutout + depth → a smooth, closed mesh: the "sculpt" style, the alternative to voxels.
 *
 * The cutout gives the silhouette and the depth model says which parts of the picture are nearer than
 * others, but neither says how thick anything is. So the shape is two things added together:
 * - inflation, as in voxelize.ts: thickness grows with the distance from the outline, so the body is
 *   round and a thin limb stays thin;
 * - depth: the parts the model sees as nearer come forward, so a snout or a raised arm sticks out
 *   and a wing behind the body sits back.
 *
 * The surface is a height field on a grid finer than the voxels, one vertex per grid pixel inside the
 * subject. Cells with all four corners inside get two triangles, cells with three get one, which cuts
 * the staircase off the outline. A wall joins the front to the back (or, for a relief, to the card)
 * along every boundary edge, so the mesh is closed.
 */

import type { DepthMap } from '../segment/cutout';
import { MAX_COVERAGE, MIN_COVERAGE, type Layout } from '../voxel/card';
import { cleanMask, coverage, resample, type Pixels, type Rect } from '../voxel/raster';
import { distanceTransform } from '../voxel/voxelize';

export interface SculptOptions {
  /** Grid columns across the cutout. */
  columns: number;
  /** Size of the cutout in output units. */
  width: number;
  height: number;
  /** 0 = flat, 1 = round. */
  inflate: number;
  /** 0 = ignore the depth map, 1 = as estimated. */
  depth: number;
  /** A relief on the card (front only, walls down to z = 0) rather than a free-standing figure. */
  oneSided: boolean;
}

export interface SculptMesh {
  /** xyz. From `sculpt`, x runs left to right over 0..width, y bottom to top over 0..height, z out of the picture. */
  positions: Float32Array;
  /** Into the cutout image; v = 0 is its top row. */
  uvs: Float32Array;
  indices: Uint32Array;
  min: [number, number, number];
  max: [number, number, number];
  /** Grid pixels inside the subject (0 = nothing stood out, and the mesh is empty). */
  pixels: number;
}

/** Thickness at full inflation, as in voxelize.ts: DOME · √(distance from the outline). */
const DOME = 3;
/** How far depth can push the nearest part forward, as a share of the subject's size. */
const RELIEF_DEPTH = 0.18;
const FIGURE_DEPTH = 0.3;
/** On a relief, depth fades in over this share of the subject's size from the outline, so the edge meets the card. */
const RAMP = 0.08;
/** A figure is never thinner than this, either side of its middle. */
const MIN_HALF = 0.4;

const EMPTY: SculptMesh = { positions: new Float32Array(), uvs: new Float32Array(), indices: new Uint32Array(), min: [0, 0, 0], max: [0, 0, 0], pixels: 0 };

const smoothstep = (e0: number, e1: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

/** Averages each masked cell with its masked 3 × 3 neighbours, `passes` times. */
function smooth(f: Float32Array, mask: Uint8Array, w: number, h: number, passes: number): void {
  const tmp = new Float32Array(f.length);
  for (let p = 0; p < passes; p++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (!mask[i]) continue;
        let sum = 0;
        let n = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= h) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx;
            if (xx < 0 || xx >= w || !mask[yy * w + xx]) continue;
            sum += f[yy * w + xx];
            n++;
          }
        }
        tmp[i] = sum / n;
      }
    }
    for (let i = 0; i < f.length; i++) if (mask[i]) f[i] = tmp[i];
  }
}

/** Bilinear sample of `d` at continuous pixel coordinates (pixel centres at integers). */
function sample(d: DepthMap, x: number, y: number): number {
  const cx = Math.min(d.width - 1, Math.max(0, x));
  const cy = Math.min(d.height - 1, Math.max(0, y));
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const x1 = Math.min(d.width - 1, x0 + 1);
  const y1 = Math.min(d.height - 1, y0 + 1);
  const fx = cx - x0;
  const fy = cy - y0;
  const top = d.data[y0 * d.width + x0] * (1 - fx) + d.data[y0 * d.width + x1] * fx;
  const bottom = d.data[y1 * d.width + x0] * (1 - fx) + d.data[y1 * d.width + x1] * fx;
  return top * (1 - fy) + bottom * fy;
}

/**
 * How near each masked grid pixel is, 0 (furthest part of the subject) to 1 (nearest). The ends are
 * the 2nd and 98th percentiles, so a few stray pixels don't flatten everything else.
 */
function nearness(d: DepthMap, mask: Uint8Array, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h);
  const values: number[] = [];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!mask[i]) continue;
      out[i] = sample(d, ((x + 0.5) * d.width) / w - 0.5, ((y + 0.5) * d.height) / h - 0.5);
      values.push(out[i]);
    }
  }
  values.sort((a, b) => a - b);
  const lo = values[Math.floor(values.length * 0.02)];
  const hi = values[Math.min(values.length - 1, Math.floor(values.length * 0.98))];
  const range = hi - lo || 1;
  for (let i = 0; i < out.length; i++) if (mask[i]) out[i] = Math.min(1, Math.max(0, (out[i] - lo) / range));
  smooth(out, mask, w, h, 3);
  return out;
}

export function sculpt(cutout: Pixels, depth: DepthMap | null, opts: SculptOptions): SculptMesh {
  const gw = Math.max(2, Math.round(opts.columns));
  const gh = Math.max(2, Math.round((gw * cutout.height) / cutout.width));
  const sx = opts.width / gw;
  const sy = opts.height / gh;
  const cell = (sx + sy) / 2;

  const px = resample(cutout, gw, gh);
  const mask = new Uint8Array(gw * gh);
  for (let i = 0; i < mask.length; i++) mask[i] = px.data[i * 4 + 3] >= 128 ? 1 : 0;
  cleanMask(mask, gw, gh);
  const c = coverage(mask);
  if (c < MIN_COVERAGE || c > MAX_COVERAGE) return EMPTY;

  // Distance to the outline, with a one-pixel empty border so the image's edge counts as outline.
  const PW = gw + 2;
  const padded = new Uint8Array(PW * (gh + 2));
  for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) padded[(y + 1) * PW + x + 1] = mask[y * gw + x];
  const pd = distanceTransform(padded, PW, gh + 2);

  let x0 = gw;
  let x1 = 0;
  let y0 = gh;
  let y1 = 0;
  let pixels = 0;
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      if (!mask[y * gw + x]) continue;
      pixels++;
      x0 = Math.min(x0, x);
      x1 = Math.max(x1, x);
      y0 = Math.min(y0, y);
      y1 = Math.max(y1, y);
    }
  }
  const extent = Math.max((x1 - x0 + 1) * sx, (y1 - y0 + 1) * sy);

  const near = depth && opts.depth > 0 ? nearness(depth, mask, gw, gh) : null;
  const amp = opts.depth * extent * (opts.oneSided ? RELIEF_DEPTH : FIGURE_DEPTH);
  const k = DOME * opts.inflate * (opts.oneSided ? 1.4 : 1);
  const front = new Float32Array(gw * gh);
  const back = new Float32Array(gw * gh);
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      const i = y * gw + x;
      if (!mask[i]) continue;
      const d = Math.max(0, (pd[(y + 1) * PW + x + 1] - 0.5) * cell);
      const dome = k * Math.sqrt(d);
      const n = near ? near[i] : 0.5;
      if (opts.oneSided) {
        front[i] = dome + amp * n * smoothstep(0, RAMP * extent, d);
      } else {
        // The middle follows the depth; the thickness is the dome, either side of it.
        const mid = amp * (n - 0.5);
        const half = Math.max(MIN_HALF, dome);
        front[i] = mid + half;
        back[i] = mid - half;
      }
    }
  }

  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];
  const vertex = (x: number, y: number, z: number) => {
    positions.push((x + 0.5) * sx, (gh - y - 0.5) * sy, z);
    uvs.push((x + 0.5) / gw, (y + 0.5) / gh);
    return positions.length / 3 - 1;
  };
  const frontId = new Int32Array(gw * gh).fill(-1);
  for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) if (mask[y * gw + x]) frontId[y * gw + x] = vertex(x, y, front[y * gw + x]);

  // Front triangles, counter-clockwise seen from +z, in grid pixel indices.
  const tris: number[] = [];
  for (let y = 0; y < gh - 1; y++) {
    for (let x = 0; x < gw - 1; x++) {
      // Bottom-left, bottom-right, top-right, top-left: counter-clockwise, since y grows downwards.
      const ring = [(y + 1) * gw + x, (y + 1) * gw + x + 1, y * gw + x + 1, y * gw + x].filter((i) => mask[i]);
      if (ring.length === 4) tris.push(ring[0], ring[1], ring[2], ring[0], ring[2], ring[3]);
      else if (ring.length === 3) tris.push(ring[0], ring[1], ring[2]);
    }
  }
  if (!tris.length) return EMPTY;

  // An edge used by one triangle only is on the boundary; its reverse belongs to no triangle.
  const N = gw * gh;
  const edges = new Set<number>();
  for (let t = 0; t < tris.length; t += 3) for (let e = 0; e < 3; e++) edges.add(tris[t + e] * N + tris[t + ((e + 1) % 3)]);
  const boundary: [number, number][] = [];
  for (const key of edges) {
    const a = Math.floor(key / N);
    const b = key % N;
    if (!edges.has(b * N + a)) boundary.push([a, b]);
  }

  for (let t = 0; t < tris.length; t += 3) indices.push(frontId[tris[t]], frontId[tris[t + 1]], frontId[tris[t + 2]]);

  // The back: a full surface for a figure, just the foot of each wall for a relief.
  const backId = new Int32Array(N).fill(-1);
  const backOf = (i: number) => {
    if (backId[i] < 0) backId[i] = vertex(i % gw, Math.floor(i / gw), opts.oneSided ? 0 : back[i]);
    return backId[i];
  };
  if (!opts.oneSided) for (let t = 0; t < tris.length; t += 3) indices.push(backOf(tris[t]), backOf(tris[t + 2]), backOf(tris[t + 1]));

  // Walls face outwards: the inside of a boundary edge a → b is on its left, seen from +z.
  for (const [a, b] of boundary) {
    const A = frontId[a];
    const B = frontId[b];
    const A2 = backOf(a);
    const B2 = backOf(b);
    indices.push(A, A2, B, B, A2, B2);
  }

  const out = Float32Array.from(positions);
  return { positions: out, uvs: Float32Array.from(uvs), indices: Uint32Array.from(indices), ...bounds(out), pixels };
}

function bounds(p: Float32Array): Pick<SculptMesh, 'min' | 'max'> {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < p.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      min[k] = Math.min(min[k], p[i + k]);
      max[k] = Math.max(max[k], p[i + k]);
    }
  }
  return { min, max };
}

function translate(m: SculptMesh, dx: number, dy: number, dz: number): SculptMesh {
  const p = m.positions;
  for (let i = 0; i < p.length; i += 3) {
    p[i] += dx;
    p[i + 1] += dy;
    p[i + 2] += dz;
  }
  return { ...m, ...bounds(p) };
}

/** Grid pixels per voxel: the mesh is twice as fine as the voxels. */
const FINE = 2;

export interface SubjectInput {
  cutout: Pixels;
  depth: DepthMap | null;
  /** Where the art crop sits on the card, normalised to 0–1. */
  art: Rect;
  /** Card size in voxels. */
  cardWidth: number;
  cardHeight: number;
}

/**
 * The subject in the same world space as the voxel card of the same size: out of the art box of an
 * upright card (relief), or standing on a card lying flat (summon).
 */
export function sculptSubject(input: SubjectInput, opts: { layout: Layout; inflate: number; depth: number }): SculptMesh {
  const W = input.cardWidth;
  const H = input.cardHeight;
  const ax = input.art.x * W;
  const ay = input.art.y * H;
  const aw = input.art.w * W;
  const ah = input.art.h * H;

  if (opts.layout === 'relief') {
    const m = sculpt(input.cutout, input.depth, { columns: aw * FINE, width: aw, height: ah, inflate: opts.inflate, depth: opts.depth, oneSided: true });
    // The card's front is at z = 0.5.
    return m.pixels ? translate(m, -W / 2 + ax, H - ay - ah, 0.5) : m;
  }

  // A bit bigger than the art box, so the figure towers over the card, as with voxels.
  const sw = W * 1.1;
  const sh = (sw * input.cutout.height) / input.cutout.width;
  const m = sculpt(input.cutout, input.depth, { columns: sw * FINE, width: sw, height: sh, inflate: opts.inflate, depth: opts.depth, oneSided: false });
  if (!m.pixels) return m;
  // It stands on the card (whose top is y = 1) over the middle of the art box.
  const standZ = ay + ah / 2 - H / 2;
  return translate(m, -(m.min[0] + m.max[0]) / 2, 1 - m.min[1], standZ - (m.min[2] + m.max[2]) / 2);
}
