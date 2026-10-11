/**
 * Cutout → voxels.
 *
 * A cutout is a flat picture, so depth has to be invented. Each opaque pixel becomes a column of
 * voxels whose thickness grows with its distance from the silhouette's edge: the outline stays thin,
 * the middle of the body bulges, and a thin limb stays thin because nothing in it is far from an edge.
 * This is "inflation", the same idea sketch-to-3D tools use, done here on a pixel grid.
 *
 * Two-sided inflation bulges both ways, for a free-standing figure. One-sided inflation bulges only
 * towards the viewer, for a relief that sits on the card.
 */

import type { Pixels } from './raster';

export interface InflateOptions {
  /** 0 = flat (one layer), 1 = fully rounded. */
  inflate: number;
  /** Bulge towards +z only; z = 0 is the bottom layer. Otherwise symmetric about z = 0. */
  oneSided?: boolean;
  /** Pixels with alpha below this are empty. */
  alphaThreshold?: number;
}

/** Voxels in grid space: x is the pixel column, y the pixel row (0 at the top), z the depth. */
export interface GridVoxels {
  count: number;
  positions: Float32Array;
  /** sRGB 0–1, with ambient occlusion baked in. */
  colors: Float32Array;
  /** Opaque pixels it was built from. */
  pixels: number;
  /** Tallest column, in voxels from z = 0. */
  maxHalf: number;
}

/** Half-thickness, in voxels, of a pixel `dist` pixels from the nearest empty one, at full inflation. */
const DEPTH = 3;

export function halfThickness(dist: number, inflate: number): number {
  // sqrt gives a dome: steep at the outline, flat across the middle. Edge pixels have dist = 1.
  return Math.round(inflate * DEPTH * Math.sqrt(Math.max(0, dist - 0.5)));
}

// ---------------------------------------------------------------------------
// Euclidean distance transform
// ---------------------------------------------------------------------------

const FAR = 1e20;

/**
 * Felzenszwalb & Huttenlocher's 1D squared distance transform: the lower envelope of the parabolas
 * rooted at each cell, built left to right in one pass. `f` holds 0 for "source" cells and FAR
 * elsewhere; `d` receives squared distances. Linear time.
 */
function edt1d(f: Float64Array, n: number, d: Float64Array, v: Int32Array, z: Float64Array): void {
  let k = 0;
  v[0] = 0;
  z[0] = -Infinity;
  z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    let s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) {
      k--;
      s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
  }
}

/**
 * Exact Euclidean distance from every cell to the nearest empty cell (mask = 0). The 2D transform
 * separates: run the 1D one down every column, then along every row of the result. The grid must
 * contain at least one empty cell.
 */
export function distanceTransform(mask: Uint8Array, w: number, h: number): Float32Array {
  const n = Math.max(w, h);
  const f = new Float64Array(n);
  const d = new Float64Array(n);
  const v = new Int32Array(n);
  const z = new Float64Array(n + 1);
  const grid = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) grid[i] = mask[i] ? FAR : 0;

  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) f[y] = grid[y * w + x];
    edt1d(f, h, d, v, z);
    for (let y = 0; y < h; y++) grid[y * w + x] = d[y];
  }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) f[x] = grid[y * w + x];
    edt1d(f, w, d, v, z);
    for (let x = 0; x < w; x++) out[y * w + x] = Math.sqrt(d[x]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Inflate
// ---------------------------------------------------------------------------

/** Ambient occlusion from how many of the 26 neighbours are solid: creases darken, ridges stay bright. */
function occlusion(solidNeighbours: number): number {
  return Math.min(1, Math.max(0.62, 1.2 - solidNeighbours * 0.022));
}

export function inflate(px: Pixels, opts: InflateOptions): GridVoxels {
  const threshold = opts.alphaThreshold ?? 128;
  const oneSided = !!opts.oneSided;

  // Pad with a one-pixel empty border, so the distance transform sees the edge of the image as empty.
  const W = px.width + 2;
  const H = px.height + 2;
  const mask = new Uint8Array(W * H);
  let pixels = 0;
  for (let y = 0; y < px.height; y++) {
    for (let x = 0; x < px.width; x++) {
      if (px.data[(y * px.width + x) * 4 + 3] >= threshold) {
        mask[(y + 1) * W + x + 1] = 1;
        pixels++;
      }
    }
  }
  if (!pixels) return { count: 0, positions: new Float32Array(), colors: new Float32Array(), pixels: 0, maxHalf: 0 };

  const dist = distanceTransform(mask, W, H);
  // Height per column; -1 = no column. A one-sided relief doesn't need to be as tall as a figure is
  // thick, since it only has one side.
  const half = new Int16Array(W * H).fill(-1);
  let maxHalf = 0;
  for (let i = 0; i < W * H; i++) {
    if (!mask[i]) continue;
    half[i] = halfThickness(dist[i], oneSided ? opts.inflate * 1.4 : opts.inflate);
    if (half[i] > maxHalf) maxHalf = half[i];
  }

  const columnHalf = (x: number, y: number) => (x < 0 || y < 0 || x >= W || y >= H ? -1 : half[y * W + x]);
  // Under a relief is the card, which counts as solid for the occlusion.
  const solid = (x: number, y: number, z: number) => (oneSided && z < 0) || (columnHalf(x, y) >= Math.abs(z) && (!oneSided || z >= 0));

  const positions: number[] = [];
  const colors: number[] = [];
  for (let y = 1; y <= px.height; y++) {
    for (let x = 1; x <= px.width; x++) {
      const h = half[y * W + x];
      if (h < 0) continue;
      const src = ((y - 1) * px.width + x - 1) * 4;
      const r = px.data[src] / 255;
      const g = px.data[src + 1] / 255;
      const b = px.data[src + 2] / 255;
      // `|| 0` turns the -0 of a flat column (z = -h with h = 0) into 0.
      for (let z = oneSided ? 0 : -h || 0; z <= h; z++) {
        const az = Math.abs(z);
        // Keep only voxels with an exposed face. The inside would never be seen, and dropping it cuts
        // the count by 3–5×.
        const exposed =
          az === h ||
          columnHalf(x - 1, y) < az ||
          columnHalf(x + 1, y) < az ||
          columnHalf(x, y - 1) < az ||
          columnHalf(x, y + 1) < az;
        if (!exposed) continue;

        let n = 0;
        for (let dz = -1; dz <= 1; dz++)
          for (let dy = -1; dy <= 1; dy++)
            for (let dx = -1; dx <= 1; dx++) if ((dx || dy || dz) && solid(x + dx, y + dy, z + dz)) n++;
        const ao = occlusion(n);

        positions.push(x - 1, y - 1, z);
        colors.push(r * ao, g * ao, b * ao);
      }
    }
  }

  return { count: positions.length / 3, positions: Float32Array.from(positions), colors: Float32Array.from(colors), pixels, maxHalf };
}
