/**
 * Card → voxels: the card itself as a slab, and the subject of its art inflated out of it.
 *
 * Two layouts:
 * - relief: the card stands upright and the subject bulges out of the art box towards the viewer.
 *   The back of the card is the card back (or, on a double-faced card, the other face).
 * - summon: the card lies on the table and the subject stands on it, full size and rounded on both
 *   sides, above the spot in the art it came from.
 *
 * Every model keeps the card's voxels first, in the same order whatever the subject, and records where
 * each subject voxel comes from on the card's surface. That lets the subject rise out of the card when
 * the cutout arrives after the card is already on screen.
 */

import { cleanMask, coverage, resample, type Pixels, type Rect } from './raster';
import { inflate } from './voxelize';

export type Layout = 'relief' | 'summon';

export interface CardInput {
  /** The whole card. */
  front: Pixels;
  /** What's on the back, as seen from behind (it gets mirrored); null for plain brown. */
  back: Pixels | null;
  /** The art crop with the subject opaque and the background transparent. */
  cutout: Pixels | null;
  /** Where the art crop sits on the card, normalised to 0–1. */
  art: Rect | null;
}

export interface CardOptions {
  layout: Layout;
  /** Card width in voxels. */
  width: number;
  /** Subject inflation: 0 = flat, 1 = round. */
  inflate: number;
}

export interface VoxelModel {
  count: number;
  /** Voxel centres, xyz. Y is up; the model stands on y = 0, centred on x. */
  positions: Float32Array;
  /** sRGB 0–1. */
  colors: Float32Array;
  /** The first `cardCount` voxels are the card, identical for every subject at the same size and layout. */
  cardCount: number;
  /** Per subject voxel (after the card's), the point on the card it rises from. */
  origins: Float32Array;
  min: [number, number, number];
  max: [number, number, number];
  /** Card size in voxels. */
  width: number;
  height: number;
  /** Subject pixels it was built from (0 = card only). */
  subjectPixels: number;
}

/** A cutout covering less than this share of the art has found nothing; more than this, everything. */
export const MIN_COVERAGE = 0.01;
export const MAX_COVERAGE = 0.9;

/** MTG cards are 63 × 88 mm with 3.2 mm corners. */
const CORNER = 3.2 / 63;
const BACK = [0.32, 0.22, 0.14];

class Builder {
  positions: number[] = [];
  colors: number[] = [];
  origins: number[] = [];

  push(x: number, y: number, z: number, r: number, g: number, b: number): void {
    this.positions.push(x, y, z);
    this.colors.push(r, g, b);
  }

  get count(): number {
    return this.positions.length / 3;
  }
}

/** Is card pixel (x, y) inside the rounded corners? */
export function insideCorners(x: number, y: number, w: number, h: number, r: number): boolean {
  // How far the pixel is past the straight part of each edge; only the corners have both.
  const dx = Math.max(0, r - x, x - (w - 1 - r));
  const dy = Math.max(0, r - y, y - (h - 1 - r));
  // The slack keeps the straight edges whole when rounding makes dx a hair over r.
  return dx * dx + dy * dy <= r * r + 1e-6;
}

/** The cutout as a 0/1 mask, cleaned, with the alpha of dropped pixels zeroed to match. */
function subjectMask(px: Pixels): Uint8Array {
  const mask = new Uint8Array(px.width * px.height);
  for (let i = 0; i < mask.length; i++) mask[i] = px.data[i * 4 + 3] >= 128 ? 1 : 0;
  cleanMask(mask, px.width, px.height);
  for (let i = 0; i < mask.length; i++) px.data[i * 4 + 3] = mask[i] ? 255 : 0;
  return mask;
}

export function buildCard(input: CardInput, opts: CardOptions): VoxelModel {
  const W = Math.max(8, Math.round(opts.width));
  const H = Math.round((W * input.front.height) / input.front.width);
  const face = resample(input.front, W, H);
  const back = input.back ? resample(input.back, W, H) : null;
  const radius = CORNER * W;
  const upright = opts.layout === 'relief';
  const b = new Builder();

  // World position of card pixel (column, row): upright facing +z, or lying face up with the top edge
  // away from the viewer.
  const wx = (cx: number) => cx - (W - 1) / 2;
  const wy = (cy: number) => H - cy - 0.5;
  const wz = (cy: number) => cy - (H - 1) / 2;

  for (let cy = 0; cy < H; cy++) {
    for (let cx = 0; cx < W; cx++) {
      if (!insideCorners(cx, cy, W, H, radius)) continue;
      const k = (cy * W + cx) * 4;
      const [r, g, bl] = [face.data[k] / 255, face.data[k + 1] / 255, face.data[k + 2] / 255];
      if (!upright) {
        b.push(wx(cx), 0.5, wz(cy), r, g, bl);
        continue;
      }
      b.push(wx(cx), wy(cy), 0, r, g, bl);
      // Seen from behind, the back's left is the front's right.
      const m = (cy * W + (W - 1 - cx)) * 4;
      if (back) b.push(wx(cx), wy(cy), -1, back.data[m] / 255, back.data[m + 1] / 255, back.data[m + 2] / 255);
      else b.push(wx(cx), wy(cy), -1, BACK[0], BACK[1], BACK[2]);
    }
  }
  const cardCount = b.count;
  let subjectPixels = 0;

  if (input.cutout && input.art) {
    const art = input.art;
    // The art box in card pixels.
    const ax = art.x * W;
    const ay = art.y * H;
    const aw = art.w * W;
    const ah = art.h * H;

    if (upright) {
      // Resample the cutout onto the card's own grid, so the relief lines up with the art under it.
      const x0 = Math.round(ax);
      const y0 = Math.round(ay);
      const w = Math.max(1, Math.round(ax + aw) - x0);
      const h = Math.max(1, Math.round(ay + ah) - y0);
      const sub = resample(input.cutout, w, h);
      const mask = subjectMask(sub);
      const c = coverage(mask);
      if (c >= MIN_COVERAGE && c <= MAX_COVERAGE) {
        const v = inflate(sub, { inflate: opts.inflate, oneSided: true });
        subjectPixels = v.pixels;
        for (let i = 0; i < v.count; i++) {
          const cx = x0 + v.positions[i * 3];
          const cy = y0 + v.positions[i * 3 + 1];
          if (!insideCorners(cx, cy, W, H, radius)) continue;
          b.push(wx(cx), wy(cy), 1 + v.positions[i * 3 + 2], v.colors[i * 3], v.colors[i * 3 + 1], v.colors[i * 3 + 2]);
          b.origins.push(wx(cx), wy(cy), 0);
        }
      }
    } else {
      // A bit bigger than the art box, so the figure towers over the card.
      const sw = Math.round(W * 1.1);
      const sh = Math.max(1, Math.round((sw * input.cutout.height) / input.cutout.width));
      const sub = resample(input.cutout, sw, sh);
      const mask = subjectMask(sub);
      const c = coverage(mask);
      if (c >= MIN_COVERAGE && c <= MAX_COVERAGE) {
        const v = inflate(sub, { inflate: opts.inflate });
        subjectPixels = v.pixels;
        let minX = Infinity;
        let maxX = -Infinity;
        let maxY = -Infinity;
        for (let i = 0; i < v.count; i++) {
          minX = Math.min(minX, v.positions[i * 3]);
          maxX = Math.max(maxX, v.positions[i * 3]);
          maxY = Math.max(maxY, v.positions[i * 3 + 1]);
        }
        const midX = (minX + maxX) / 2;
        // It stands over the middle of the art box.
        const standZ = wz(ay + ah / 2);
        for (let i = 0; i < v.count; i++) {
          const gx = v.positions[i * 3];
          const gy = v.positions[i * 3 + 1];
          b.push(gx - midX, maxY - gy + 1.5, standZ + v.positions[i * 3 + 2], v.colors[i * 3], v.colors[i * 3 + 1], v.colors[i * 3 + 2]);
          // It rises from where that pixel is painted on the card.
          b.origins.push(wx(ax + (gx * aw) / sw), 0.5, wz(ay + (gy * ah) / sh));
        }
      }
    }
  }

  const positions = Float32Array.from(b.positions);
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      min[k] = Math.min(min[k], positions[i + k] - 0.5);
      max[k] = Math.max(max[k], positions[i + k] + 0.5);
    }
  }

  return {
    count: b.count,
    positions,
    colors: Float32Array.from(b.colors),
    cardCount,
    origins: Float32Array.from(b.origins),
    min,
    max,
    width: W,
    height: H,
    subjectPixels,
  };
}

/**
 * The start of the rise: the card as it is, and every subject voxel shrunk to nothing at its origin
 * on the card. Paired index for index with `model`.
 */
export function riseStart(model: VoxelModel): { count: number; positions: Float32Array; colors: Float32Array; scales: Float32Array } {
  const positions = new Float32Array(model.positions);
  positions.set(model.origins, model.cardCount * 3);
  const scales = new Float32Array(model.count);
  scales.fill(1, 0, model.cardCount);
  return { count: model.count, positions, colors: model.colors, scales };
}
