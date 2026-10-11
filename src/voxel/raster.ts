/**
 * Pixel-grid helpers: area resampling, finding the art on the card, and cleaning up a cutout mask.
 */

/** RGBA pixels, row-major from the top-left (the layout of `ImageData`). */
export interface Pixels {
  width: number;
  height: number;
  data: Uint8ClampedArray | Uint8Array;
}

/** A rectangle. In source pixels for `resample`, normalised to 0–1 for `locateArt`. */
export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

interface Taps {
  /** First source index of each output cell's taps, into `idx`/`w`. */
  start: Int32Array;
  idx: Int32Array;
  w: Float32Array;
}

/**
 * Box-filter weights along one axis: output cell i covers [a, b) of the source, and each source cell
 * contributes in proportion to its overlap. Shrinking averages; enlarging repeats.
 */
function taps(from: number, length: number, n: number, limit: number): Taps {
  const start = new Int32Array(n + 1);
  const idx: number[] = [];
  const w: number[] = [];
  for (let i = 0; i < n; i++) {
    start[i] = idx.length;
    const a = from + (i * length) / n;
    const b = from + ((i + 1) * length) / n;
    let total = 0;
    const first = idx.length;
    for (let s = Math.floor(a); s < b; s++) {
      const overlap = Math.min(b, s + 1) - Math.max(a, s);
      if (overlap <= 0) continue;
      idx.push(Math.min(limit - 1, Math.max(0, s)));
      w.push(overlap);
      total += overlap;
    }
    for (let k = first; k < idx.length; k++) w[k] /= total;
  }
  start[n] = idx.length;
  return { start, idx: Int32Array.from(idx), w: Float32Array.from(w) };
}

/**
 * Area-averaged resize of `rect` (default: the whole image) to `outW`×`outH`. Colour is weighted by
 * alpha, so a cutout's edge doesn't pick up the colour of the background it was cut from.
 */
export function resample(px: Pixels, outW: number, outH: number, rect: Rect = { x: 0, y: 0, w: px.width, h: px.height }): Pixels {
  const tx = taps(rect.x, rect.w, outW, px.width);
  const ty = taps(rect.y, rect.h, outH, px.height);
  const out = new Uint8ClampedArray(outW * outH * 4);
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      let r = 0;
      let g = 0;
      let b = 0;
      let a = 0;
      for (let j = ty.start[y]; j < ty.start[y + 1]; j++) {
        const row = ty.idx[j] * px.width;
        for (let i = tx.start[x]; i < tx.start[x + 1]; i++) {
          const k = (row + tx.idx[i]) * 4;
          const wa = tx.w[i] * ty.w[j] * px.data[k + 3];
          r += px.data[k] * wa;
          g += px.data[k + 1] * wa;
          b += px.data[k + 2] * wa;
          a += wa;
        }
      }
      const o = (y * outW + x) * 4;
      if (a > 0) {
        out[o] = r / a;
        out[o + 1] = g / a;
        out[o + 2] = b / a;
      }
      // The weights sum to 1, so `a` is already the average alpha.
      out[o + 3] = a;
    }
  }
  return { width: outW, height: outH, data: out };
}

function gray(px: Pixels): Float32Array {
  const g = new Float32Array(px.width * px.height);
  for (let i = 0; i < g.length; i++) g[i] = (0.299 * px.data[i * 4] + 0.587 * px.data[i * 4 + 1] + 0.114 * px.data[i * 4 + 2]) / 255;
  return g;
}

/** Copies `mask` (one byte per pixel, 0–255) into the alpha channel of a copy of `px`. */
export function withAlpha(px: Pixels, mask: Uint8Array | Uint8ClampedArray): Pixels {
  const data = new Uint8ClampedArray(px.data);
  for (let i = 0; i < mask.length; i++) data[i * 4 + 3] = mask[i];
  return { width: px.width, height: px.height, data };
}

// ---------------------------------------------------------------------------
// Finding the art on the card
// ---------------------------------------------------------------------------

export interface ArtMatch {
  /** Where the art sits on the card, normalised to the card's size. */
  rect: Rect;
  /** Mean squared grey-level difference at the match, 0–1. A true match is well under 0.01. */
  error: number;
}

/** Mean squared difference of `t` (tw×th) laid on `c` (cw wide) at (x, y), giving up above `limit`. */
function ssd(c: Float32Array, cw: number, t: Float32Array, tw: number, th: number, x: number, y: number, limit: number): number {
  let sum = 0;
  const cap = limit * tw * th;
  for (let j = 0; j < th; j++) {
    const co = (y + j) * cw + x;
    const to = j * tw;
    for (let i = 0; i < tw; i++) {
      const d = c[co + i] - t[to + i];
      sum += d * d;
    }
    if (sum > cap) return Infinity;
  }
  return sum / (tw * th);
}

/**
 * Finds where the art crop sits on the card image. Every frame (old border, modern, borderless,
 * showcase) puts the art somewhere different, but Scryfall's art crop is cut from the same scan, so
 * template matching finds it: an exhaustive search over size and position on a 64-pixel-wide copy,
 * then two refinements at double the resolution, each searching a few pixels around the last answer.
 */
export function locateArt(card: Pixels, art: Pixels): ArtMatch | null {
  const aspect = art.height / art.width;
  let best: { x: number; y: number; w: number; err: number } | null = null;
  let scale = 0;

  for (const cw of [64, 128, 256]) {
    const ch = Math.round((cw * card.height) / card.width);
    const C = gray(resample(card, cw, ch));
    const prev = best;
    const k = scale ? cw / scale : 0;
    best = null;
    const wLo = prev ? Math.round(prev.w * k) - 2 : Math.round(cw * 0.45);
    const wHi = prev ? Math.round(prev.w * k) + 2 : cw;
    for (let tw = Math.max(4, wLo); tw <= Math.min(cw, wHi); tw++) {
      const th = Math.round(tw * aspect);
      if (th > ch || th < 3) continue;
      const T = gray(resample(art, tw, th));
      const xLo = prev ? Math.round(prev.x * k) - 3 : 0;
      const yLo = prev ? Math.round(prev.y * k) - 3 : 0;
      const xHi = prev ? xLo + 6 : cw - tw;
      const yHi = prev ? yLo + 6 : ch - th;
      for (let y = Math.max(0, yLo); y <= Math.min(ch - th, yHi); y++) {
        for (let x = Math.max(0, xLo); x <= Math.min(cw - tw, xHi); x++) {
          const err = ssd(C, cw, T, tw, th, x, y, best?.err ?? Infinity);
          if (err < (best?.err ?? Infinity)) best = { x, y, w: tw, err };
        }
      }
    }
    if (!best) return null;
    scale = cw;
    if (cw === 256) {
      return {
        rect: { x: best.x / cw, y: best.y / ch, w: best.w / cw, h: Math.round(best.w * aspect) / ch },
        error: best.err,
      };
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Mask clean-up
// ---------------------------------------------------------------------------

/**
 * Labels the 4-connected regions where `mask[i] === value`. Returns the label per cell (-1 elsewhere),
 * each region's size, and whether it touches the image border.
 */
function regions(mask: Uint8Array, w: number, h: number, value: number) {
  const label = new Int32Array(w * h).fill(-1);
  const sizes: number[] = [];
  const border: boolean[] = [];
  const stack: number[] = [];
  for (let s = 0; s < w * h; s++) {
    if (mask[s] !== value || label[s] >= 0) continue;
    const id = sizes.length;
    let size = 0;
    let edge = false;
    label[s] = id;
    stack.push(s);
    while (stack.length) {
      const i = stack.pop()!;
      size++;
      const x = i % w;
      const y = (i - x) / w;
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) edge = true;
      const visit = (j: number) => {
        if (mask[j] === value && label[j] < 0) {
          label[j] = id;
          stack.push(j);
        }
      };
      if (x > 0) visit(i - 1);
      if (x < w - 1) visit(i + 1);
      if (y > 0) visit(i - w);
      if (y < h - 1) visit(i + w);
    }
    sizes.push(size);
    border.push(edge);
  }
  return { label, sizes, border };
}

/**
 * Tidies a segmentation mask (1 = subject) in place. Specks the model left in the background become
 * single floating voxels, and pinholes in the subject become pits, so: drop subject regions smaller
 * than `minPart` of the largest one, and fill enclosed holes of up to `maxHole` cells.
 */
export function cleanMask(mask: Uint8Array, w: number, h: number, minPart = 0.02, maxHole = 6): Uint8Array {
  const fg = regions(mask, w, h, 1);
  const largest = Math.max(0, ...fg.sizes);
  const keep = fg.sizes.map((s) => s >= Math.max(4, largest * minPart));
  for (let i = 0; i < mask.length; i++) if (mask[i] && !keep[fg.label[i]]) mask[i] = 0;

  const bg = regions(mask, w, h, 0);
  for (let i = 0; i < mask.length; i++) {
    const r = bg.label[i];
    if (r >= 0 && !bg.border[r] && bg.sizes[r] <= maxHole) mask[i] = 1;
  }
  return mask;
}

/** Share of `mask` cells that are set. */
export function coverage(mask: Uint8Array): number {
  let n = 0;
  for (let i = 0; i < mask.length; i++) n += mask[i] ? 1 : 0;
  return mask.length ? n / mask.length : 0;
}
