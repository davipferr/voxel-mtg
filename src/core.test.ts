import { describe, expect, it } from 'vitest';

import { parseHash, parseSymbols, stepNumber, toCard } from './data/scryfall';
import { planTransition, restingPlan, sampleAt, seed, snapshotOf, spatialOrder, spring, type Snapshot } from './scene/morph';
import { buildCard, insideCorners, riseStart } from './voxel/card';
import { cleanMask, coverage, locateArt, resample, type Pixels } from './voxel/raster';
import { distanceTransform, halfThickness, inflate } from './voxel/voxelize';

/** An RGBA image from rows of characters: '.' is transparent, any other letter is an opaque colour. */
function sprite(rows: string[], palette: Record<string, [number, number, number]> = {}): Pixels {
  const height = rows.length;
  const width = rows[0].length;
  const data = new Uint8ClampedArray(width * height * 4);
  rows.forEach((row, y) =>
    [...row].forEach((ch, x) => {
      if (ch === '.') return;
      const [r, g, b] = palette[ch] ?? [255, 0, 0];
      data.set([r, g, b, 255], (y * width + x) * 4);
    }),
  );
  return { width, height, data };
}

/** An opaque image whose pixels come from `f(x, y)` → [r, g, b]. */
function image(width: number, height: number, f: (x: number, y: number) => [number, number, number]): Pixels {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) data.set([...f(x, y), 255], (y * width + x) * 4);
  return { width, height, data };
}

/** Seeded PRNG (mulberry32), so planning is deterministic in tests. */
function rng(seedValue: number) {
  let a = seedValue;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const keyOf = (s: Snapshot, i: number) => `${s.positions[i * 3]},${s.positions[i * 3 + 1]},${s.positions[i * 3 + 2]}`;

// ---------------------------------------------------------------------------

describe('distance transform', () => {
  it('matches brute force', () => {
    const w = 9;
    const h = 7;
    const mask = new Uint8Array(w * h);
    // An irregular blob, empty around the edges.
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) if ((x * 7 + y * 3) % 5 !== 0) mask[y * w + x] = 1;
    const d = distanceTransform(mask, w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let best = Infinity;
        for (let yy = 0; yy < h; yy++)
          for (let xx = 0; xx < w; xx++) if (!mask[yy * w + xx]) best = Math.min(best, Math.hypot(x - xx, y - yy));
        expect(d[y * w + x]).toBeCloseTo(best, 5);
      }
    }
  });

  it('is zero on empty cells', () => {
    const d = distanceTransform(new Uint8Array(4), 2, 2);
    expect([...d]).toEqual([0, 0, 0, 0]);
  });
});

describe('inflate', () => {
  const at = (v: { positions: Float32Array }, i: number) => [v.positions[i * 3], v.positions[i * 3 + 1], v.positions[i * 3 + 2]];

  it('makes one voxel per pixel when flat, in grid space', () => {
    const v = inflate(sprite(['.xx.', 'xxxx', '.xx.']), { inflate: 0 });
    expect(v.count).toBe(8);
    expect(v.pixels).toBe(8);
    expect(v.maxHalf).toBe(0);
    for (let i = 0; i < v.count; i++) expect(at(v, i)[2]).toBe(0);
    // Row 0 is the top row: its pixels are columns 1 and 2.
    const top = [...Array(v.count)].map((_, i) => at(v, i)).filter((p) => p[1] === 0).map((p) => p[0]);
    expect(top.sort()).toEqual([1, 2]);
  });

  it('keeps the pixel colours', () => {
    const v = inflate(sprite(['a'], { a: [255, 128, 0] }), { inflate: 0 });
    // A lone voxel has no neighbours, so no occlusion.
    expect(v.colors[0]).toBeCloseTo(1);
    expect(v.colors[1]).toBeCloseTo(128 / 255);
    expect(v.colors[2]).toBeCloseTo(0);
  });

  const square = sprite(Array.from({ length: 15 }, () => 'x'.repeat(15)));

  it('two-sided: thicker in the middle than at the edge, and symmetric front to back', () => {
    const v = inflate(square, { inflate: 1 });
    let centre = 0;
    let corner = 0;
    let front = 0;
    let back = 0;
    for (let i = 0; i < v.count; i++) {
      const [x, y, z] = at(v, i);
      if (x === 7 && y === 7) centre = Math.max(centre, z);
      if (x === 0 && y === 14) corner = Math.max(corner, z);
      if (z > 0) front++;
      if (z < 0) back++;
    }
    expect(centre).toBe(halfThickness(8, 1));
    expect(corner).toBe(halfThickness(1, 1));
    expect(centre).toBeGreaterThan(corner);
    expect(front).toBe(back);
    expect(v.maxHalf).toBe(centre);
  });

  it('one-sided: grows only towards +z, from z = 0', () => {
    const v = inflate(square, { inflate: 1, oneSided: true });
    let top = 0;
    for (let i = 0; i < v.count; i++) {
      const [x, y, z] = at(v, i);
      expect(z).toBeGreaterThanOrEqual(0);
      if (x === 7 && y === 7) top = Math.max(top, z);
    }
    expect(top).toBe(v.maxHalf);
    expect(top).toBeGreaterThan(halfThickness(8, 1));
  });

  it('hollows out the inside', () => {
    const v = inflate(square, { inflate: 1 });
    const set = new Set([...Array(v.count)].map((_, i) => at(v, i).join()));
    // The centre column's middle voxel is buried.
    expect(set.has('7,7,0')).toBe(false);
    expect(set.has(`7,7,${halfThickness(8, 1)}`)).toBe(true);
  });

  it('half-thickness grows with distance and inflation', () => {
    expect(halfThickness(1, 0)).toBe(0);
    expect(halfThickness(9, 1)).toBeGreaterThan(halfThickness(2, 1));
    expect(halfThickness(9, 1.5)).toBeGreaterThan(halfThickness(9, 1));
  });

  it('is empty for a transparent image', () => {
    expect(inflate(sprite(['..', '..']), { inflate: 1 }).count).toBe(0);
  });
});

describe('resample', () => {
  it('averages blocks when shrinking', () => {
    const px = image(4, 4, (x, y) => [x < 2 ? 0 : 200, y < 2 ? 0 : 100, (x + y) % 2 ? 40 : 0]);
    const out = resample(px, 2, 2);
    expect([...out.data.slice(0, 4)]).toEqual([0, 0, 20, 255]);
    expect([...out.data.slice(12, 16)]).toEqual([200, 100, 20, 255]);
  });

  it("weights colour by alpha, so a cutout's edge keeps the subject's colour", () => {
    const px = { width: 2, height: 1, data: Uint8ClampedArray.of(255, 0, 0, 255, 0, 0, 255, 0) };
    const out = resample(px, 1, 1);
    expect([...out.data]).toEqual([255, 0, 0, 128]);
  });

  it('crops a rectangle', () => {
    const px = image(6, 4, (x, y) => [x * 10, y * 10, 0]);
    const out = resample(px, 2, 2, { x: 3, y: 1, w: 2, h: 2 });
    expect([out.data[0], out.data[1]]).toEqual([30, 10]);
    expect([out.data[12], out.data[13]]).toEqual([40, 20]);
  });
});

describe('locateArt', () => {
  // A dark frame around a painted art box and a pale text box, like a real card.
  const W = 244;
  const H = 340;
  const ART = { x: 20, y: 38, w: 204, h: 150 };
  const paint = (x: number, y: number): [number, number, number] => [
    128 + 60 * Math.sin(x * 0.07 + y * 0.031) + 40 * Math.sin(y * 0.13),
    128 + 70 * Math.sin(y * 0.05 - x * 0.09),
    128 + 50 * Math.cos(x * 0.04 + y * 0.08) + 30 * Math.sin(x * 0.17),
  ];
  const card = image(W, H, (x, y) => {
    if (x >= ART.x && x < ART.x + ART.w && y >= ART.y && y < ART.y + ART.h) return paint(x, y);
    if (x >= 20 && x < 224 && y >= 200 && y < 310) return [225, 220, 205];
    return [24, 22, 20];
  });

  it('finds the art crop on the card', () => {
    // Scryfall's art crops are cut from a bigger scan than the card image, so scale it up.
    const art = resample(card, 306, 225, ART);
    const m = locateArt(card, art)!;
    expect(m.error).toBeLessThan(0.005);
    expect(Math.abs(m.rect.x * W - ART.x)).toBeLessThan(2.5);
    expect(Math.abs(m.rect.y * H - ART.y)).toBeLessThan(2.5);
    expect(Math.abs(m.rect.w * W - ART.w)).toBeLessThan(3);
    expect(Math.abs(m.rect.h * H - ART.h)).toBeLessThan(3);
  });

  it('reports a poor fit for art that is not on the card', () => {
    const other = image(300, 220, (x, y) => [(x * 37 + y * 11) % 256, (x * 5 + y * 71) % 256, (x * y) % 256]);
    expect(locateArt(card, other)!.error).toBeGreaterThan(0.02);
  });
});

describe('cleanMask', () => {
  const grid = (rows: string[]) => Uint8Array.from(rows.join('').split(''), (c) => (c === '#' ? 1 : 0));
  const show = (m: Uint8Array, w: number) =>
    Array.from({ length: m.length / w }, (_, y) => [...m.slice(y * w, y * w + w)].map((v) => (v ? '#' : '.')).join(''));

  it('drops specks and fills pinholes, but keeps real gaps', () => {
    const rows = [
      '#.............',
      '..##########..',
      '..##.#######..',
      '..##########..',
      '..###....###..',
      '..###....###..',
      '..###....###..',
      '..##########..',
      '..............',
    ];
    const m = cleanMask(grid(rows), 14, 9, 0.05, 6);
    expect(show(m, 14)).toEqual([
      '..............',
      '..##########..',
      '..##########..',
      '..##########..',
      '..###....###..',
      '..###....###..',
      '..###....###..',
      '..##########..',
      '..............',
    ]);
  });

  it('measures coverage', () => {
    expect(coverage(Uint8Array.of(1, 0, 0, 1))).toBe(0.5);
    expect(coverage(new Uint8Array())).toBe(0);
  });
});

describe('buildCard', () => {
  const front = image(40, 56, (x) => (x < 20 ? [255, 0, 0] : [0, 0, 255]));
  // A round blob in the middle of a transparent art crop.
  const cutout = (radius: number) => {
    const px = image(60, 44, () => [0, 200, 0]);
    for (let y = 0; y < 44; y++) for (let x = 0; x < 60; x++) px.data[(y * 60 + x) * 4 + 3] = Math.hypot(x - 30, y - 22) <= radius ? 255 : 0;
    return px;
  };
  const art = { x: 0.1, y: 0.1, w: 0.8, h: 0.45 };
  const opts = { width: 20, inflate: 1 } as const;
  const zs = (m: { positions: Float32Array; count: number }, from = 0) => [...Array(m.count - from)].map((_, i) => m.positions[(from + i) * 3 + 2]);

  it('relief: an upright two-layer slab with rounded corners', () => {
    const m = buildCard({ front, back: null, cutout: null, art: null }, { ...opts, layout: 'relief' });
    expect(m.width).toBe(20);
    expect(m.height).toBe(28);
    expect(m.cardCount).toBe(m.count);
    expect(m.count % 2).toBe(0);
    // The corners are cut off.
    expect(m.count / 2).toBeLessThan(20 * 28);
    expect(new Set(zs(m))).toEqual(new Set([0, -1]));
    expect(m.min[1]).toBe(0);
    expect(m.max[1]).toBe(28);
  });

  it('rounds the corners but keeps every straight edge whole', () => {
    // 120 × 166 with a 6.095-voxel radius once lost its right column and bottom row to rounding.
    const r = (3.2 / 63) * 120;
    for (let y = 8; y < 158; y++) expect(insideCorners(0, y, 120, 166, r) && insideCorners(119, y, 120, 166, r)).toBe(true);
    for (let x = 8; x < 112; x++) expect(insideCorners(x, 0, 120, 166, r) && insideCorners(x, 165, 120, 166, r)).toBe(true);
    expect(insideCorners(0, 0, 120, 166, r)).toBe(false);
    expect(insideCorners(119, 165, 120, 166, r)).toBe(false);
  });

  it('relief: the back is mirrored, as seen from behind', () => {
    const m = buildCard({ front, back: front, cutout: null, art: null }, { ...opts, layout: 'relief' });
    for (let i = 0; i < m.count; i++) {
      if (m.positions[i * 3 + 2] !== -1) continue;
      const left = m.positions[i * 3] < 0;
      // The front is red on the left; from behind, the left of the back shows its blue right side.
      expect(m.colors[i * 3 + 2]).toBe(left ? 1 : 0);
    }
  });

  it('summon: a single layer lying flat', () => {
    const m = buildCard({ front, back: null, cutout: null, art: null }, { ...opts, layout: 'summon' });
    for (let i = 0; i < m.count; i++) expect(m.positions[i * 3 + 1]).toBe(0.5);
  });

  it('relief: the subject bulges out of the card, after the same card voxels', () => {
    const bare = buildCard({ front, back: null, cutout: null, art }, { ...opts, layout: 'relief' });
    const m = buildCard({ front, back: null, cutout: cutout(14), art }, { ...opts, layout: 'relief' });
    expect(m.cardCount).toBe(bare.count);
    expect([...m.positions.slice(0, bare.count * 3)]).toEqual([...bare.positions]);
    expect(m.subjectPixels).toBeGreaterThan(0);
    expect(m.origins.length).toBe((m.count - m.cardCount) * 3);
    expect(Math.min(...zs(m, m.cardCount))).toBe(1);
    expect(m.max[2]).toBeGreaterThan(2);
    for (let i = 0; i < m.count - m.cardCount; i++) expect(m.origins[i * 3 + 2]).toBe(0);
  });

  it('summon: the subject stands on the card and rises from the card surface', () => {
    const m = buildCard({ front, back: null, cutout: cutout(14), art }, { ...opts, layout: 'summon' });
    expect(m.count).toBeGreaterThan(m.cardCount);
    for (let i = m.cardCount; i < m.count; i++) expect(m.positions[i * 3 + 1]).toBeGreaterThanOrEqual(1.5);
    for (let i = 0; i < m.count - m.cardCount; i++) expect(m.origins[i * 3 + 1]).toBe(0.5);
  });

  it('skips a cutout that found nothing, or everything', () => {
    for (const radius of [0.5, 100]) {
      const m = buildCard({ front, back: null, cutout: cutout(radius), art }, { ...opts, layout: 'relief' });
      expect(m.subjectPixels).toBe(0);
      expect(m.count).toBe(m.cardCount);
    }
  });

  it('rises: the subject starts invisible on the card and ends in place', () => {
    const m = buildCard({ front, back: null, cutout: cutout(14), art }, { ...opts, layout: 'summon' });
    const start = riseStart(m);
    expect(start.count).toBe(m.count);
    expect(start.scales[0]).toBe(1);
    expect(start.scales[m.count - 1]).toBe(0);
    expect([...start.positions.slice(m.cardCount * 3)]).toEqual([...m.origins]);

    const plan = planTransition(start, snapshotOf(m), 'rise', rng(4));
    // Paired index for index; the card doesn't move or wait.
    expect([...plan.to]).toEqual([...m.positions]);
    for (let i = 0; i < m.cardCount; i++) expect(plan.delay[i]).toBe(0);
    expect(sampleAt(plan, plan.end + 1).count).toBe(m.count);
  });
});


describe('spring', () => {
  it('starts at 0, ends at exactly 1, and overshoots in between', () => {
    expect(spring(0)).toBe(0);
    expect(spring(1)).toBe(1);
    let peak = 0;
    for (let t = 0; t <= 1; t += 0.01) peak = Math.max(peak, spring(t));
    expect(peak).toBeGreaterThan(1.05);
    expect(peak).toBeLessThan(1.25);
    expect(Math.abs(spring(0.999) - 1)).toBeLessThan(0.01);
  });
});

describe('transition planning', () => {
  const model = (n: number, colour = 1): Snapshot =>
    snapshotOf({
      count: n,
      positions: Float32Array.from({ length: n * 3 }, (_, i) => (i % 3 === 1 ? Math.floor(i / 3) : (i * 7) % 5)),
      colors: new Float32Array(n * 3).fill(colour),
    });

  for (const [n, m] of [
    [10, 10],
    [4, 13],
    [13, 4],
    [1, 50],
  ]) {
    it(`uses every voxel of both sides (${n} → ${m})`, () => {
      const a = model(n);
      const b = model(m);
      const plan = planTransition(a, b, 'morph', rng(1));
      expect(plan.count).toBe(Math.max(n, m));
      // Each target voxel is landed on by exactly one visible voxel.
      const landed = new Map<string, number>();
      for (let i = 0; i < plan.count; i++) {
        if (plan.scale[i * 2 + 1] === 0) continue;
        const k = `${plan.to[i * 3]},${plan.to[i * 3 + 1]},${plan.to[i * 3 + 2]}`;
        landed.set(k, (landed.get(k) ?? 0) + 1);
      }
      expect(landed.size).toBe(m);
      expect([...landed.values()].every((c) => c === 1)).toBe(true);
      // And each source voxel leaves visibly exactly once.
      expect([...plan.scale].filter((_, i) => i % 2 === 0 && plan.scale[i] === 1).length).toBe(n);
    });
  }

  it('ends exactly at the target, whatever the mode', () => {
    const b = model(20, 0.5);
    for (const mode of ['explode', 'morph', 'recolor'] as const) {
      const plan = planTransition(model(30), b, mode, rng(7));
      const end = sampleAt(plan, plan.end + 1);
      expect(end.count).toBe(20);
      const want = new Set([...Array(20)].map((_, i) => keyOf(b, i)));
      const got = new Set([...Array(end.count)].map((_, i) => keyOf(end, i)));
      expect(got).toEqual(want);
      expect([...end.colors].every((c) => Math.abs(c - 0.5) < 1e-6)).toBe(true);
    }
  });

  it('starts exactly at the source', () => {
    const a = model(12);
    const plan = planTransition(a, model(30), 'explode', rng(3));
    const start = sampleAt(plan, 0);
    expect(start.count).toBe(12);
    expect(new Set([...Array(12)].map((_, i) => keyOf(start, i)))).toEqual(new Set([...Array(12)].map((_, i) => keyOf(a, i))));
  });

  it('keeps every voxel in place when only the colour changes', () => {
    const a = model(25, 0.2);
    const b = model(25, 0.9);
    const plan = planTransition(a, b, 'recolor', rng(5));
    for (let i = 0; i < plan.count; i++) {
      for (let k = 0; k < 3; k++) expect(plan.from[i * 3 + k]).toBe(plan.to[i * 3 + k]);
    }
    // Mid-way, positions haven't moved.
    const mid = sampleAt(plan, 0.4);
    for (let i = 0; i < mid.count; i++) expect(keyOf(mid, i)).toBe(keyOf(a, spatialOrder(a)[i]));
  });

  it('grows the first model out of a single seed', () => {
    const plan = planTransition(seed(0, 0.5, 0), model(8), 'explode', rng(2));
    expect(sampleAt(plan, 0).count).toBe(0);
    expect(sampleAt(plan, plan.end).count).toBe(8);
  });

  it('a resting plan holds still', () => {
    const b = model(9);
    const plan = restingPlan(b);
    const s = sampleAt(plan, 0);
    for (let i = 0; i < 9; i++) expect(keyOf(s, i)).toBe(keyOf(b, i));
  });
});


describe('rise planning', () => {
  it('falls back to a morph when the two sides are not paired', () => {
    const a = snapshotOf({ count: 3, positions: new Float32Array(9), colors: new Float32Array(9) });
    const b = snapshotOf({ count: 5, positions: new Float32Array(15), colors: new Float32Array(15) });
    expect(planTransition(a, b, 'rise', rng(1)).count).toBe(5);
  });
});

describe('Scryfall helpers', () => {
  it('splits mana symbols out of rules text', () => {
    expect(parseSymbols('{T}: Add {G}.')).toEqual([{ symbol: '{T}' }, { text: ': Add ' }, { symbol: '{G}' }, { text: '.' }]);
    expect(parseSymbols('Flying')).toEqual([{ text: 'Flying' }]);
    expect(parseSymbols('')).toEqual([]);
  });

  it('reads printings and names from the URL hash', () => {
    expect(parseHash('#fdn/763')).toEqual({ set: 'fdn', number: '763' });
    expect(parseHash('#plst/MH1-21')).toEqual({ set: 'plst', number: 'MH1-21' });
    expect(parseHash('#Shivan%20Dragon')).toEqual({ name: 'Shivan Dragon' });
    expect(parseHash('')).toBeNull();
  });

  it('steps plain collector numbers only', () => {
    expect(stepNumber('12', 1)).toBe('13');
    expect(stepNumber('12', -1)).toBe('11');
    expect(stepNumber('1', -1)).toBeNull();
    expect(stepNumber('12a', 1)).toBeNull();
  });

  const images = (tag: string) => ({ small: `${tag}-s.jpg`, normal: `${tag}.jpg`, art_crop: `${tag}-art.jpg` });
  const base = {
    id: 'x',
    set: 'tst',
    set_name: 'Test',
    collector_number: '1',
    rarity: 'rare',
    released_at: '2020-01-01',
    color_identity: ['G'],
    scryfall_uri: 'https://scryfall.com/x',
    artist: 'Someone',
  };

  it('flattens a single-faced card', () => {
    const c = toCard({ ...base, name: 'Bear', layout: 'normal', type_line: 'Creature — Bear', power: '2', toughness: '2', image_uris: images('bear') });
    expect(c.faces).toHaveLength(1);
    expect(c.twoSided).toBe(false);
    expect(c.faces[0].image).toEqual({ card: 'bear.jpg', art: 'bear-art.jpg', small: 'bear-s.jpg' });
    expect(c.faces[0].power).toBe('2');
    expect(c.faces[0].artist).toBe('Someone');
    expect(c.cardBackId).toBe('0aeebaf5-8c7d-4636-9e82-8c27447861f7');
  });

  it('gives each side of a double-faced card its own image', () => {
    const c = toCard({
      ...base,
      name: 'Day // Night',
      layout: 'transform',
      card_faces: [
        { name: 'Day', image_uris: images('day') },
        { name: 'Night', image_uris: images('night'), artist: 'Other' },
      ],
    });
    expect(c.twoSided).toBe(true);
    expect(c.faces.map((f) => f.image?.card)).toEqual(['day.jpg', 'night.jpg']);
    expect(c.faces[1].artist).toBe('Other');
  });

  it('shares the one image between the halves of an adventure', () => {
    const c = toCard({
      ...base,
      name: 'Giant // Stomp',
      layout: 'adventure',
      image_uris: images('giant'),
      card_faces: [{ name: 'Giant' }, { name: 'Stomp' }],
    });
    expect(c.twoSided).toBe(false);
    expect(c.faces.map((f) => f.image?.card)).toEqual(['giant.jpg', 'giant.jpg']);
  });
});
