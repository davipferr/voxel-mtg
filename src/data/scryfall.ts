/**
 * Scryfall: https://scryfall.com/docs/api
 *
 * Free, keyless and CORS-enabled, and so are the card images on cards.scryfall.io and the backs on
 * backs.scryfall.io, which is what lets the canvas read their pixels. Scryfall asks for 50–100 ms
 * between requests and for clients to cache, so requests go out one at a time, 100 ms apart, and
 * every response is memoised for the session.
 */

const API = 'https://api.scryfall.com';

/** The ordinary Magic card back. */
export const DEFAULT_BACK_ID = '0aeebaf5-8c7d-4636-9e82-8c27447861f7';

export const backUrl = (id: string) => `https://backs.scryfall.io/normal/${id[0]}/${id[1]}/${id}.jpg`;

export const symbolUrl = (symbol: string) => `https://svgs.scryfall.io/card-symbols/${symbol.replace(/[{}/]/g, '')}.svg`;

/** Accent colours by colour identity. */
export const COLOR_ACCENTS: Record<string, string> = {
  W: '#e9dfb4',
  U: '#3d8fd6',
  B: '#9a7fb0',
  R: '#e0543a',
  G: '#3fae66',
  multi: '#d8b247',
  colorless: '#a9a9b5',
};

export const COLOR_NAMES: Record<string, string> = { W: 'White', U: 'Blue', B: 'Black', R: 'Red', G: 'Green' };

export function accentFor(colors: string[]): string {
  if (colors.length === 0) return COLOR_ACCENTS.colorless;
  if (colors.length > 1) return COLOR_ACCENTS.multi;
  return COLOR_ACCENTS[colors[0]] ?? COLOR_ACCENTS.colorless;
}

export interface Face {
  name: string;
  manaCost: string;
  typeLine: string;
  oracleText: string;
  flavorText: string;
  power: string | null;
  toughness: string | null;
  loyalty: string | null;
  defense: string | null;
  artist: string;
  colors: string[];
  /** Null when the card has no image (rare: some unreleased cards). */
  image: { card: string; art: string; small: string } | null;
}

export interface Card {
  id: string;
  name: string;
  set: string;
  setName: string;
  collectorNumber: string;
  rarity: string;
  releasedAt: string;
  layout: string;
  colorIdentity: string[];
  faces: Face[];
  /** True when each face is printed on its own side (transform, modal DFC…), so there's a back face to show. */
  twoSided: boolean;
  cardBackId: string;
  scryfallUri: string;
}

// ---------------------------------------------------------------------------
// Fetching
// ---------------------------------------------------------------------------

export class ScryfallError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

const cache = new Map<string, Promise<unknown>>();
let queue: Promise<unknown> = Promise.resolve();

/** One request at a time, 100 ms apart. */
function throttled<T>(task: () => Promise<T>): Promise<T> {
  const run = queue.then(task, task);
  queue = run.catch(() => {}).then(() => new Promise((r) => setTimeout(r, 100)));
  return run;
}

async function request<T>(path: string): Promise<T> {
  const r = await throttled(() => fetch(`${API}${path}`, { headers: { Accept: 'application/json' } }));
  const body = (await r.json().catch(() => null)) as { object?: string; details?: string } | null;
  if (!r.ok || body?.object === 'error') {
    throw new ScryfallError(body?.details ?? `Scryfall answered ${r.status}.`, r.status);
  }
  return body as T;
}

/** Memoised GET. A failed request is forgotten, so the next attempt retries. Random draws aren't cached. */
function get<T>(path: string, memo = true): Promise<T> {
  if (!memo) return request<T>(path);
  let p = cache.get(path) as Promise<T> | undefined;
  if (!p) {
    p = request<T>(path);
    cache.set(path, p);
    p.catch(() => cache.delete(path));
  }
  return p;
}

// ---------------------------------------------------------------------------
// Cards
// ---------------------------------------------------------------------------

interface RawImages {
  small: string;
  normal: string;
  art_crop: string;
}

interface RawFace {
  name: string;
  mana_cost?: string;
  type_line?: string;
  oracle_text?: string;
  flavor_text?: string;
  power?: string;
  toughness?: string;
  loyalty?: string;
  defense?: string;
  artist?: string;
  colors?: string[];
  image_uris?: RawImages;
}

export interface RawCard extends RawFace {
  id: string;
  set: string;
  set_name: string;
  collector_number: string;
  rarity: string;
  released_at: string;
  layout: string;
  color_identity: string[];
  card_faces?: RawFace[];
  card_back_id?: string;
  scryfall_uri: string;
}

function face(f: RawFace, card: RawCard): Face {
  const img = f.image_uris ?? card.image_uris;
  return {
    name: f.name,
    manaCost: f.mana_cost ?? '',
    typeLine: f.type_line ?? '',
    oracleText: f.oracle_text ?? '',
    flavorText: f.flavor_text ?? '',
    power: f.power ?? null,
    toughness: f.toughness ?? null,
    loyalty: f.loyalty ?? null,
    defense: f.defense ?? null,
    artist: f.artist ?? card.artist ?? '',
    colors: f.colors ?? card.colors ?? [],
    image: img ? { card: img.normal, art: img.art_crop, small: img.small } : null,
  };
}

/**
 * Flattens Scryfall's card object. Split, adventure and flip cards have `card_faces` but one image;
 * double-faced ones have an image per face.
 */
export function toCard(raw: RawCard): Card {
  const faces = raw.card_faces?.length ? raw.card_faces.map((f) => face(f, raw)) : [face(raw, raw)];
  return {
    id: raw.id,
    name: raw.name,
    set: raw.set,
    setName: raw.set_name,
    collectorNumber: raw.collector_number,
    rarity: raw.rarity,
    releasedAt: raw.released_at,
    layout: raw.layout,
    colorIdentity: raw.color_identity,
    faces,
    twoSided: !!raw.card_faces && raw.card_faces.length > 1 && raw.card_faces.every((f) => f.image_uris),
    cardBackId: raw.card_back_id ?? DEFAULT_BACK_ID,
    scryfallUri: raw.scryfall_uri,
  };
}

const remember = (raw: RawCard) => {
  const card = toCard(raw);
  // Seed the by-printing cache, so going back to a card found by name or at random is free.
  cache.set(`/cards/${card.set}/${encodeURIComponent(card.collectorNumber)}`, Promise.resolve(raw));
  return card;
};

export const cardByName = (name: string) => get<RawCard>(`/cards/named?fuzzy=${encodeURIComponent(name)}`).then(remember);

export const cardByPrinting = (set: string, number: string) =>
  get<RawCard>(`/cards/${encodeURIComponent(set.toLowerCase())}/${encodeURIComponent(number)}`).then(remember);

export const randomCard = (query: string) =>
  get<RawCard>(`/cards/random${query.trim() ? `?q=${encodeURIComponent(query.trim())}` : ''}`, false).then(remember);

export async function autocomplete(text: string): Promise<string[]> {
  if (text.trim().length < 2) return [];
  const r = await get<{ data: string[] }>(`/cards/autocomplete?q=${encodeURIComponent(text.trim())}`);
  return r.data;
}

// ---------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------

export type TextPart = { text: string } | { symbol: string };

/** Splits rules text into plain runs and mana symbols: "{T}: Add {G}." → [{T}, ": Add ", {G}, "."]. */
export function parseSymbols(text: string): TextPart[] {
  const parts: TextPart[] = [];
  const re = /\{[^}]+\}/g;
  let last = 0;
  for (let m = re.exec(text); m; m = re.exec(text)) {
    if (m.index > last) parts.push({ text: text.slice(last, m.index) });
    parts.push({ symbol: m[0] });
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push({ text: text.slice(last) });
  return parts;
}

/** "fdn/763" or "Shivan Dragon" from a URL hash. */
export function parseHash(hash: string): { set: string; number: string } | { name: string } | null {
  const h = decodeURIComponent(hash.replace(/^#/, '')).trim();
  if (!h) return null;
  const m = /^([a-z0-9]{2,6})\/([^/]+)$/i.exec(h);
  return m ? { set: m[1], number: m[2] } : { name: h };
}

/** The next printing in the set: "12" → "13". Null for collector numbers that aren't plain numbers. */
export function stepNumber(number: string, step: number): string | null {
  if (!/^\d+$/.test(number)) return null;
  const n = Number(number) + step;
  return n >= 1 ? String(n) : null;
}
