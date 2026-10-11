/**
 * Cuts the subject out of a card's art, using the segmentation worker. Results are memoised per art
 * URL for the session, so revisiting a card or changing a setting doesn't run the model again.
 */

import type { Response } from './worker';
import { loadPixels } from '../voxel/image';
import { withAlpha, type Pixels } from '../voxel/raster';

export type Status =
  | { state: 'idle' }
  | { state: 'downloading'; loaded: number; total: number }
  | { state: 'ready'; device: string };

let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, { resolve: (m: { width: number; height: number; mask: Uint8Array }) => void; reject: (e: Error) => void }>();
const cache = new Map<string, Promise<Pixels>>();
const done = new Set<string>();
const listeners = new Set<(s: Status) => void>();
let status: Status = { state: 'idle' };

function setStatus(s: Status): void {
  status = s;
  for (const l of listeners) l(s);
}

/** Calls `fn` with the model's status now and on every change. */
export function onStatus(fn: (s: Status) => void): void {
  listeners.add(fn);
  fn(status);
}

function getWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  worker.addEventListener('message', (e: MessageEvent<Response>) => {
    const msg = e.data;
    if (msg.type === 'progress') setStatus({ state: 'downloading', loaded: msg.loaded, total: msg.total });
    else if (msg.type === 'ready') setStatus({ state: 'ready', device: msg.device });
    else if (msg.type === 'done') {
      pending.get(msg.id)?.resolve(msg);
      pending.delete(msg.id);
    } else if (msg.type === 'error') {
      pending.get(msg.id)?.reject(new Error(msg.message));
      pending.delete(msg.id);
    }
  });
  return worker;
}

/** True when the cutout for `url` is already made, so asking for it won't keep anyone waiting. */
export const hasCutout = (url: string) => done.has(url);

/** The art at `url` with the background made transparent. */
export function cutout(url: string): Promise<Pixels> {
  let p = cache.get(url);
  if (!p) {
    const id = nextId++;
    const mask = new Promise<{ width: number; height: number; mask: Uint8Array }>((resolve, reject) => {
      pending.set(id, { resolve, reject });
      getWorker().postMessage({ id, url });
    });
    p = Promise.all([loadPixels(url), mask]).then(([art, m]) => {
      if (m.width !== art.width || m.height !== art.height) throw new Error('The cutout came back the wrong size.');
      done.add(url);
      return withAlpha(art, m.mask);
    });
    cache.set(url, p);
    p.catch(() => cache.delete(url));
  }
  return p;
}
