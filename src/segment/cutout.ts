/**
 * Cuts the subject out of a card's art, and estimates the art's depth, using the model worker.
 * Results are memoised per art URL for the session, so revisiting a card or changing a setting doesn't
 * run a model again.
 */

import type { Response, Task } from './worker';
import { loadPixels } from '../voxel/image';
import { withAlpha, type Pixels } from '../voxel/raster';

export type { Task };

export type Status =
  | { state: 'idle' }
  | { state: 'downloading'; loaded: number; total: number }
  | { state: 'ready'; device: string };

/** Relative inverse depth, one value per pixel, row-major from the top-left: larger is nearer. */
export interface DepthMap {
  width: number;
  height: number;
  data: Float32Array;
}

interface Raw {
  width: number;
  height: number;
  data: Uint8Array | Float32Array;
}

let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, { resolve: (r: Raw) => void; reject: (e: Error) => void }>();
const cutouts = new Map<string, Promise<Pixels>>();
const depths = new Map<string, Promise<DepthMap>>();
const done: Record<Task, Set<string>> = { cutout: new Set(), depth: new Set() };
const listeners = new Set<(task: Task, s: Status) => void>();
const status: Record<Task, Status> = { cutout: { state: 'idle' }, depth: { state: 'idle' } };

function setStatus(task: Task, s: Status): void {
  status[task] = s;
  for (const l of listeners) l(task, s);
}

/** Calls `fn` with each model's status now and on every change. */
export function onStatus(fn: (task: Task, s: Status) => void): void {
  listeners.add(fn);
  fn('cutout', status.cutout);
  fn('depth', status.depth);
}

function getWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  worker.addEventListener('message', (e: MessageEvent<Response>) => {
    const msg = e.data;
    if (msg.type === 'progress') setStatus(msg.task, { state: 'downloading', loaded: msg.loaded, total: msg.total });
    else if (msg.type === 'ready') setStatus(msg.task, { state: 'ready', device: msg.device });
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

function run(task: Task, url: string): Promise<Raw> {
  const id = nextId++;
  return new Promise<Raw>((resolve, reject) => {
    pending.set(id, { resolve, reject });
    getWorker().postMessage({ id, task, url });
  });
}

/** True when the cutout for `url` is already made, so asking for it won't keep anyone waiting. */
export const hasCutout = (url: string) => done.cutout.has(url);
/** True when the depth of `url` is already estimated. */
export const hasDepth = (url: string) => done.depth.has(url);

/** The art at `url` with the background made transparent. */
export function cutout(url: string): Promise<Pixels> {
  let p = cutouts.get(url);
  if (!p) {
    p = Promise.all([loadPixels(url), run('cutout', url)]).then(([art, m]) => {
      if (m.width !== art.width || m.height !== art.height) throw new Error('The cutout came back the wrong size.');
      done.cutout.add(url);
      return withAlpha(art, m.data as Uint8Array);
    });
    cutouts.set(url, p);
    p.catch(() => cutouts.delete(url));
  }
  return p;
}

/** How near each pixel of the art at `url` is, at the art's own size. */
export function depth(url: string): Promise<DepthMap> {
  let p = depths.get(url);
  if (!p) {
    p = run('depth', url).then((d) => {
      done.depth.add(url);
      return { width: d.width, height: d.height, data: d.data as Float32Array };
    });
    depths.set(url, p);
    p.catch(() => depths.delete(url));
  }
  return p;
}
