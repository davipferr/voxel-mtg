/**
 * The art models, off the main thread.
 *
 * Cutout: BRIA's RMBG-1.4 (a salient-object segmenter) through transformers.js and ONNX Runtime Web.
 * The first run downloads the model from the Hugging Face hub, and transformers.js keeps it in the
 * browser's Cache Storage after that. Measured on one art crop:
 *
 * | Device, weights      | Download | Per image |
 * |----------------------|----------|-----------|
 * | WebGPU, fp16         | 88 MB    | 0.6–1.2 s |
 * | WebAssembly, int8    | 44 MB    | ~5 s      |
 * | WebGPU, int8         | 44 MB    | 7–11 s (int8 ops fall back to the CPU) |
 *
 * So: fp16 on a GPU that can do 16-bit floats in shaders, int8 on the CPU otherwise.
 *
 * Depth: Depth Anything V2 Small, a monocular depth estimator, with the same choice of device and
 * weights (50 MB fp16, 27 MB int8). Its output is relative inverse depth: larger is nearer, with no
 * fixed scale, so it only says which parts of the picture stand in front of which.
 */

import { AutoModel, AutoProcessor, RawImage, env, pipeline, type Tensor } from '@huggingface/transformers';

const CUTOUT_MODEL = 'briaai/RMBG-1.4';
const DEPTH_MODEL = 'onnx-community/depth-anything-v2-small';

env.allowLocalModels = false;

export type Task = 'cutout' | 'depth';
export type Request = { id: number; task: Task; url: string };
export type Response =
  | { type: 'progress'; task: Task; loaded: number; total: number }
  | { type: 'ready'; task: Task; device: string }
  /** Cutout: one byte of opacity per pixel. Depth: one float per pixel, larger = nearer. */
  | { type: 'done'; id: number; width: number; height: number; data: Uint8Array | Float32Array }
  | { type: 'error'; id: number; message: string };

const post = (msg: Response, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(msg, transfer);

type Progress = (p: { status: string; file?: string; loaded?: number; total?: number }) => void;

/** Bytes per file, for one overall progress figure per model. */
function progressFor(task: Task): Progress {
  const files = new Map<string, { loaded: number; total: number }>();
  return (p) => {
    if (p.status !== 'progress' || !p.file || !p.total) return;
    files.set(p.file, { loaded: p.loaded ?? 0, total: p.total });
    let loaded = 0;
    let total = 0;
    for (const f of files.values()) {
      loaded += f.loaded;
      total += f.total;
    }
    post({ type: 'progress', task, loaded, total });
  };
}
const progress: Record<Task, Progress> = { cutout: progressFor('cutout'), depth: progressFor('depth') };

/** A WebGPU adapter that supports `shader-f16`. */
let f16: Promise<boolean> | null = null;
function hasF16GPU(): Promise<boolean> {
  f16 ??= (async () => {
    const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<{ features: Set<string> } | null> } }).gpu;
    try {
      const adapter = gpu && (await gpu.requestAdapter());
      return !!adapter?.features.has('shader-f16');
    } catch {
      return false;
    }
  })();
  return f16;
}

type Device = 'webgpu' | 'wasm';

interface Result {
  width: number;
  height: number;
  data: Uint8Array | Float32Array;
}

interface Session {
  device: Device;
  run(image: RawImage): Promise<Result>;
}

async function openCutout(device: Device): Promise<Session> {
  const [model, processor] = await Promise.all([
    // The repo's config.json is written for Python; `custom` tells transformers.js not to look for a
    // known architecture and just run the ONNX graph.
    AutoModel.from_pretrained(CUTOUT_MODEL, {
      config: { model_type: 'custom' } as never,
      device,
      dtype: device === 'webgpu' ? 'fp16' : 'q8',
      progress_callback: progress.cutout,
    }),
    AutoProcessor.from_pretrained(CUTOUT_MODEL, { progress_callback: progress.cutout }),
  ]);
  return {
    device,
    async run(image) {
      const { pixel_values } = await processor(image);
      const { output } = await model({ input: pixel_values });
      const mask = await RawImage.fromTensor(output[0].mul(255).to('uint8')).resize(image.width, image.height);
      return { width: mask.width, height: mask.height, data: new Uint8Array(mask.data) };
    },
  };
}

async function openDepth(device: Device): Promise<Session> {
  const estimate = await pipeline('depth-estimation', DEPTH_MODEL, {
    device,
    dtype: device === 'webgpu' ? 'fp16' : 'q8',
    progress_callback: progress.depth,
  });
  return {
    device,
    async run(image) {
      // The pipeline resizes its prediction back to the image's size.
      const { predicted_depth } = (await estimate(image)) as unknown as { predicted_depth: Tensor };
      const [height, width] = predicted_depth.dims.slice(-2);
      return { width, height, data: Float32Array.from(predicted_depth.to('float32').data as Float32Array) };
    },
  };
}

const opens: Record<Task, (device: Device) => Promise<Session>> = { cutout: openCutout, depth: openDepth };
const sessions: Partial<Record<Task, Promise<Session>>> = {};

function load(task: Task): Promise<Session> {
  let s = sessions[task];
  if (!s) {
    const open = opens[task];
    s = (async () => {
      const session = (await hasF16GPU()) ? await open('webgpu').catch(() => open('wasm')) : await open('wasm');
      post({ type: 'ready', task, device: session.device });
      return session;
    })();
    sessions[task] = s;
    s.catch(() => delete sessions[task]);
  }
  return s;
}

self.addEventListener('message', async (e: MessageEvent<Request>) => {
  const { id, task, url } = e.data;
  try {
    let s = await load(task);
    const image = await RawImage.fromURL(url);
    let result: Result;
    try {
      result = await s.run(image);
    } catch (err) {
      // Some GPUs fail at run time rather than at load time; fall back to the CPU once.
      if (s.device !== 'webgpu') throw err;
      const cpu = opens[task]('wasm');
      sessions[task] = cpu;
      s = await cpu;
      post({ type: 'ready', task, device: s.device });
      result = await s.run(image);
    }
    post({ type: 'done', id, ...result }, [result.data.buffer]);
  } catch (err) {
    post({ type: 'error', id, message: err instanceof Error ? err.message : String(err) });
  }
});
