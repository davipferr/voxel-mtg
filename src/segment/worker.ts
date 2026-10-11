/**
 * Background removal, off the main thread.
 *
 * BRIA's RMBG-1.4 (a salient-object segmenter) through transformers.js and ONNX Runtime Web. The first
 * run downloads the model from the Hugging Face hub, and transformers.js keeps it in the browser's
 * Cache Storage after that. Measured on one art crop:
 *
 * | Device, weights      | Download | Per image |
 * |----------------------|----------|-----------|
 * | WebGPU, fp16         | 88 MB    | 0.6–1.2 s |
 * | WebAssembly, int8    | 44 MB    | ~5 s      |
 * | WebGPU, int8         | 44 MB    | 7–11 s (int8 ops fall back to the CPU) |
 *
 * So: fp16 on a GPU that can do 16-bit floats in shaders, int8 on the CPU otherwise.
 */

import { AutoModel, AutoProcessor, RawImage, env, type PreTrainedModel, type Processor } from '@huggingface/transformers';

const MODEL = 'briaai/RMBG-1.4';

env.allowLocalModels = false;

export type Request = { id: number; url: string };
export type Response =
  | { type: 'progress'; loaded: number; total: number }
  | { type: 'ready'; device: string }
  | { type: 'done'; id: number; width: number; height: number; mask: Uint8Array }
  | { type: 'error'; id: number; message: string };

const post = (msg: Response, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(msg, transfer);

// Bytes per file, for one overall progress figure.
const files = new Map<string, { loaded: number; total: number }>();
function onProgress(p: { status: string; file?: string; loaded?: number; total?: number }): void {
  if (p.status !== 'progress' || !p.file || !p.total) return;
  files.set(p.file, { loaded: p.loaded ?? 0, total: p.total });
  let loaded = 0;
  let total = 0;
  for (const f of files.values()) {
    loaded += f.loaded;
    total += f.total;
  }
  post({ type: 'progress', loaded, total });
}

/** A WebGPU adapter that supports `shader-f16`. */
async function hasF16GPU(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<{ features: Set<string> } | null> } }).gpu;
  try {
    const adapter = gpu && (await gpu.requestAdapter());
    return !!adapter?.features.has('shader-f16');
  } catch {
    return false;
  }
}

interface Session {
  model: PreTrainedModel;
  processor: Processor;
  device: string;
}

async function open(device: 'webgpu' | 'wasm'): Promise<Session> {
  const [model, processor] = await Promise.all([
    // The repo's config.json is written for Python; `custom` tells transformers.js not to look for a
    // known architecture and just run the ONNX graph.
    AutoModel.from_pretrained(MODEL, {
      config: { model_type: 'custom' } as never,
      device,
      dtype: device === 'webgpu' ? 'fp16' : 'q8',
      progress_callback: onProgress,
    }),
    AutoProcessor.from_pretrained(MODEL, { progress_callback: onProgress }),
  ]);
  return { model, processor, device };
}

let session: Promise<Session> | null = null;

function load(): Promise<Session> {
  session ??= (async () => {
    const s = (await hasF16GPU()) ? await open('webgpu').catch(() => open('wasm')) : await open('wasm');
    post({ type: 'ready', device: s.device });
    return s;
  })();
  session.catch(() => (session = null));
  return session;
}

async function segment(s: Session, image: RawImage): Promise<RawImage> {
  const { pixel_values } = await s.processor(image);
  const { output } = await s.model({ input: pixel_values });
  return RawImage.fromTensor(output[0].mul(255).to('uint8')).resize(image.width, image.height);
}

self.addEventListener('message', async (e: MessageEvent<Request>) => {
  const { id, url } = e.data;
  try {
    let s = await load();
    const image = await RawImage.fromURL(url);
    let mask: RawImage;
    try {
      mask = await segment(s, image);
    } catch (err) {
      // Some GPUs fail at run time rather than at load time; fall back to the CPU once.
      if (s.device !== 'webgpu') throw err;
      session = open('wasm');
      s = await session;
      post({ type: 'ready', device: s.device });
      mask = await segment(s, image);
    }
    const data = new Uint8Array(mask.data);
    post({ type: 'done', id, width: mask.width, height: mask.height, mask: data }, [data.buffer]);
  } catch (err) {
    post({ type: 'error', id, message: err instanceof Error ? err.message : String(err) });
  }
});
