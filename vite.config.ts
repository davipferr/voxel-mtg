import { defineConfig } from 'vite';

export default defineConfig({
  // Relative asset URLs, so the build works under any sub-path (e.g. GitHub Pages /voxel-mtg/).
  base: './',
  // The segmentation worker imports transformers.js, which needs ES-module workers.
  worker: { format: 'es' },
  // transformers.js ships prebuilt and loads ONNX Runtime's WebAssembly itself; pre-bundling it breaks that.
  optimizeDeps: { exclude: ['@huggingface/transformers'] },
  // three.js alone is ~600 kB minified, and the worker carries transformers.js; one bundle each is fine.
  build: { chunkSizeWarningLimit: 1200 },
});
