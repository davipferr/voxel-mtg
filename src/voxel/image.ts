import type { Pixels } from './raster';

const cache = new Map<string, Promise<Pixels>>();

/**
 * Loads an image and reads its pixels. `crossOrigin = 'anonymous'` makes the browser request it with
 * CORS; without it a cross-origin image still displays, but the canvas is "tainted" and
 * `getImageData` throws.
 */
export function loadPixels(url: string): Promise<Pixels> {
  let p = cache.get(url);
  if (!p) {
    p = new Promise<Pixels>((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.decoding = 'async';
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
        ctx.drawImage(img, 0, 0);
        const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
        resolve({ data, width, height });
      };
      img.onerror = () => reject(new Error('Could not load the card image'));
      img.src = url;
    });
    cache.set(url, p);
    p.catch(() => cache.delete(url));
  }
  return p;
}
