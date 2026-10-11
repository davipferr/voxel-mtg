/**
 * User settings, bound to `[data-setting]` inputs and remembered in localStorage (a per-viewer
 * convenience: it can be missing or throw in private mode, and the defaults apply then).
 */

import type { TransitionMode } from '../scene/morph';
import type { Layout } from '../voxel/card';

export interface Settings {
  layout: Layout;
  /** Cut the subject out of the art and inflate it. */
  popOut: boolean;
  transition: Extract<TransitionMode, 'explode' | 'morph'>;
  /** Card width in voxels. */
  detail: number;
  inflate: number;
  /** Scryfall search the random button draws from. */
  randomQuery: string;
  autoRotate: boolean;
}

export const DEFAULTS: Settings = {
  layout: 'relief',
  popOut: true,
  transition: 'explode',
  detail: 120,
  inflate: 1,
  randomQuery: 't:creature',
  autoRotate: true,
};

const KEY = 'voxel-mtg:settings';

export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return { ...DEFAULTS, ...(JSON.parse(raw) as Partial<Settings>) };
  } catch {
    // ignore
  }
  return { ...DEFAULTS };
}

export function saveSettings(s: Settings): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    // ignore
  }
}

type Input = HTMLInputElement | HTMLSelectElement;

/**
 * Two-way binds inputs with `data-setting="key"` to `settings`; calls `onChange` after each edit.
 * Returns `sync`, which pushes `settings` back into the inputs after a change made in code.
 */
export function bindSettings(root: ParentNode, settings: Settings, onChange: (key: keyof Settings) => void) {
  const inputs = [...root.querySelectorAll<Input>('[data-setting]')];
  const sync = () => {
    for (const input of inputs) {
      const key = input.dataset.setting as keyof Settings;
      if (input instanceof HTMLInputElement && input.type === 'checkbox') input.checked = settings[key] as boolean;
      else input.value = String(settings[key]);
    }
  };
  for (const input of inputs) {
    // Sliders fire `input` continuously while dragging, and a text box on every keystroke; rebuilding
    // that often would restart the transition dozens of times, so they report on release / commit.
    const isText = input instanceof HTMLInputElement && (input.type === 'range' || input.type === 'text');
    input.addEventListener(isText ? 'change' : 'input', () => {
      const key = input.dataset.setting as keyof Settings;
      const s = settings as unknown as Record<string, unknown>;
      if (input instanceof HTMLInputElement && input.type === 'checkbox') s[key] = input.checked;
      else if (input instanceof HTMLInputElement && input.type === 'range') s[key] = Number(input.value);
      else s[key] = input.value;
      saveSettings(settings);
      onChange(key);
    });
  }
  sync();
  return sync;
}
