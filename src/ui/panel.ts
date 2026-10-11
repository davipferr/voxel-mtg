import { COLOR_NAMES, accentFor, parseSymbols, symbolUrl, type Card, type Face } from '../data/scryfall';
import type { VoxelModel } from '../voxel/card';

const el = <K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

/** Text with its {mana symbols} drawn as Scryfall's SVGs (and kept as alt text, for copying and screen readers). */
function rich(text: string): DocumentFragment {
  const frag = document.createDocumentFragment();
  for (const part of parseSymbols(text)) {
    if ('text' in part) {
      frag.append(part.text);
      continue;
    }
    const img = el('img', 'sym');
    img.src = symbolUrl(part.symbol);
    img.alt = part.symbol;
    img.title = part.symbol;
    img.addEventListener('error', () => img.replaceWith(part.symbol), { once: true });
    frag.append(img);
  }
  return frag;
}

const stat = (f: Face) =>
  f.power !== null && f.toughness !== null
    ? `${f.power}/${f.toughness}`
    : f.loyalty !== null
      ? `Loyalty ${f.loyalty}`
      : f.defense !== null
        ? `Defense ${f.defense}`
        : '';

/** The card entry on the left. */
export class Panel {
  constructor(private readonly root: HTMLElement) {}

  private q<T extends HTMLElement = HTMLElement>(sel: string): T {
    return this.root.querySelector<T>(sel)!;
  }

  render(card: Card, faceIndex: number): void {
    const f = card.faces[faceIndex];
    this.q('#name').textContent = f.name;
    this.q('#mana').replaceChildren(rich(f.manaCost));
    this.q('#type-line').textContent = f.typeLine;

    const identity = card.colorIdentity.length ? card.colorIdentity : [''];
    const colors = identity.map((c) => {
      const b = el('span', 'pip', c ? (COLOR_NAMES[c] ?? c) : 'Colorless');
      b.style.setProperty('--c', accentFor(c ? [c] : []));
      return b;
    });
    const rarity = el('span', `rarity ${card.rarity}`, card.rarity);
    this.q('#colors').replaceChildren(...colors, rarity);

    this.q('#oracle').replaceChildren(
      ...f.oracleText.split('\n').map((line) => {
        const p = el('p');
        p.append(rich(line));
        return p;
      }),
    );
    const flavor = this.q('#flavor-text');
    flavor.textContent = f.flavorText;
    flavor.hidden = !f.flavorText;
    const st = this.q('#stat');
    st.textContent = stat(f);
    st.hidden = !st.textContent;

    this.q('#set').textContent = `${card.setName} · ${card.set.toUpperCase()} #${card.collectorNumber} · ${card.releasedAt.slice(0, 4)}`;
    this.q('#artist').textContent = f.artist ? `Illustrated by ${f.artist}` : '';

    const thumb = this.q<HTMLImageElement>('#thumb');
    if (f.image) {
      thumb.src = f.image.small;
      thumb.alt = `${f.name}, the printed card`;
    }
    thumb.hidden = !f.image;
    this.q<HTMLAnchorElement>('#scryfall-link').href = card.scryfallUri;
    this.q<HTMLAnchorElement>('#scryfall-more').href = card.scryfallUri;

    const flip = this.q<HTMLButtonElement>('#flip');
    flip.hidden = !card.twoSided;
    flip.textContent = faceIndex === 0 ? '↻ Back face' : '↻ Front face';
  }

  voxels(model: VoxelModel): void {
    const size = `${Math.round(model.max[0] - model.min[0])}×${Math.round(model.max[1] - model.min[1])}×${Math.round(model.max[2] - model.min[2])}`;
    this.q('#voxel-info').textContent =
      `${model.count.toLocaleString('en-US')} voxels · ${size}` +
      (model.subjectPixels ? ` · subject from ${model.subjectPixels.toLocaleString('en-US')} pixels` : '');
  }

  /** The line about the cutout: progress, or why there's no subject. Empty hides it. */
  cutout(text: string): void {
    const p = this.q('#cutout-info');
    p.textContent = text;
    p.hidden = !text;
  }
}
