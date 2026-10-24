import * as THREE from 'three';

import type { SculptMesh } from '../sculpt/mesh';
import { CORNER, type Layout } from '../voxel/card';
import type { Pixels } from '../voxel/raster';

/**
 * The sculpt style: the card as a thin textured slab with rounded corners, and the subject as one
 * smooth mesh painted with the art. Same world space as the voxel card of the same size, so the camera
 * framing and pedestal work for both. The subject grows out of the card when it arrives.
 */

const GROW_SECONDS = 0.9;
const BACK = 0x52382a;

const ease = (t: number) => 1 - Math.pow(1 - t, 3);

function roundedCard(w: number, h: number, r: number): THREE.Shape {
  const s = new THREE.Shape();
  const x0 = -w / 2;
  const x1 = w / 2;
  s.moveTo(x0 + r, 0);
  s.lineTo(x1 - r, 0);
  s.absarc(x1 - r, r, r, -Math.PI / 2, 0, false);
  s.lineTo(x1, h - r);
  s.absarc(x1 - r, h - r, r, 0, Math.PI / 2, false);
  s.lineTo(x0 + r, h);
  s.absarc(x0 + r, h - r, r, Math.PI / 2, Math.PI, false);
  s.lineTo(x0, r);
  s.absarc(x0 + r, r, r, Math.PI, Math.PI * 1.5, false);
  return s;
}

/** A face of the card in its own plane, z = 0, with the image's top row at the card's top. */
function face(shape: THREE.Shape, w: number, h: number): THREE.ShapeGeometry {
  const g = new THREE.ShapeGeometry(shape, 12);
  const p = g.getAttribute('position');
  const uv = g.getAttribute('uv');
  for (let i = 0; i < p.count; i++) uv.setXY(i, (p.getX(i) + w / 2) / w, 1 - p.getY(i) / h);
  return g;
}

export class SculptView {
  readonly group = new THREE.Group();
  private readonly textures = new Map<Pixels, THREE.Texture>();
  private card: THREE.Group | null = null;
  private cardKey = '';
  private cardBox = new THREE.Box3();
  /** Scales the subject out of the card: along z for a relief, y for a figure. */
  private readonly pivot = new THREE.Group();
  private subject: THREE.Mesh | null = null;
  private subjectBox = new THREE.Box3();
  private growAxis: 'y' | 'z' = 'z';
  private growTime = GROW_SECONDS;
  private used: (Pixels | null)[] = [];

  constructor() {
    this.group.add(this.pivot);
  }

  /** Triangles in the subject. */
  get triangles(): number {
    return this.subject ? (this.subject.geometry.index?.count ?? 0) / 3 : 0;
  }

  /** The card, `width` × `height` voxels, upright (relief) or lying face up (summon). */
  setCard(front: Pixels, back: Pixels | null, width: number, height: number, layout: Layout): void {
    const key = `${width} ${height} ${layout}`;
    if (this.card && key === this.cardKey && this.used[0] === front && this.used[1] === back) return;
    this.dropCard();
    this.cardKey = key;

    const shape = roundedCard(width, height, CORNER * width);
    // As thick as the voxel card: two voxels upright, one lying down.
    const thickness = layout === 'relief' ? 2 : 1;
    const card = new THREE.Group();

    const frontMesh = new THREE.Mesh(face(shape, width, height), new THREE.MeshStandardMaterial({ map: this.texture(front), roughness: 0.55 }));
    frontMesh.position.z = 0.01;
    const backMaterial = back ? new THREE.MeshStandardMaterial({ map: this.texture(back), roughness: 0.6 }) : new THREE.MeshStandardMaterial({ color: BACK, roughness: 0.7 });
    const backMesh = new THREE.Mesh(face(shape, width, height), backMaterial);
    // Turned round, so it reads the right way from behind.
    backMesh.rotation.y = Math.PI;
    backMesh.position.z = -thickness - 0.01;
    // The extrusion's caps sit under the faces; only its sides show.
    const hidden = new THREE.MeshBasicMaterial({ visible: false });
    const edge = new THREE.Mesh(
      new THREE.ExtrudeGeometry(shape, { depth: thickness, bevelEnabled: false, curveSegments: 12 }),
      [hidden, new THREE.MeshStandardMaterial({ color: 0x15130f, roughness: 0.8 })],
    );
    edge.position.z = -thickness;
    card.add(frontMesh, backMesh, edge);
    for (const m of [frontMesh, backMesh, edge]) {
      m.castShadow = true;
      m.receiveShadow = true;
    }

    if (layout === 'relief') {
      // Front at z = 0.5, like the voxel card's front face.
      card.position.z = 0.5;
    } else {
      // Face up with the top edge away from the viewer, front at y = 1.
      card.rotation.x = -Math.PI / 2;
      card.position.set(0, 1, height / 2);
    }
    this.card = card;
    this.group.add(card);
    card.updateMatrixWorld(true);
    this.cardBox.setFromObject(card);
    this.used[0] = front;
    this.used[1] = back;
    this.prune();
  }

  /** The subject, painted with `art`; `grow` raises it out of the card, otherwise it's there at once. */
  setSubject(mesh: SculptMesh | null, art: Pixels | null, layout: Layout, grow: boolean): void {
    if (this.subject) {
      this.pivot.remove(this.subject);
      this.subject.geometry.dispose();
      (this.subject.material as THREE.Material).dispose();
      this.subject = null;
    }
    this.used[2] = null;
    this.subjectBox.makeEmpty();
    if (mesh && art && mesh.pixels) {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3));
      g.setAttribute('uv', new THREE.BufferAttribute(mesh.uvs, 2));
      g.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
      g.computeVertexNormals();
      const subject = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ map: this.texture(art), roughness: 0.5, side: THREE.DoubleSide }));
      subject.castShadow = true;
      subject.receiveShadow = true;
      this.subjectBox.set(new THREE.Vector3(...mesh.min), new THREE.Vector3(...mesh.max));

      // Grow from the card's surface: z = 0.5 upright, y = 1 lying down.
      this.growAxis = layout === 'relief' ? 'z' : 'y';
      const base = layout === 'relief' ? 0.5 : 1;
      this.pivot.position.set(0, 0, 0);
      this.pivot.position[this.growAxis] = base;
      subject.position.set(0, 0, 0);
      subject.position[this.growAxis] = -base;
      this.pivot.add(subject);
      this.subject = subject;
      this.used[2] = art;
      this.growTime = grow ? 0 : GROW_SECONDS;
      this.applyGrow();
    }
    this.prune();
  }

  /** Where everything will be once it has grown. */
  bounds(): { min: [number, number, number]; max: [number, number, number] } {
    const b = this.cardBox.clone().union(this.subjectBox);
    return { min: b.min.toArray(), max: b.max.toArray() };
  }

  update(dt: number): void {
    if (this.growTime >= GROW_SECONDS) return;
    this.growTime = Math.min(GROW_SECONDS, this.growTime + dt);
    this.applyGrow();
  }

  private applyGrow(): void {
    this.pivot.scale.set(1, 1, 1);
    this.pivot.scale[this.growAxis] = Math.max(0.001, ease(this.growTime / GROW_SECONDS));
  }

  private dropCard(): void {
    if (!this.card) return;
    this.group.remove(this.card);
    this.card.traverse((o) => {
      if (!(o instanceof THREE.Mesh)) return;
      o.geometry.dispose();
      for (const m of [o.material].flat()) m.dispose();
    });
    this.card = null;
  }

  private texture(px: Pixels): THREE.Texture {
    let t = this.textures.get(px);
    if (!t) {
      const data = px.data instanceof Uint8Array ? px.data : new Uint8Array(px.data.buffer, px.data.byteOffset, px.data.byteLength);
      t = new THREE.DataTexture(data, px.width, px.height, THREE.RGBAFormat);
      t.colorSpace = THREE.SRGBColorSpace;
      t.generateMipmaps = true;
      t.minFilter = THREE.LinearMipmapLinearFilter;
      t.magFilter = THREE.LinearFilter;
      t.anisotropy = 4;
      t.needsUpdate = true;
      this.textures.set(px, t);
    }
    return t;
  }

  /** Frees the textures of images no longer shown. */
  private prune(): void {
    for (const [px, t] of this.textures) {
      if (this.used.includes(px)) continue;
      t.dispose();
      this.textures.delete(px);
    }
  }
}
