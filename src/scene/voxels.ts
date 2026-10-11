import * as THREE from 'three';

import { SPRING_GLSL, planTransition, restingPlan, sampleAt, seed, snapshotOf, type Plan, type Snapshot, type TransitionMode } from './morph';
import type { VoxelModel } from '../voxel/card';

/**
 * Every voxel is one instance of a single cube. The instance matrix holds where the voxel ends up,
 * which is what raycasting sees, and the vertex shader adds the offset of wherever its flight has
 * got to. Lighting, shadows and picking all come from three.js; only the motion is custom.
 */

const SHADER_COMMON = /* glsl */ `
attribute vec3 aFrom;
attribute vec3 aMid;
attribute vec3 aFromColor;
attribute vec3 aToColor;
attribute vec2 aScale;
attribute float aDelay;
attribute vec4 aSpin;

uniform float uTime;
uniform float uDuration;
uniform float uPop;
uniform vec3 uPokeOrigin;
uniform float uPokeAge;

${SPRING_GLSL}

vec3 rotateAxis(vec3 v, vec3 axis, float a) {
  // Rodrigues' rotation formula.
  return v * cos(a) + cross(axis, v) * sin(a) + axis * dot(axis, v) * (1.0 - cos(a));
}

struct VoxState {
  vec3 offset;
  float scale;
  float angle;
  vec3 color;
};

// Mirrors sampleAt() in morph.ts.
VoxState voxState() {
  vec3 to = instanceMatrix[3].xyz;
  float t = clamp((uTime - aDelay) / uDuration, 0.0, 1.0);
  float e = voxSpring(t);
  float u = 1.0 - e;
  vec3 ctrl = 0.5 * (aFrom + to) + aMid;
  vec3 p = u * u * aFrom + 2.0 * u * e * ctrl + e * e * to;

  // A poke sends a ring outward from the hit point: a damped sine that arrives later the further
  // away a voxel is.
  vec3 rel = p - uPokeOrigin;
  float d = length(rel);
  float age = uPokeAge - d / 45.0;
  if (age > 0.0 && age < 2.0) {
    p += rel / max(d, 0.001) * 2.4 * exp(-age * 3.5) * sin(age * 16.0) * exp(-d / 28.0);
  }

  VoxState s;
  s.offset = p - to;
  s.scale = mix(aScale.x, aScale.y, smoothstep(0.0, 1.0, t)) * (1.0 + uPop * sin(PI * t));
  s.angle = aSpin.w * sin(PI * t);
  s.color = mix(aFromColor, aToColor, smoothstep(0.1, 0.9, t));
  return s;
}
`;

type Uniforms = Record<string, THREE.IUniform>;

/** Injects the flight into a built-in material's vertex shader (and, for the visible one, its colour). */
function patch(material: THREE.Material, uniforms: Uniforms, withColor: boolean): void {
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${SHADER_COMMON}\n${withColor ? 'varying vec3 vVoxColor;\nvarying vec3 vLocal;' : ''}`)
      .replace('void main() {', 'void main() {\n  VoxState vs = voxState();')
      .replace(
        '#include <begin_vertex>',
        `vec3 transformed = rotateAxis(position * vs.scale, aSpin.xyz, vs.angle) + vs.offset;
        ${withColor ? 'vVoxColor = vs.color;\nvLocal = position;' : ''}`,
      );
    if (!withColor) return;
    shader.vertexShader = shader.vertexShader.replace(
      '#include <beginnormal_vertex>',
      '#include <beginnormal_vertex>\nobjectNormal = rotateAxis(objectNormal, aSpin.xyz, vs.angle);',
    );
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vVoxColor;\nvarying vec3 vLocal;')
      .replace(
        'vec4 diffuseColor = vec4( diffuse, opacity );',
        /* glsl */ `
        // Darken towards the cube's edges, so neighbouring voxels of one colour still read as blocks.
        // On a face one coordinate is ±0.5; the edge is where the second largest one gets there too.
        vec3 q = abs(vLocal) * 2.0;
        float second = q.x + q.y + q.z - max(q.x, max(q.y, q.z)) - min(q.x, min(q.y, q.z));
        float edge = smoothstep(0.78, 0.98, second);
        vec4 diffuseColor = vec4(diffuse * vVoxColor * (1.0 - 0.3 * edge), opacity);`,
      );
  };
}

const srgbToLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));

export class VoxelCloud {
  readonly mesh: THREE.InstancedMesh;
  private readonly uniforms = {
    uTime: { value: 0 },
    uDuration: { value: 1 },
    uPop: { value: 0 },
    uPokeOrigin: { value: new THREE.Vector3() },
    uPokeAge: { value: 99 },
  };
  private capacity = 0;
  private plan: Plan | null = null;
  private time = 0;
  private moving = false;
  private waiters: (() => void)[] = [];

  constructor() {
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const material = new THREE.MeshStandardMaterial({ roughness: 0.62, metalness: 0.0 });
    patch(material, this.uniforms, true);
    const depth = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking });
    patch(depth, this.uniforms, false);

    this.mesh = new THREE.InstancedMesh(geometry, material, 1);
    this.mesh.customDepthMaterial = depth;
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    // Mid-flight voxels leave the resting bounds, so the bounds can't be used for culling.
    this.mesh.frustumCulled = false;
    this.mesh.count = 0;
    this.reserve(16_384);
  }

  /** True while a transition is playing. */
  get animating(): boolean {
    return this.moving;
  }

  /** Transitions to `model`. The first model grows out of the floor. `from` overrides where it starts. */
  show(model: VoxelModel, mode: TransitionMode, from?: Snapshot): void {
    from ??= !this.plan ? seed(0, 0.5, 0) : this.moving ? sampleAt(this.plan, this.time) : this.plan.target;
    this.upload(planTransition(from, snapshotOf(model), mode));
    this.time = 0;
    this.moving = true;
  }

  /** Resolves when nothing is moving: at once, or when the transition playing now has landed. */
  settled(): Promise<void> {
    if (!this.moving) return Promise.resolve();
    return new Promise((resolve) => this.waiters.push(resolve));
  }

  /** Sends a ripple out from a point on the model (world space; the mesh never moves). */
  poke(point: THREE.Vector3): void {
    this.uniforms.uPokeOrigin.value.copy(point);
    this.uniforms.uPokeAge.value = 0;
  }

  /** The resting voxel under the ray, if any. Mid-flight voxels aren't where the instance matrix says. */
  pick(raycaster: THREE.Raycaster): THREE.Vector3 | null {
    if (this.moving || this.mesh.count === 0) return null;
    return raycaster.intersectObject(this.mesh, false)[0]?.point ?? null;
  }

  update(dt: number): void {
    this.uniforms.uPokeAge.value += dt;
    if (!this.plan || !this.moving) return;
    this.time += dt;
    this.uniforms.uTime.value = this.time;
    if (this.time >= this.plan.end) {
      // Settle: drop the duplicates that shrank to nothing and stop the clock.
      this.upload(restingPlan(this.plan.target));
      this.moving = false;
      for (const resolve of this.waiters.splice(0)) resolve();
    }
  }

  private reserve(n: number): void {
    if (n <= this.capacity) return;
    let cap = Math.max(this.capacity, 1);
    while (cap < n) cap *= 2;
    this.capacity = cap;

    const matrices = new Float32Array(cap * 16);
    for (let i = 0; i < cap; i++) matrices.set([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], i * 16);
    this.mesh.instanceMatrix = new THREE.InstancedBufferAttribute(matrices, 16);

    const g = this.mesh.geometry;
    const attr = (name: string, size: number) => g.setAttribute(name, new THREE.InstancedBufferAttribute(new Float32Array(cap * size), size));
    attr('aFrom', 3);
    attr('aMid', 3);
    attr('aFromColor', 3);
    attr('aToColor', 3);
    attr('aScale', 2);
    attr('aDelay', 1);
    attr('aSpin', 4);
  }

  private upload(plan: Plan): void {
    this.reserve(plan.count);
    this.plan = plan;
    const g = this.mesh.geometry;
    const write = (name: string, data: Float32Array, convert = false) => {
      const a = g.getAttribute(name) as THREE.InstancedBufferAttribute;
      const arr = a.array as Float32Array;
      if (convert) for (let i = 0; i < data.length; i++) arr[i] = srgbToLinear(data[i]);
      else arr.set(data);
      a.clearUpdateRanges();
      a.addUpdateRange(0, plan.count * a.itemSize);
      a.needsUpdate = true;
    };
    write('aFrom', plan.from);
    write('aMid', plan.mid);
    write('aFromColor', plan.fromColor, true);
    write('aToColor', plan.toColor, true);
    write('aScale', plan.scale);
    write('aDelay', plan.delay);
    write('aSpin', plan.spin);

    const m = this.mesh.instanceMatrix;
    const arr = m.array as Float32Array;
    for (let i = 0; i < plan.count; i++) {
      arr[i * 16 + 12] = plan.to[i * 3];
      arr[i * 16 + 13] = plan.to[i * 3 + 1];
      arr[i * 16 + 14] = plan.to[i * 3 + 2];
    }
    m.clearUpdateRanges();
    m.addUpdateRange(0, plan.count * 16);
    m.needsUpdate = true;

    this.mesh.count = plan.count;
    // Raycasting checks cached bounds before testing instances; they belong to the old model.
    this.mesh.boundingSphere = null;
    this.mesh.boundingBox = null;
    this.uniforms.uDuration.value = plan.duration;
    this.uniforms.uPop.value = plan.pop;
    this.uniforms.uTime.value = 0;
  }
}
