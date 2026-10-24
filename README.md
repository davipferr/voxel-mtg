# Voxel MTG

Magic: The Gathering cards rebuilt in 3D voxels, with the subject of the art popping out of the card.
The card becomes a voxel slab. An in-browser segmentation model cuts the creature (or whatever the art
is of) out of the illustration, and that cutout is inflated into a rounded voxel figure that rises out of
the card. Built with **TypeScript + Vite + Three.js** on **[Scryfall](https://scryfall.com/docs/api)**
(free, no key, CORS-enabled, and so are its images) and **[transformers.js](https://huggingface.co/docs/transformers.js)**.

It's the sibling of [voxel-pokedex](../voxel-pokedex), which does the same with Pokémon sprites. The
voxel engine (transitions, GPU animation, poke ripple) is shared; what's new is getting depth out of a
card, which has no transparent background to give it a silhouette.

```bash
npm install
npm run dev
```

`npm test` runs the unit tests. `npm run build` type-checks and builds to `dist/`, which deploys to GitHub
Pages through [.github/workflows/ci.yml](.github/workflows/ci.yml). Enable Pages with source "GitHub Actions"
in the repo settings.

## Layouts

| | |
|---|---|
| **Relief** | The card stands upright and the subject bulges out of the art box towards you. The back of the card is the Magic card back, or the other face on a double-faced card. |
| **Summon** | The card lies on the table and the subject stands on it, full size and rounded on both sides, above the spot in the art it came from. |

## Controls

| | |
|---|---|
| <kbd>←</kbd> <kbd>→</kbd> | Previous / next card in the same set |
| <kbd>R</kbd> | Random card, drawn from a Scryfall search (`t:god` by default; change it in ⚙) |
| <kbd>T</kbd> | Turn a double-faced card over |
| <kbd>L</kbd> | Relief / Summon |
| <kbd>B</kbd> | Burst: explode and reassemble |
| <kbd>A</kbd> | Turntable on / off |
| <kbd>/</kbd> | Search by name (with Scryfall's autocomplete) |
| <kbd>F</kbd> | Fullscreen |
| Click | Poke the card: a ripple runs through the voxels |

The URL keeps the printing (`#fdn/763`), or takes a name (`#Shivan Dragon`), so links are shareable.

## How it works

| Piece | File | Notes |
|---|---|---|
| Scryfall client | [src/data/scryfall.ts](src/data/scryfall.ts) | `/cards/named?fuzzy=`, `/cards/{set}/{number}`, `/cards/random?q=`, `/cards/autocomplete`. One request at a time, 100 ms apart, memoised, as Scryfall asks. Double-faced cards have an image per face; split and adventure cards share one. |
| Cutout | [src/segment/worker.ts](src/segment/worker.ts) | BRIA RMBG-1.4, a salient-object segmenter, run on Scryfall's `art_crop` in a Web Worker. WebGPU with fp16 weights (88 MB, ~1 s per card) on GPUs that support `shader-f16`, int8 WebAssembly otherwise (44 MB, ~5 s). The model is cached by the browser after the first download, and each cutout is memoised for the session. |
| Finding the art | [src/voxel/raster.ts](src/voxel/raster.ts) | Every frame puts the art somewhere different (old border, modern, borderless, showcase). The art crop is cut from the same scan as the card, so **template matching** finds it: an exhaustive search over size and position at 64 px wide, then two refinements at double the resolution. |
| Resampling | [src/voxel/raster.ts](src/voxel/raster.ts) | Area-averaged (box filter with fractional coverage), with colour weighted by alpha so a cutout's edge doesn't pick up the background's colour. The card is ~120 voxels wide by default; full resolution would be a million voxels and look like noise. |
| Mask clean-up | [src/voxel/raster.ts](src/voxel/raster.ts) | Connected-component labelling drops background specks and fills pinholes, which would otherwise become floating voxels and pits. A cutout covering under 1% or over 90% of the art is treated as "no clear subject". |
| Inflation | [src/voxel/voxelize.ts](src/voxel/voxelize.ts) | Same as voxel-pokedex: an exact Euclidean distance transform gives each pixel its distance to the outline, and the column height is `3·√(d − ½)`. Two-sided for Summon; one-sided (bulging towards the viewer only, a bit taller) for Relief. Only voxels with an exposed face are kept, and ambient occlusion is baked into the colours. |
| Card model | [src/voxel/card.ts](src/voxel/card.ts) | The card slab with rounded corners (63 × 88 mm with 3.2 mm corners), then the subject. The card's voxels always come first, in the same order, and each subject voxel records the point on the card it comes from, which is what the rise animation needs. |
| Rise | [src/scene/morph.ts](src/scene/morph.ts) | When the cutout arrives after the card is on screen, a `rise` transition pairs the two models voxel for voxel: the card stays put and the subject grows out of the art, the shortest trips first. |
| Depth | [src/segment/worker.ts](src/segment/worker.ts) | Depth Anything V2 Small, run on the art crop in the same worker as the cutout (50 MB fp16 on WebGPU, 27 MB int8 otherwise). It gives relative depth: which parts of the picture are nearer, not how far. |
| Sculpt | [src/sculpt/mesh.ts](src/sculpt/mesh.ts) | The default style (<kbd>V</kbd> switches to voxels). A smooth closed mesh on a grid twice as fine as the voxels: inflation for the thickness, plus the depth map bringing nearer parts forward. Outline cells with three corners inside get one triangle, so the edge isn't a staircase, and walls close the mesh. Painted with the full-resolution art. |
| Rendering | [src/scene/voxels.ts](src/scene/voxels.ts) | One `InstancedMesh` of cubes, with every flight evaluated in the vertex shader from a single time uniform (see voxel-pokedex). |

## Licences

Card data and images come from Scryfall. Magic: The Gathering and its card art are © Wizards of the
Coast. This is unofficial fan content, not affiliated with or endorsed by Wizards. RMBG-1.4 is
[licensed by BRIA](https://huggingface.co/briaai/RMBG-1.4) for non-commercial use. For anything
commercial, swap in an MIT-licensed model such as `onnx-community/BiRefNet_lite-ONNX` (a larger
download). The model id is one constant in the worker.
