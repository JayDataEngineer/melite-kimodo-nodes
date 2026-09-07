# melite-kimodo-nodes

Tech Noir humanoid custom nodes for ComfyUI — ONE install unit for the
body + motion family (2026-09-07 merge: the `melite-anny-estimator-nodes`
pack folded in — shared torch/`melite_model_base` spine, one registry
entry; node type names unchanged).

## Nodes

Motion (the NVIDIA-package wrap — THE kimodo implementation):

- `KimodoLoader`
- `KimodoTextToPose`

Anny (image → Anny, the Apache-2.0 all-age parametric human model —
both engines of the license-clean path, in `nodes_anny.py`):

- `MultiHMR2ImageToAnny` — naver/multi-hmr2 regression, in-process,
  VRAM-tracked (multi-person params `.pkl` + meshes `.glb` + render)
- `AnnyFitOptimize` — naver/anny-fit camera-space optimization,
  out-of-process over `ANNYFIT_SRC`/`ANNYFIT_PYTHON`
- `AnnySelectPerson` / `AnnyGroundBody` / `Sam3dBodyImageToAnny` /
  `AnnyParamsToGlb` — graph-side selectors, grounding, preset instantiate

Both anny upstreams are NAVER non-commercial research licenses;
Anny itself is Apache 2.0.

## Install

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/JayDataEngineer/melite-kimodo-nodes.git
```

Restart ComfyUI.

## Provenance

Published from the inference estate (`inference.cpp` repo, `plugins/comfyui/custom_nodes/melite-kimodo-nodes`) on 2026-09-01.

## License

MIT — see LICENSE.
